# app/rag/retriever_qdrant.py
import re
import pandas as pd
from typing import Optional, List
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, Filter, FieldCondition, MatchValue, Range

def _e5_doc(t: str) -> str:   return f"passage: {t.strip()}"
def _e5_query(t: str) -> str: return f"query: {t.strip()}"

class ProductRetrieverQdrant:
    def __init__(self, csv_path: str, collection: str="laptops",
                 qdrant_url: str="http://127.0.0.1:6333",
                 model_name_or_path: str="intfloat/e5-base-v2",
                 recreate: bool=False, api_key: Optional[str]=None):
        self.collection = collection
        self.search_enabled = True
        self._model = None
        self._model_name = model_name_or_path

        # ---- load CSV (raise immediately if missing so app can log it) ----
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.df["_search_text"] = (
            self.df.get("brand", "").astype(str) + " " +
            self.df.get("model", self.df.get("Model", "")).astype(str) + " " +
            self.df.get("Specification", "").astype(str)
        ).str.replace(r"\s+", " ", regex=True).str.strip()
        self.df["brand_lc"] = self.df.get("brand", "").astype(str).str.lower().str.strip()
        self.df["_price_num"] = pd.to_numeric(
            self.df.get("price", "").astype(str).str.replace(r"[^\d.]", "", regex=True), errors="coerce"
        )

        # ---- connect Qdrant if possible ----
        try:
            self.client = QdrantClient(url=qdrant_url.strip(), api_key=api_key)
            self.client.get_collection(self.collection)
        except Exception as e:
            # No collection / cannot connect
            self.client = None
            self.search_enabled = False
            print(f"[WARN] Qdrant disabled; falling back to CSV search: {e}")
            if recreate:
                print("[WARN] REINDEX_ON_START=1 requested, but Qdrant is not reachable.")

        # Optional: initialize (create+upsert) if explicitly requested and reachable
        if recreate and self.search_enabled and self.client is not None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            vecs = self._model.encode([_e5_doc(t) for t in self.df["_search_text"].fillna("")],
                                      normalize_embeddings=True).astype("float32")
            self.client.recreate_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=vecs.shape[1], distance=Distance.DOT),
            )
            pts = [{"id": int(i), "vector": vecs[i].tolist(), "payload": self.df.iloc[i].to_dict()}
                   for i in range(len(self.df))]
            self.client.upsert(collection_name=self.collection, points=pts)

    # ------------------- helpers -------------------
    @staticmethod
    def _brand_filter_df(df: pd.DataFrame, brand: Optional[str]) -> pd.DataFrame:
        if not brand: return df
        return df[df["brand_lc"] == str(brand).lower().strip()]

    @staticmethod
    def _price_slice_df(df: pd.DataFrame, min_price: Optional[float], max_price: Optional[float]) -> pd.DataFrame:
        out = df
        if min_price is not None: out = out[out["_price_num"].ge(min_price, fill_value=False)]
        if max_price is not None: out = out[out["_price_num"].le(max_price, fill_value=False)]
        return out

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)

    # ------------------- fallback keyword search -------------------
    def _csv_keyword_search(self, q: str, k: int) -> pd.DataFrame:
        t = str(q or "").strip().lower()
        # split into alphanumeric tokens ≥3 chars to avoid noise
        toks: List[str] = [w for w in re.findall(r"[a-z0-9]{3,}", t)]
        if not toks:
            return pd.DataFrame([])
        # simple score = number of token hits in _search_text
        hay = self.df["_search_text"].str.lower().fillna("")
        score = sum(hay.str.contains(re.escape(tok), regex=True) for tok in toks).astype(int)
        hits = self.df.loc[score[score > 0].sort_values(ascending=False).index].copy()
        hits["_score"] = score[score > 0].sort_values(ascending=False).values.astype(float)
        return hits.head(k)

    # ------------------- public search -------------------
    def search(self, q: str, k: int=30, brand: Optional[str]=None,
               min_price: Optional[float]=None, max_price: Optional[float]=None) -> pd.DataFrame:
        # If Qdrant off, use fallback
        if not self.search_enabled or self.client is None:
            df = self._csv_keyword_search(q, k=200)
            if df.empty: return df
            df = self._brand_filter_df(df, brand)
            df = self._price_slice_df(df, min_price, max_price)
            return df.head(k).reset_index(drop=True)

        # Qdrant path
        self._ensure_model()
        qv = self._model.encode([_e5_query(q)], normalize_embeddings=True)[0].astype("float32").tolist()

        # build filter
        flt = None
        must = []
        if brand:
            must.append(FieldCondition(key="brand_lc", match=MatchValue(value=str(brand).lower())))
        if min_price is not None:
            must.append(FieldCondition(key="_price_num", range=Range(gte=min_price)))
        if max_price is not None:
            must.append(FieldCondition(key="_price_num", range=Range(lte=max_price)))
        if must: flt = Filter(must=must)

        res = self.client.search(collection_name=self.collection, query_vector=qv, query_filter=flt, limit=k)
        rows = []
        for r in res:
            p = dict(r.payload); p["_score"] = float(r.score); rows.append(p)
        return pd.DataFrame(rows)
