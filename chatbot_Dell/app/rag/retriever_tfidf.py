import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ProductRetriever:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.brand_col = "brand"
        self.model_col = "model"
        self.price_col = "price"

        # collect spec columns = everything except brand/model/price
        spec_cols = [c for c in self.df.columns if c not in [self.brand_col, self.model_col, self.price_col]]

        # build spec summary text
        self.df["_spec_summary"] = self.df[spec_cols].fillna("").astype(str).agg(
            lambda r: "; ".join(f"{c}: {r[c]}" for c in spec_cols if r[c].strip()), axis=1
        )

        # TF-IDF corpus = brand + model + price + specs
        corpus = (
            self.df[self.brand_col].astype(str) + " " +
            self.df[self.model_col].astype(str) + " " +
            "price:" + self.df[self.price_col].astype(str).fillna("") + " " +
            self.df["_spec_summary"]
        )

        self.vectorizer = TfidfVectorizer(ngram_range=(1,2))
        self.X = self.vectorizer.fit_transform(corpus)

    def search(self, query: str, k=5):
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.X).ravel()
        top = sims.argsort()[::-1][:k]
        out = self.df.iloc[top].copy()
        out["_score"] = sims[top]
        return out
