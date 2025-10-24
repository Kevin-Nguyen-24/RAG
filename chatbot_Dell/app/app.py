# app/app.py
import os, re, time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.llm.compose import compose_freeform, compose_out_of_scope
from app.rag.retriever_qdrant import ProductRetrieverQdrant as ProductRetriever

from app.rag.helper import (
    parse_price_intent, detect_market_word, resolve_market,
    market_to_currency, price_to_display, fx_convert_from_target_to_base,
    filter_to_dell, build_context_from_hits, safe_brand_model,
    _to_num_price, _cheapest_row, SIMPLE_PRICE_SPEC, BEST_PRICE_PAT,
    GENERAL_LIST_PAT, should_confirm, YES_WORDS,
)

# ---------------- Paths & env ----------------
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
load_dotenv(BASE_DIR / ".env")

# ---------------- Settings ----------------
USE_LLM = os.getenv("USE_LLM", "1") == "1"
TOP_SCORE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.12"))
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = os.getenv("CSV_PATH", str((BASE_DIR / "laptops_norm.csv").resolve()))
ONLY_DELL = os.getenv("ONLY_DELL", "1") == "1"

# ---------------- Qdrant ----------------
QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "https://41d4fe8d-6b68-4b1b-8016-fe6dd2a3d8f7.us-east4-0.gcp.cloud.qdrant.io:6333"
) 
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "laptops")
EMB_MODEL = os.getenv("EMB_MODEL_PATH", "intfloat/e5-base-v2")
REINDEX_ON_START = os.getenv("REINDEX_ON_START", "0") == "1"
QDRANT_API_KEY = os.getenv("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.eE5-BjmVOJ2ihMD1XtmmOl5ur6A2rfTzQQDnr4Tf9Ko")

# ---------- Data (CSV) ----------
# app/app.py (around the retriever creation)
try:
    retriever = ProductRetriever(
        csv_path=CSV_PATH,
        collection=QDRANT_COLLECTION,
        qdrant_url=QDRANT_URL,
        model_name_or_path=EMB_MODEL,
        recreate=(os.getenv("REINDEX_ON_START","0")=="1"),
        api_key=QDRANT_API_KEY,
    )
except Exception as e:
    print(f"[WARN] Retriever init failed: {e}")
    import pandas as _pd
    class _Dummy:
        def search(self, *a, **k): return _pd.DataFrame([])
    retriever = _Dummy()

    print(f"[WARN] Retriever init failed (CSV={CSV_PATH}): {e}")
    


# ---- Market/currency/tax ----
PRICE_CURRENCY = os.getenv("PRICE_CURRENCY", "INR").upper()
DEFAULT_MARKET = os.getenv("DEFAULT_MARKET", "HK").upper()
BASE_CURRENCY = os.getenv("BASE_CURRENCY", PRICE_CURRENCY).upper()

FX = {
    "INR": {
        "CAD": float(os.getenv("FX_INR_TO_CAD", "0.016")),
        "HKD": float(os.getenv("FX_INR_TO_HKD", "0.094")),
        "INR": 1.0,
    },
    "USD": {
        "CAD": float(os.getenv("FX_USD_TO_CAD", "1.36")),
        "HKD": float(os.getenv("FX_USD_TO_HKD", "7.80")),
        "USD": 1.0,
    },
}

CA_PROVINCE = os.getenv("CA_PROVINCE", "ON").upper()
INCLUDE_TAX_LINE = os.getenv("INCLUDE_TAX_LINE", "1") == "1"

CA_TAX = {
    "AB": 0.05, "BC": 0.12, "MB": 0.12, "NB": 0.15, "NL": 0.15, "NS": 0.15,
    "NT": 0.05, "NU": 0.05, "ON": 0.13, "PE": 0.15, "QC": 0.14975, "SK": 0.11, "YT": 0.05
}
HK_TAX = 0.0

# ---------------- FastAPI ----------------
TEMPLATES_DIR = BASE_DIR / "templates"
app = FastAPI()

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
def health():
    from os.path import exists
    return {
        "status": "ok",
        "csv_found": exists(CSV_PATH),
        "csv_path": CSV_PATH,
    }


GREETING_MSG = (
    "My name is Kevin. I’m the virtual assistant for our company’s products. "
    "What can I help you with?"
)
def is_greeting(text: str) -> bool:
    import re as _re
    return bool(_re.match(r"^(hi|hello|hey|yo|hola|greetings)\b", text.strip().lower()))

@app.get("/greet")
def greet():
    return {"answer": GREETING_MSG, "mode": "GREETING"}

# Keep a pending price clarification across one follow-up
PENDING_PRICE_REQ: dict | None = None
PENDING_PRICE_TS: float = 0.0
LAST_CONFIRM_NEXT: str | None = None
LAST_CONFIRM_TS: float = 0.0

# ---------------- Ask API ----------------
class Ask(BaseModel):
    query: str

def price_display(value, market: str) -> str:
    return price_to_display(
        value, market,
        BASE_CURRENCY=BASE_CURRENCY,
        FX=FX,
        INCLUDE_TAX_LINE=INCLUDE_TAX_LINE,
        CA_TAX=CA_TAX,
        CA_PROVINCE=CA_PROVINCE,
        HK_TAX=HK_TAX,
    )

def list_dell_in_price_band(market: str, min_target: float | None, max_target: float | None, k: int = 12) -> str:
    min_base = fx_convert_from_target_to_base(min_target, market, BASE_CURRENCY, FX) if min_target is not None else None
    max_base = fx_convert_from_target_to_base(max_target, market, BASE_CURRENCY, FX) if max_target is not None else None

    hits = retriever.search("Dell laptop", k=50, brand="Dell", min_price=min_base, max_price=max_base)
    dell_hits = filter_to_dell(hits)
    if dell_hits.empty:
        return "Sorry, I couldn’t find Dell laptops matching that price."

    df = dell_hits.copy()
    df["num_price"] = df["price"].map(_to_num_price)
    df = df[df["num_price"].notna()].sort_values("num_price")
    if df.empty:
        return "Sorry, I couldn’t find Dell laptops with valid prices in that range."

    take = df.head(k)
    lines = []
    for _, r in take.iterrows():
        title = safe_brand_model(r)
        price_disp = price_display(r.get("price"), market)
        spec = str(r.get("Specification") or "")[:350]
        lines.append(f"{title}\nPrice: {price_disp}\nSpecs: {spec}")
    return "\n\n".join(lines)

@app.post("/ask")
def ask(payload: Ask):
    global LAST_CONFIRM_NEXT, LAST_CONFIRM_TS
    global PENDING_PRICE_REQ, PENDING_PRICE_TS

    q = (payload.query or "").strip()
    if not q:
        return {"answer": "Please type a question.", "mode": "EMPTY"}

    if q.lower() in YES_WORDS and LAST_CONFIRM_NEXT and (time.time() - LAST_CONFIRM_TS) < 120:
        follow_q = LAST_CONFIRM_NEXT
        LAST_CONFIRM_NEXT = None

        hits = retriever.search(follow_q, k=1, brand="Dell" if ONLY_DELL else None)
        dell_hits = filter_to_dell(hits)
        if not dell_hits.empty:
            top = dell_hits.iloc[0]
            title = safe_brand_model(top)
            price = price_display(top.get("price"), resolve_market(follow_q, DEFAULT_MARKET))
            spec = str(top.get("Specification") or "")
            msg = f"{title}\nPrice: {price}"
            if spec:
                msg += f"\nSpecs: {spec[:400]}"
            return {"answer": msg, "mode": "CONFIRM_RESULT"}

    if is_greeting(q):
        return {"answer": GREETING_MSG, "mode": "GREETING"}

    mk_follow = detect_market_word(q)
    if mk_follow and PENDING_PRICE_REQ and (time.time() - PENDING_PRICE_TS) < 180:
        min_t = PENDING_PRICE_REQ.get("min_target")
        max_t = PENDING_PRICE_REQ.get("max_target")
        note = "Note: **no sales tax** in Hong Kong." if mk_follow == "HK" else \
               "Note: prices shown with ~13% estimated tax for Ontario, Canada."
        body = list_dell_in_price_band(mk_follow, min_t, max_t, k=12)
        PENDING_PRICE_REQ, PENDING_PRICE_TS = None, 0.0
        return {
            "answer": f"Here are Dell options for your budget in **{market_to_currency(mk_follow)}**.\n{note}\n\n{body}",
            "mode": "DELL_BY_PRICE"
        }

    if should_confirm(q):
        hits_any = retriever.search(q, k=10, brand="Dell" if ONLY_DELL else None)
        dell_hits = filter_to_dell(hits_any)
        if not dell_hits.empty:
            top = dell_hits.iloc[0]
            candidate = safe_brand_model(top)
            follow = f"show specs and price for {candidate}"
            LAST_CONFIRM_NEXT, LAST_CONFIRM_TS = follow, time.time()
            return {
                "answer": f"Did you mean **{candidate}**? Reply **Yes** to see its specifications, or **No** to try again.",
                "mode": "CONFIRM",
                "confirm_next": follow,
            }

    min_t, max_t, how = parse_price_intent(q)
    if how:
        chosen_mk = detect_market_word(q) or resolve_market(q, DEFAULT_MARKET)
        explicit_mk = detect_market_word(q)

        if not explicit_mk:
            PENDING_PRICE_REQ = {"min_target": min_t, "max_target": max_t}
            PENDING_PRICE_TS = time.time()
            est_line = [
                "• **HK** = HK$ (no sales tax).",
                "• **CA** = CA$ (Ontario estimated 13% HST included in totals).",
            ]
            if how == "range" and min_t is not None and max_t is not None:
                range_text = f"between **{int(min_t)}–{int(max_t)}**"
            elif max_t is not None and min_t is None:
                range_text = f"under **{int(max_t)}**"
            elif min_t is not None and max_t is not None:
                range_text = f"{int(min_t)}–{int(max_t)}"
            elif min_t is not None:
                range_text = f"around **{int(min_t)}**"
            else:
                range_text = "your budget"

            return {
                "answer": (
                    f"I can list Dell laptops {range_text}. Which market should I use?\n"
                    f"{'  \n'.join(est_line)}\n\n"
                    f"Reply **HK** or **CA**, and I’ll show models with prices & key specs."
                ),
                "mode": "PRICE_MARKET_PROMPT"
            }

        body = list_dell_in_price_band(chosen_mk, min_t, max_t, k=12)
        note = "Note: **no sales tax** in Hong Kong." if chosen_mk == "HK" else \
               "Note: prices shown with ~13% estimated tax for Ontario, Canada."
        return {"answer": f"Here are Dell options in **{market_to_currency(chosen_mk)}**. {note}\n\n{body}",
                "mode": "DELL_BY_PRICE"}

    if GENERAL_LIST_PAT.search(q) and "specs and price" not in q.lower():
        hits = retriever.search("Dell laptop", k=50, brand="Dell")
        dell_hits = filter_to_dell(hits)
        if dell_hits.empty:
            return {"answer": "Sorry, I couldn’t find Dell laptops in my catalog.", "mode": "DELL_GENERAL"}

        dell_hits = dell_hits.copy()
        df = dell_hits.copy()
        df["num_price"] = df["price"].map(_to_num_price)
        df = df[df["num_price"].notna()].sort_values("num_price")
        if df.empty:
            return {"answer": "Sorry, no Dell laptops with valid prices found.", "mode": "DELL_GENERAL"}

        high = df.iloc[[-1]]
        mid_start = max(len(df) // 2 - 1, 0)
        mid = df.iloc[mid_start:mid_start + 2]
        picks = pd.concat([high, mid])

        market = resolve_market(q, DEFAULT_MARKET)
        lines = [
            f"{safe_brand_model(r)}\nPrice: {price_display(r.get('price'), market)}\n"
            f"Specs: {str(r.get('Specification') or '')[:250]}"
            for _, r in picks.iterrows()
        ]
        return {"answer": "Here are some Dell laptops across budget, mid, and premium ranges:\n\n" + "\n\n".join(lines),
                "mode": "DELL_GENERAL"}

    PRODUCT_KEYWORDS = re.compile(
        r"\b(dell|inspiron|xps|vostro|latitude|precision|alienware|laptop|desktop|notebook|pc|\d{3,5})\b", re.I
    )
    if not PRODUCT_KEYWORDS.search(q):
        return {"answer": compose_out_of_scope(q), "mode": "OUT_OF_SCOPE"}

    market = resolve_market(q, DEFAULT_MARKET)

    if should_confirm(q):
        hits_any = retriever.search(q, k=10, brand="Dell" if ONLY_DELL else None)
        dell_hits = filter_to_dell(hits_any)
        if not dell_hits.empty:
            top = dell_hits.iloc[0]
            candidate = safe_brand_model(top)
            follow = f"show specs and price for {candidate}"
            LAST_CONFIRM_NEXT, LAST_CONFIRM_TS = follow, time.time()
            return {
                "answer": f"Did you mean **{candidate}**? Reply **Yes** to see its specifications, or **No** to try again.",
                "mode": "CONFIRM",
                "confirm_next": follow,
            }

    mentions_non_dell = bool(re.search(r"\b(hp|lenovo|asus|acer|msi|apple|razer|samsung|ibm|microsoft)\b", q, re.I))
    brand_filter = "Dell" if (ONLY_DELL or not mentions_non_dell) else None

    max_price = None
    m = re.search(r"\b(under|less\s*than|<=)\s*\$?(\d+)", q, re.I)
    if m:
        try: max_price = float(m.group(2))
        except Exception: max_price = None

    hits = retriever.search(q, k=30, brand=brand_filter, min_price=None, max_price=max_price)

    if mentions_non_dell and ONLY_DELL:
        if hits.empty:
            return {"answer": "We focus on Dell. I couldn’t find a close Dell match to that request.", "mode": "DELL_ALTERNATIVE"}
        ctx = build_context_from_hits(hits, market, k=3)
        return {"answer": f"We specialize in **Dell**. Based on what you asked, here are close Dell alternatives:\n{ctx}\n\n"
                          f"If you share the exact CPU/RAM/GPU you need, I’ll narrow this further.",
                "mode": "DELL_ALTERNATIVE"}

    if not hits.empty and BEST_PRICE_PAT.search(q):
        row = _cheapest_row(hits)
        if row is not None:
            title = safe_brand_model(row)
            price = price_display(row.get("price"), market)
            spec = str(row.get("Specification") or "")
            msg = f"**{title}**\nBest price: {price}"
            if spec:
                msg += f"\nKey specs: {spec[:320]}"
            return {"answer": msg, "mode": "BEST_PRICE"}

    if not hits.empty and SIMPLE_PRICE_SPEC.search(q):
        top = hits.iloc[0]
        title = safe_brand_model(top)
        price = price_display(top.get("price"), market)
        spec = str(top.get("Specification") or "")
        if spec:
            return {"answer": f"{title}\nPrice: {price}\nSpecs: {spec[:400]}", "mode": "FAST_SPECS"}
        return {"answer": f"{title}\nPrice: {price}", "mode": "FAST_SPECS"}

    if hits.empty or float(hits.iloc[0].get("_score", 0.0)) < TOP_SCORE_THRESHOLD:
        return {"answer": compose_out_of_scope(q), "mode": "OUT_OF_SCOPE"}

    context = build_context_from_hits(hits, market, k=4)
    if USE_LLM:
        txt = compose_freeform(context=context, user_query=q)
        if txt:
            return {"answer": txt, "mode": "RAG"}

    top = hits.iloc[0]
    title = safe_brand_model(top)
    price = price_display(top.get("price"), market)
    spec = str(top.get("Specification") or "")
    if spec:
        return {"answer": f"{title}\nPrice: {price}\nSpecs: {spec[:400]}", "mode": "RAG_FALLBACK"}
    return {"answer": f"{title}\nPrice: {price}", "mode": "RAG_FALLBACK"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app.app:app", host="0.0.0.0", port=port, reload=False)
