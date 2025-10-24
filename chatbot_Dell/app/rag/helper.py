# app/rag/helper.py
from __future__ import annotations
import math, re
from typing import Optional, Tuple
import numpy as np
import pandas as pd

# ---------------- Regex & constants ----------------
PRICE_RANGE_PAT = re.compile(
    r"(?:between|from)\s*\$?\s*(\d{3,6})\s*(?:-|to|and)\s*\$?\s*(\d{3,6})", re.I
)
PRICE_UNDER_PAT = re.compile(r"(?:under|less\s*than|below|<=?)\s*[\$₹]?\s*(\d{3,6})(?:\s*dola(?:r|rs?)?)?\b", re.I)
PRICE_ABOUT_PAT = re.compile(r"(?:about|around|approx(?:\.|imately)?)\s*[\$₹]?\s*(\d{3,6})(?:\s*dola(?:r|rs?)?)?\b", re.I)
PRICE_BUDGET_PAT = re.compile(r"(?:budget|max(?:imum)?|up\s*to)\s*[\$₹]?\s*(\d{3,6})(?:\s*dola(?:r|rs?)?)?\b", re.I)
PRICE_BARE_PAT = re.compile(r"^\s*[\$₹]?\s*(\d{3,6})\s*(?:dola(?:r|rs?)?|cad|hkd|ca|hk)?\s*$", re.I)

MARKET_WORDS = {
    "hk": "HK", "hkd": "HK", "hongkong": "HK", "hong": "HK",
    "ca": "CA", "cad": "CA", "canada": "CA", "ontario": "CA", "on": "CA",
    "bc": "CA", "quebec": "CA", "qc": "CA", "toronto": "CA"
}

SIMPLE_PRICE_SPEC = re.compile(
    r"(how\s+much|price|cost|specs?\b|specification|details|show\s+spec|give\s+spec)", re.I
)
BEST_PRICE_PAT = re.compile(r"\b(best|lowest|cheapest)\b.*\b(price|deal|cost)\b", re.I)
GENERAL_LIST_PAT = re.compile(r"\b(show|list|what)\b.*\b(dell|laptops?|models?)\b", re.I)

CONFIRM_FULLMATCH = re.compile(r"^(?:dell\s*)?\d{3,5}(?:\s*(?:price|specs?|good|specifications?))?$", re.I)
SERIES_PAT = re.compile(r"\b(inspiron|vostro|xps|latitude|precision|alienware|g\s?\d|optiplex)\b", re.I)

# Enhanced patterns for storage and hardware detection
STORAGE_PAT = re.compile(r"\b(\d+)\s*(?:gb|tb)\s*(ssd|storage|hdd)\b", re.I)
RAM_PAT = re.compile(r"\b(\d+)\s*gb\s*ram\b", re.I)
CPU_PAT = re.compile(r"\b(i[357]|ryzen\s*[357579]|intel|amd)\b", re.I)
GPU_PAT = re.compile(r"\b(rtx|gtx|nvidia|radeon|rx)\s*\d*\b", re.I)
SCREEN_PAT = re.compile(r"\b(\d+(?:\.\d+)?)\s*inch\b", re.I)

CURRENCY_SYMBOL = {"CAD": "CA$", "HKD": "HK$", "USD": "$", "INR": "₹"}

YES_WORDS = {"yes", "y", "ok", "okay", "sure"}

__all__ = [
    # parsing/market
    "parse_price_intent", "detect_market_word", "resolve_market",
    # currency/tax
    "market_to_currency", "price_to_display", "fx_convert_from_target_to_base",
    # df utils
    "filter_to_dell", "build_context_from_hits", "safe_brand_model",
    "_to_num_price", "_cheapest_row",
    # patterns/flags/guards
    "SIMPLE_PRICE_SPEC", "BEST_PRICE_PAT", "GENERAL_LIST_PAT",
    "should_confirm", "YES_WORDS",
    # hardware filtering
    "extract_hardware_filters", "apply_hardware_filters",
]


# ---------------- Small helpers ----------------
def s(x) -> str:
    if x is None: return ""
    if isinstance(x, float) and math.isnan(x): return ""
    return str(x)

def _to_num_price(x: str) -> float:
    txt = str(x or "")
    txt = re.sub(r"[^\d.]", "", txt)
    try:
        return float(txt) if txt else np.nan
    except Exception:
        return np.nan

def _cheapest_row(df: pd.DataFrame):
    if df is None or df.empty or "price" not in df.columns: return None
    nums = df["price"].map(_to_num_price)
    if not nums.notna().any(): return None
    idx = int(nums.idxmin())
    return df.loc[idx]

def filter_to_dell(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "brand" not in df.columns: return df
    return df[df["brand"].astype(str).str.contains("dell", case=False, na=False)]

def safe_brand_model(row) -> str:
    b = s(row.get("brand")).strip()
    m = (s(row.get("model")) or s(row.get("Model"))).strip()
    if not m: return b
    if b and m.lower().startswith(b.lower() + " "): return m
    return f"{b} {m}".strip()

def build_fact_line(r, market: str, price_to_display_fn=None) -> str:
    title = safe_brand_model(r)
    price_disp = (price_to_display_fn or price_to_display)(r.get("price"), market)
    spec = s(r.get("Specification")).strip()
    if not spec:
        picks = []
        for k in ("processor_tier","ram_memory","primary_storage_capacity","primary_storage_type",
                  "gpu_type","display_size","OS"):
            v = s(r.get(k)).strip()
            if v: picks.append(f"{k}: {v}")
        spec = "; ".join(picks)
    return f"- {title} — price: {price_disp}; specs: {spec[:500]}"

def build_context_from_hits(hits: pd.DataFrame, market: str, k=4) -> str:
    lines = [build_fact_line(r, market) for _, r in hits.head(k).iterrows()]
    return "\n".join(lines)

# ---------------- Parsing & market ----------------
def parse_price_intent(q: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    t = q.lower().strip()
    m = PRICE_RANGE_PAT.search(t)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        if lo > hi: lo, hi = hi, lo
        return lo, hi, "range"
    m = PRICE_UNDER_PAT.search(t)
    if m:
        return None, float(m.group(1)), "under"
    m = PRICE_ABOUT_PAT.search(t)
    if m:
        mid = float(m.group(1))
        return mid * 0.9, mid * 1.1, "about"
    m = PRICE_BUDGET_PAT.search(t)
    if m:
        return None, float(m.group(1)), "budget"
    m = PRICE_BARE_PAT.match(t)
    if m:
        return None, float(m.group(1)), "bare"
    return None, None, None

def detect_market_word(q: str) -> Optional[str]:
    t = re.sub(r"[\s\-]", "", q.lower())
    for k, m in MARKET_WORDS.items():
        if k in t:
            return m
    return None

def resolve_market(query: str, default_market: str = "CA") -> str:
    t = query.lower()
    market_keywords = [
        (r"\bhk(d)?\b|\bhong\s*kong\b", "HK"),
        (r"\bcanada\b|\bcad\b|\bontario\b|\btoronto\b|\bqc\b|\bqu[eé]bec\b|\bbc\b", "CA"),
    ]
    for pat, mkt in market_keywords:
        if re.search(pat, t, re.I):
            return mkt
    return default_market

def should_confirm(q: str) -> bool:
    t = q.strip().lower()
    if SERIES_PAT.search(t): return False
    if CONFIRM_FULLMATCH.fullmatch(t): return True
    # Enhanced confirmation for partial model queries
    if re.search(r"^(?:dell\s*)?\d{3,5}(?:\s*(?:price|specs?|good|specifications?))?$", t, re.I):
        return True
    return bool(re.search(r"\b\d{3,5}\b", t))

def extract_hardware_filters(q: str) -> dict:
    """Extract hardware specifications from query for filtering"""
    filters = {}
    
    # Storage filtering
    storage_match = STORAGE_PAT.search(q)
    if storage_match:
        size, unit = storage_match.groups()
        if unit.lower() in ['ssd', 'storage']:
            filters['storage_size'] = int(size)
            filters['storage_type'] = unit.lower()
    
    # RAM filtering
    ram_match = RAM_PAT.search(q)
    if ram_match:
        filters['ram_gb'] = int(ram_match.group(1))
    
    # CPU filtering
    cpu_match = CPU_PAT.search(q)
    if cpu_match:
        filters['cpu_type'] = cpu_match.group(1).lower()
    
    # GPU filtering
    gpu_match = GPU_PAT.search(q)
    if gpu_match:
        filters['gpu_type'] = gpu_match.group(0).lower()
    
    # Screen size filtering
    screen_match = SCREEN_PAT.search(q)
    if screen_match:
        filters['screen_size'] = float(screen_match.group(1))
    
    return filters

def apply_hardware_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply hardware filters to DataFrame"""
    if df.empty:
        return df
    
    result = df.copy()
    
    # Apply RAM filter
    if 'ram_gb' in filters:
        ram_gb = filters['ram_gb']
        if 'ram_memory' in result.columns:
            # ram_memory column contains numeric values like 16, not "16gb"
            result = result[result['ram_memory'].astype(str).str.contains(str(ram_gb), case=False, na=False)]
        elif 'Specification' in result.columns:
            result = result[result['Specification'].astype(str).str.contains(f"{ram_gb}gb", case=False, na=False)]
    
    # Apply CPU filter
    if 'cpu_type' in filters:
        cpu_type = filters['cpu_type']
        if 'processor_tier' in result.columns:
            result = result[result['processor_tier'].astype(str).str.contains(cpu_type, case=False, na=False)]
        elif 'Specification' in result.columns:
            result = result[result['Specification'].astype(str).str.contains(cpu_type, case=False, na=False)]
    
    # Apply GPU filter
    if 'gpu_type' in filters:
        gpu_type = filters['gpu_type']
        if 'gpu_type' in result.columns:
            result = result[result['gpu_type'].astype(str).str.contains(gpu_type, case=False, na=False)]
        elif 'Specification' in result.columns:
            result = result[result['Specification'].astype(str).str.contains(gpu_type, case=False, na=False)]
    
    # Apply storage filter
    if 'storage_size' in filters:
        size = filters['storage_size']
        if 'primary_storage_capacity' in result.columns:
            # primary_storage_capacity column contains numeric values like 512, not "512gb"
            result = result[result['primary_storage_capacity'].astype(str).str.contains(str(size), case=False, na=False)]
        elif 'Specification' in result.columns:
            result = result[result['Specification'].astype(str).str.contains(f"{size}gb", case=False, na=False)]
    
    # Apply screen size filter
    if 'screen_size' in filters:
        screen = filters['screen_size']
        if 'display_size' in result.columns:
            result = result[result['display_size'].astype(str).str.contains(f"{screen}", case=False, na=False)]
        elif 'Specification' in result.columns:
            result = result[result['Specification'].astype(str).str.contains(f"{screen}", case=False, na=False)]
    
    return result

# ---------------- Currency / tax ----------------
def market_to_currency(market: str) -> str:
    return "HKD" if market == "HK" else "CAD"

def _fx_convert(num: float, from_cur: str, market: str, FX: dict) -> tuple[float, str]:
    target = market_to_currency(market)
    if from_cur.upper() == target:
        return num, target
    table = (FX or {}).get(from_cur.upper(), {})
    rate = float(table.get(target, 1.0))
    return num * rate, target

def _fmt_currency(num: float, code: str) -> str:
    sym = CURRENCY_SYMBOL.get(code.upper(), code + " ")
    return f"{sym}{num:,.0f}"

def _tax_rate_for_market(market: str, CA_TAX: dict, CA_PROVINCE: str, HK_TAX: float) -> float:
    if market == "HK": return HK_TAX
    return (CA_TAX or {}).get(CA_PROVINCE, 0.13)

def price_to_display(
    raw_price,
    market: str,
    *,
    BASE_CURRENCY: str = "INR",
    FX: dict | None = None,
    INCLUDE_TAX_LINE: bool = True,
    CA_TAX: dict | None = None,
    CA_PROVINCE: str = "ON",
    HK_TAX: float = 0.0,
) -> str:
    """Render a price string converted to the market currency (and est. tax if enabled)."""
    try:
        if isinstance(raw_price, (int, float)) and not math.isnan(float(raw_price)):
            base = float(raw_price)
        else:
            txt = re.sub(r"[^\d.]", "", str(raw_price or ""))
            if not txt:
                return "not available"
            base = float(txt)
    except Exception:
        return str(raw_price) if raw_price else "not available"

    num_conv, code = _fx_convert(base, BASE_CURRENCY, market, FX or {})
    out = _fmt_currency(num_conv, code)
    if INCLUDE_TAX_LINE:
        tr = _tax_rate_for_market(market, CA_TAX or {}, CA_PROVINCE, HK_TAX)
        if tr > 0:
            with_tax = num_conv * (1.0 + tr)
            out += f" (≈ {_fmt_currency(with_tax, code)} incl. est. tax)"
    return out

def fx_convert_from_target_to_base(amount: float | None, market: str, BASE_CURRENCY: str, FX: dict) -> float | None:
    """Invert BASE->target from FX to compute target->BASE."""
    if amount is None:
        return None
    target = market_to_currency(market)
    table = (FX or {}).get(BASE_CURRENCY.upper(), {})
    fwd = float(table.get(target, 1.0))
    if fwd <= 0:
        return amount
    return amount / fwd
