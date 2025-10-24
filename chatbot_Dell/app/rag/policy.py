# rag/policy.py
import re

COMPETITORS = r"\b(hp|hewlett|ibm|lenovo|asus|acer|msi|apple|microsoft|razer|samsung)\b"

def classify(query: str):
    q = query.lower()
    is_dell = "dell" in q
    wants_specs = any(w in q for w in ["how much","price","cost","spec","specs","specification","summary","details"])
    is_comp = re.search(COMPETITORS, q, re.I) is not None
    if is_dell and wants_specs: return "DELL_PRICE_SPECS"
    if is_comp: return "COMPETITOR_TO_DELL"
    if is_dell: return "DELL_GENERAL"
    return "FALLBACK"
