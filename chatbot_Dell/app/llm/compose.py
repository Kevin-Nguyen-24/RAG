# app/llm/compose.py
import os
import requests
from typing import Optional

# ---- Ollama settings ----
# MODEL_SERVER: your Ollama base URL (e.g., https://ollama-gemma-....run.app)
MODEL_SERVER = os.getenv("MODEL_SERVER", "http://127.0.0.1:11434")
# MODEL_NAME: an Ollama tag you pulled/deployed (e.g., gemma2:2b-instruct, llama3.1:8b-instruct)
MODEL_NAME   = os.getenv("MODEL_NAME", "gemma2:2b-instruct")

# Generation knobs
TEMP        = float(os.getenv("LLM_TEMPERATURE", "0.2"))
MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS", "256"))
TIMEOUT     = float(os.getenv("LLM_TIMEOUT", "60"))
TOP_P       = float(os.getenv("LLM_TOP_P", "0.9"))
REPEAT_PEN  = float(os.getenv("LLM_REPEAT_PENALTY", "1.12"))

# If your Ollama service is behind a reverse proxy that expects a static bearer:
#   export LLM_AUTH_BEARER="SUPERSECRET"
LLM_AUTH_BEARER = os.getenv("LLM_AUTH_BEARER")

# If your Ollama runs on private Cloud Run and requires an ID token:
#   export LLM_USE_ID_TOKEN=1
LLM_USE_ID_TOKEN = os.getenv("LLM_USE_ID_TOKEN", "0") == "1"

APOLOGY = (
    "I am sorry. I am designed to answer questions about our company product and service. "
    "You can ask me about our products and services."
)

def _gcp_id_token(audience: str) -> Optional[str]:
    """Fetch an ID token for Cloud Run (works only when running on GCP with metadata server)."""
    try:
        r = requests.get(
            "http://metadata/computeMetadata/v1/instance/service-accounts/default/identity",
            params={"audience": audience, "format": "full"},
            headers={"Metadata-Flavor": "Google"},
            timeout=3,
        )
        return r.text if r.ok else None
    except Exception:
        return None

def _call_local_llm(prompt: str) -> Optional[str]:
    """Call Ollama's /api/generate endpoint with a single prompt."""
    try:
        headers = {"Content-Type": "application/json"}

        # Prefer ID token when requested (private Cloud Run)
        if LLM_USE_ID_TOKEN:
            tok = _gcp_id_token(MODEL_SERVER)
            if tok:
                headers["Authorization"] = f"Bearer {tok}"
        # Otherwise fall back to a static bearer (if provided)
        elif LLM_AUTH_BEARER:
            headers["Authorization"] = f"Bearer {LLM_AUTH_BEARER}"

        r = requests.post(
            f"{MODEL_SERVER}/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "temperature": TEMP,
                "top_p": TOP_P,
                "repeat_penalty": REPEAT_PEN,
                "num_predict": MAX_TOKENS,
                "stream": False,
            },
            headers=headers,
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        js = r.json()
        return (js.get("response") or "").strip() or None
    except Exception:
        return None

def _prompt_rag(user_q: str, context: str) -> str:
    return f"""You are a concise laptop expert. Use ONLY the FACTS section.

USER QUERY:
{user_q}

FACTS:
{context}

GUIDELINES:
- Use only information in FACTS; if a detail is missing, say "not available".
- If the user asks for a specific model → give price + 4–8 key specs.
- If they ask for options → list up to 3 most relevant models with 1–2 deciding specs.
- If they ask to compare → summarize key differences.
- Answer in <= 120 words. No tables.

Now write the final answer:"""

# Public helpers
def compose_freeform(context: str, user_query: str) -> Optional[str]:
    return _call_local_llm(_prompt_rag(user_query, context))

def compose_out_of_scope(_: str) -> str:
    return APOLOGY

# Backward-compat shim
def compose(context: str, user_query: str) -> Optional[str]:
    return compose_freeform(context, user_query)
