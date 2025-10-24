FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CSV_PATH=/app/app/laptops_norm.csv \
    REINDEX_ON_START=0

WORKDIR /app

# 1) Faster, reliable wheels
RUN python -m pip install --upgrade pip
# Pin a CPU torch wheel first (avoids surprises on slim images)
RUN pip install --no-cache-dir torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu

# 2) App deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) Copy code (includes templates/static and laptops_norm.csv)
COPY app/ ./app/

EXPOSE 8080
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8080"]
