#!/bin/bash
set -e

# Start Qdrant in the background
/qdrant/qdrant &

# Wait a little for Qdrant to boot
sleep 5

# Start FastAPI app on the port Cloud Run provides
exec uvicorn app.app:app --host 0.0.0.0 --port ${PORT:-8080}
