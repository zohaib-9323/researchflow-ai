#!/bin/sh
# Start the private API, load the demo documents in the background, then serve the UI.
set -e

# Use the Space's persistent storage when it is enabled (mounted at /data).
if [ -z "$DATA_DIR" ]; then
  if [ -d /data ] && [ -w /data ]; then
    export DATA_DIR=/data/researchflow
  else
    export DATA_DIR=/app/data
  fi
fi
mkdir -p "$DATA_DIR"

uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000 &
python -m app.scripts.seed_demo --api http://127.0.0.1:8000 --papers &

# XSRF protection is off because Spaces serve the app inside an iframe, which breaks
# Streamlit's upload token; set APP_ACCESS_CODE to keep the app private.
exec streamlit run frontend/streamlit_app.py \
  --server.address=0.0.0.0 \
  --server.port=7860 \
  --server.headless=true \
  --server.enableXsrfProtection=false \
  --browser.gatherUsageStats=false
