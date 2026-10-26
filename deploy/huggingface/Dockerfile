# Single-container image for a Hugging Face Docker Space.
# The FastAPI backend listens on 127.0.0.1:8000 (private); only Streamlit is public on 7860.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    CHAT_MODEL=gemini-3.5-flash \
    CHAT_FALLBACK_MODELS=gemini-3.5-flash-lite,gemini-2.5-flash \
    EMBEDDING_PROVIDER=huggingface \
    API_BASE_URL=http://127.0.0.1:8000 \
    CORS_ORIGINS=http://127.0.0.1:7860

WORKDIR /app

# CPU-only PyTorch first so sentence-transformers does not pull CUDA wheels.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.4,<3"

COPY pyproject.toml README.md ./
COPY app ./app
RUN pip install .

# Bake the embedding model into the image so the Space starts quickly.
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY .streamlit ./.streamlit
COPY frontend ./frontend
COPY evaluation ./evaluation
COPY deploy/huggingface/start.sh ./start.sh

# Spaces run the container as uid 1000.
RUN useradd --create-home --uid 1000 user \
    && mkdir -p /app/data \
    && chmod +x /app/start.sh \
    && chown -R user /app
USER user

EXPOSE 7860
CMD ["/app/start.sh"]
