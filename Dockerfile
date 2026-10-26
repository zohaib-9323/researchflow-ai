FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# CPU-only PyTorch first so sentence-transformers does not pull CUDA wheels.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.4,<3"

COPY pyproject.toml README.md ./
COPY app ./app
RUN pip install .

COPY .streamlit ./.streamlit
COPY frontend ./frontend
COPY evaluation ./evaluation

RUN useradd --create-home --uid 1000 researchflow \
    && mkdir -p /app/data /app/.cache/huggingface \
    && chown -R researchflow /app/data /app/.cache
USER researchflow

EXPOSE 8000 8501
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
