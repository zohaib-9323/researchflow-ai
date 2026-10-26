"""Embedding model construction (isolated so providers can be swapped by config)."""

from __future__ import annotations

import hashlib
import math
import re

from langchain_core.embeddings import Embeddings

from app.config import Settings
from app.core.exceptions import EmbeddingUnavailableError

_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbeddings(Embeddings):
    """Deterministic, offline bag-of-words embeddings (feature hashing).

    Lexical only — it has no semantic understanding. Intended for the test suite and
    for development on machines that cannot run sentence-transformers. Never the
    default.
    """

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = _TOKEN.findall(text.lower())
        for token in tokens:
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def embedding_model_id(settings: Settings) -> str:
    """Identifier stored with each index so a model change is detected."""
    if settings.embedding_provider == "hashing":
        return "hashing-384"
    return f"huggingface:{settings.embedding_model}"


def build_embeddings(settings: Settings) -> Embeddings:
    if settings.embedding_provider == "hashing":
        return HashingEmbeddings()
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:  # pragma: no cover - depends on platform
        raise EmbeddingUnavailableError(
            "langchain-huggingface is not installed. Install the project dependencies."
        ) from exc
    try:
        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": settings.embedding_device},
            encode_kwargs={"normalize_embeddings": True},
        )
    except ImportError as exc:  # pragma: no cover - depends on platform
        raise EmbeddingUnavailableError(
            "sentence-transformers/PyTorch is not available on this platform (for example "
            "Intel macOS). Run the app with Docker, or set EMBEDDING_PROVIDER=hashing for "
            "development only."
        ) from exc
