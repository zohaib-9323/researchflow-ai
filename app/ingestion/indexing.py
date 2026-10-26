"""Per-collection FAISS indexes.

Vectors are stored in FAISS keyed by the integer primary key of the `chunks` table;
all chunk text and metadata live in the application database. Each collection has its
own index directory, so collections can never leak results into each other.

Index files are written atomically (temp file + rename) to avoid torn writes.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from langchain_core.embeddings import Embeddings

from app.core.exceptions import IndexingError

_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_INDEX_FILE = "index.faiss"
_META_FILE = "meta.json"


class IndexModelMismatchError(IndexingError):
    code = "index_model_mismatch"
    default_message = (
        "This collection was indexed with a different embedding model. Rebuild the "
        "index (python -m app.scripts.rebuild_index) before searching."
    )


class FaissIndexManager:
    def __init__(self, base_dir: Path, embeddings: Embeddings, model_id: str) -> None:
        self.base_dir = base_dir
        self.embeddings = embeddings
        self.model_id = model_id
        # faiss ships incomplete type stubs; indexes are typed as Any.
        self._indexes: dict[str, Any] = {}
        self._locks: dict[str, threading.RLock] = {}
        self._guard = threading.Lock()

    # -- paths & locking ----------------------------------------------------------------
    def _dir(self, collection_id: str) -> Path:
        if not _SAFE_ID.match(collection_id):
            raise IndexingError("Invalid collection identifier.")
        return self.base_dir / collection_id

    def _lock(self, collection_id: str) -> threading.RLock:
        with self._guard:
            return self._locks.setdefault(collection_id, threading.RLock())

    # -- load / save ----------------------------------------------------------------------
    def _load(self, collection_id: str) -> Any:
        if collection_id in self._indexes:
            return self._indexes[collection_id]
        directory = self._dir(collection_id)
        index_path, meta_path = directory / _INDEX_FILE, directory / _META_FILE
        if not index_path.exists():
            return None
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        if meta.get("embedding_model") != self.model_id:
            raise IndexModelMismatchError()
        index = faiss.read_index(str(index_path))
        self._indexes[collection_id] = index
        return index

    def _save(self, collection_id: str, index: Any) -> None:
        directory = self._dir(collection_id)
        directory.mkdir(parents=True, exist_ok=True)
        tmp_index = directory / f"{_INDEX_FILE}.tmp"
        faiss.write_index(index, str(tmp_index))
        os.replace(tmp_index, directory / _INDEX_FILE)
        tmp_meta = directory / f"{_META_FILE}.tmp"
        tmp_meta.write_text(json.dumps({"embedding_model": self.model_id, "dimensions": index.d}))
        os.replace(tmp_meta, directory / _META_FILE)

    @staticmethod
    def _as_matrix(vectors: list[list[float]]) -> np.ndarray:
        matrix = np.asarray(vectors, dtype="float32")
        faiss.normalize_L2(matrix)  # cosine similarity via inner product
        return matrix

    # -- public API -----------------------------------------------------------------------
    def add(self, collection_id: str, ids: list[int], texts: list[str]) -> None:
        if not ids:
            return
        vectors = self._as_matrix(self.embeddings.embed_documents(texts))
        with self._lock(collection_id):
            index = self._load(collection_id)
            if index is None:
                index = faiss.IndexIDMap2(faiss.IndexFlatIP(int(vectors.shape[1])))
            if index.d != vectors.shape[1]:
                raise IndexModelMismatchError()
            index.add_with_ids(vectors, np.asarray(ids, dtype="int64"))
            try:
                self._save(collection_id, index)
            except OSError as exc:
                index.remove_ids(np.asarray(ids, dtype="int64"))
                raise IndexingError("The vector index could not be saved.") from exc
            self._indexes[collection_id] = index

    def remove(self, collection_id: str, ids: list[int]) -> None:
        if not ids:
            return
        with self._lock(collection_id):
            index = self._load(collection_id)
            if index is None:
                return
            index.remove_ids(np.asarray(ids, dtype="int64"))
            try:
                self._save(collection_id, index)
            except OSError as exc:
                # Drop the cached copy so the on-disk state is reloaded next time.
                self._indexes.pop(collection_id, None)
                raise IndexingError("The vector index could not be updated.") from exc

    def search(self, collection_id: str, query: str, k: int) -> list[tuple[int, float]]:
        with self._lock(collection_id):
            index = self._load(collection_id)
            if index is None or index.ntotal == 0:
                return []
            vector = self._as_matrix([self.embeddings.embed_query(query)])
            scores, ids = index.search(vector, min(k, index.ntotal))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0], strict=True) if i != -1]

    def count(self, collection_id: str) -> int:
        with self._lock(collection_id):
            index = self._load(collection_id)
            return 0 if index is None else int(index.ntotal)

    def reset(self, collection_id: str) -> None:
        with self._lock(collection_id):
            self._indexes.pop(collection_id, None)
            directory = self._dir(collection_id)
            if directory.exists():
                shutil.rmtree(directory)
