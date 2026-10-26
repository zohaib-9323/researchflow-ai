"""Document ingestion and lifecycle management.

Pipeline: validate -> extract -> attach metadata -> chunk -> embed + index -> persist.
A document is only marked `indexed` after every step succeeds; any failure leaves it
`failed` with an understandable error and no searchable chunks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document as LCDocument
from sqlalchemy.orm import Session

from app.config import Settings
from app.core.exceptions import (
    DuplicateDocumentError,
    IndexingError,
    NotFoundError,
    ResearchFlowError,
)
from app.core.logging import get_logger
from app.ingestion.chunking import split_documents
from app.ingestion.indexing import FaissIndexManager
from app.ingestion.loaders import load_document
from app.persistence.database import Database
from app.persistence.models import Chunk, Document, new_id, utcnow
from app.persistence.repositories import (
    ChunkRepository,
    CollectionRepository,
    DocumentRepository,
)
from app.utils.file_validation import ValidatedUpload, validate_upload

logger = get_logger(__name__)


@dataclass
class UploadOutcome:
    filename: str
    document: Document | None
    error_code: str | None = None
    error_message: str | None = None


class DocumentService:
    def __init__(self, settings: Settings, database: Database, index: FaissIndexManager) -> None:
        self.settings = settings
        self.database = database
        self.index = index
        self.upload_dir: Path = settings.upload_path

    # -- queries ------------------------------------------------------------------------
    def list_documents(self, collection_id: str | None = None) -> list[Document]:
        with self.database.session() as session:
            return list(DocumentRepository(session).list(collection_id))

    def get_document(self, document_id: str) -> Document:
        with self.database.session() as session:
            document = DocumentRepository(session).get(document_id)
            if document is None or document.status == "deleting":
                raise NotFoundError("Document not found.")
            return document

    # -- ingestion ------------------------------------------------------------------------
    def upload(self, collection_id: str, raw_name: str | None, content: bytes) -> Document:
        """Validate and ingest one file. Raises on validation errors / duplicates.

        Extraction or indexing failures do not raise: they are recorded on the
        document (status `failed`) so the user can see why.
        """
        upload = validate_upload(raw_name, content, self.settings.max_upload_bytes)
        with self.database.session() as session:
            if CollectionRepository(session).get(collection_id) is None:
                raise NotFoundError("Collection not found.")
            existing = DocumentRepository(session).find_by_hash(collection_id, upload.content_hash)
            if existing is not None and existing.status == "failed":
                # A previously failed copy is replaced by the retry.
                self._purge(session, existing)
                session.flush()
            elif existing is not None:
                raise DuplicateDocumentError(
                    f"'{upload.filename}' is already in this collection "
                    f"(uploaded as '{existing.filename}').",
                    details={"document_id": existing.id},
                )
            document = DocumentRepository(session).add(
                Document(
                    id=new_id(),
                    collection_id=collection_id,
                    filename=upload.filename,
                    file_type=upload.file_type,
                    size_bytes=upload.size_bytes,
                    content_hash=upload.content_hash,
                    storage_name=f"{new_id()}.{upload.file_type}",
                    status="processing",
                )
            )
        self._store_file(document.storage_name, upload.content)
        return self._ingest(document, upload)

    def upload_many(
        self, collection_id: str, files: list[tuple[str | None, bytes]]
    ) -> list[UploadOutcome]:
        outcomes: list[UploadOutcome] = []
        for raw_name, content in files:
            try:
                document = self.upload(collection_id, raw_name, content)
                outcomes.append(UploadOutcome(filename=document.filename, document=document))
            except ResearchFlowError as exc:
                outcomes.append(
                    UploadOutcome(
                        filename=raw_name or "document",
                        document=None,
                        error_code=exc.code,
                        error_message=exc.message,
                    )
                )
        return outcomes

    def _store_file(self, storage_name: str, content: bytes) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        path = self._storage_path(storage_name)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(content)
        os.replace(tmp, path)

    def _storage_path(self, storage_name: str) -> Path:
        path = (self.upload_dir / storage_name).resolve()
        if path.parent != self.upload_dir.resolve():  # path-traversal guard
            raise IndexingError("Invalid storage path.")
        return path

    def _ingest(self, document: Document, upload: ValidatedUpload) -> Document:
        log = {"document_id": document.id, "file_type": document.file_type}
        try:
            pages = load_document(upload.content, upload.file_type, document.id, document.filename)
            chunks = split_documents(pages, self.settings.chunk_size, self.settings.chunk_overlap)
            if not chunks:
                raise IndexingError("The document produced no searchable text chunks.")
            page_count = pages[0].metadata.get("page_count") if upload.file_type == "pdf" else None
            self._index_chunks(document, chunks, page_count)
        except ResearchFlowError as exc:
            logger.warning("document_ingestion_failed", extra={**log, "error_code": exc.code})
            return self._mark_failed(document.id, exc.code, exc.message)
        except Exception as exc:
            # Embedding backends raise library-specific errors; record them safely.
            logger.exception("document_ingestion_error", extra=log)
            return self._mark_failed(
                document.id, "indexing_failed", f"Indexing failed ({type(exc).__name__})."
            )
        logger.info("document_indexed", extra={**log, "chunk_count": document.chunk_count})
        return self.get_document(document.id)

    def _index_chunks(
        self, document: Document, chunks: list[LCDocument], page_count: int | None
    ) -> None:
        """Insert chunk rows and vectors atomically (DB rollback + index compensation)."""
        added_ids: list[int] = []
        try:
            with self.database.session() as session:
                rows = ChunkRepository(session).add_many(
                    [
                        Chunk(
                            document_id=document.id,
                            collection_id=document.collection_id,
                            chunk_index=chunk.metadata["chunk_index"],
                            page_number=chunk.metadata.get("page"),
                            text=chunk.page_content,
                        )
                        for chunk in chunks
                    ]
                )
                ids = [row.id for row in rows]
                self.index.add(document.collection_id, ids, [row.text for row in rows])
                added_ids = ids
                stored = self._require(DocumentRepository(session).get(document.id))
                stored.status = "indexed"
                stored.chunk_count = len(rows)
                stored.page_count = page_count
                stored.indexed_at = utcnow()
                stored.error_code = stored.error_message = None
                document.chunk_count = len(rows)
        except BaseException:
            if added_ids:
                try:
                    self.index.remove(document.collection_id, added_ids)
                except IndexingError:
                    # Rows were rolled back, so the stale vectors can never resolve to
                    # searchable chunks; a rebuild removes them from the index.
                    logger.error("index_compensation_failed", extra={"document_id": document.id})
            raise

    def _mark_failed(self, document_id: str, code: str, message: str) -> Document:
        with self.database.session() as session:
            stored = self._require(DocumentRepository(session).get(document_id))
            stored.status = "failed"
            stored.error_code = code
            stored.error_message = message
            stored.chunk_count = 0
            return stored

    @staticmethod
    def _require(document: Document | None) -> Document:
        if document is None:  # deleted concurrently while being ingested
            raise NotFoundError("Document not found.")
        return document

    # -- deletion ---------------------------------------------------------------------------
    def delete_document(self, document_id: str) -> None:
        """Hide immediately, remove vectors, then delete rows and the stored file.

        If the index update fails the document stays in the hidden `deleting` state
        (never searchable) and the error is reported; a rebuild completes the removal.
        """
        with self.database.session() as session:
            document = DocumentRepository(session).get(document_id)
            if document is None or document.status == "deleting":
                raise NotFoundError("Document not found.")
            document.status = "deleting"
            collection_id = document.collection_id
            chunk_ids = ChunkRepository(session).ids_for_document(document_id)
        self.index.remove(collection_id, chunk_ids)
        with self.database.session() as session:
            document = DocumentRepository(session).get(document_id)
            if document is not None:
                self._purge(session, document)
        logger.info("document_deleted", extra={"document_id": document_id})

    def _purge(self, session: Session, document: Document) -> None:
        storage_name = document.storage_name
        ChunkRepository(session).delete_for_document(document.id)
        DocumentRepository(session).delete(document)
        try:
            self._storage_path(storage_name).unlink(missing_ok=True)
        except OSError:
            logger.warning("stored_file_not_removed", extra={"document_id": document.id})

    # -- maintenance -------------------------------------------------------------------------
    def rebuild_index(self, collection_id: str) -> int:
        """Re-embed every indexed chunk of a collection from the database."""
        self.index.reset(collection_id)
        total = 0
        with self.database.session() as session:
            documents = DocumentRepository(session)
            # Finish deletions that were interrupted by an index failure.
            for document in documents.list_by_status(collection_id, "deleting"):
                self._purge(session, document)
            for document in documents.list_indexed(collection_id):
                rows = ChunkRepository(session).for_document(document.id)
                self.index.add(collection_id, [r.id for r in rows], [r.text for r in rows])
                total += len(rows)
        return total
