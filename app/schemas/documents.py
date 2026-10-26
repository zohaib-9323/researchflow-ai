from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class CollectionOut(BaseModel):
    id: str
    name: str
    indexed_documents: int
    created_at: datetime


class CollectionCreate(BaseModel):
    name: str


class DocumentOut(BaseModel):
    id: str
    collection_id: str
    filename: str
    file_type: str
    size_bytes: int
    status: str
    error_code: str | None = None
    error_message: str | None = None
    page_count: int | None = None
    chunk_count: int
    created_at: datetime
    indexed_at: datetime | None = None


class UploadResult(BaseModel):
    filename: str
    ok: bool
    document: DocumentOut | None = None
    error_code: str | None = None
    error_message: str | None = None


class UploadResponse(BaseModel):
    results: list[UploadResult]
    succeeded: int
    failed: int
