from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, Query, Response, UploadFile, status

from app.api.dependencies import current_user, get_container
from app.container import Container
from app.core.exceptions import FileTooLargeError, ValidationFailedError
from app.persistence.models import Document
from app.persistence.repositories import CollectionRepository
from app.schemas.documents import (
    CollectionCreate,
    CollectionOut,
    DocumentOut,
    UploadResponse,
    UploadResult,
)
from app.services.user_service import CurrentUser, UserService

router = APIRouter(tags=["documents"])


def to_out(document: Document) -> DocumentOut:
    return DocumentOut.model_validate(document, from_attributes=True)


@router.get("/collections", response_model=list[CollectionOut])
def list_collections(container: Container = Depends(get_container)) -> list[CollectionOut]:
    with container.database.session() as session:
        repo = CollectionRepository(session)
        return [
            CollectionOut(
                id=c.id,
                name=c.name,
                indexed_documents=repo.indexed_document_count(c.id),
                created_at=c.created_at,
            )
            for c in repo.list()
        ]


@router.post("/collections", response_model=CollectionOut, status_code=status.HTTP_201_CREATED)
def create_collection(
    body: CollectionCreate,
    container: Container = Depends(get_container),
    user: CurrentUser | None = Depends(current_user),
) -> CollectionOut:
    UserService.require_admin(user)
    name = body.name.strip()
    if not name or len(name) > 200:
        raise ValidationFailedError("Collection names must be 1-200 characters.")
    with container.database.session() as session:
        repo = CollectionRepository(session)
        if repo.get_by_name(name) is not None:
            raise ValidationFailedError(f"A collection named '{name}' already exists.")
        collection = repo.create(name)
        return CollectionOut(
            id=collection.id,
            name=collection.name,
            indexed_documents=0,
            created_at=collection.created_at,
        )


@router.post("/documents/upload", response_model=UploadResponse)
async def upload_documents(
    files: list[UploadFile] = File(..., description="PDF or TXT files"),
    collection_id: str | None = Form(default=None),
    container: Container = Depends(get_container),
    user: CurrentUser | None = Depends(current_user),
) -> UploadResponse:
    """Upload and synchronously ingest files. Each file gets its own result entry."""
    UserService.require_admin(user)
    settings = container.settings
    if len(files) > settings.max_files_per_upload:
        raise ValidationFailedError(
            f"At most {settings.max_files_per_upload} files can be uploaded at once."
        )
    if collection_id is None:
        with container.database.session() as session:
            collection_id = CollectionRepository(session).get_or_create_default().id

    payloads: list[tuple[str | None, bytes]] = []
    results: list[UploadResult] = []
    for upload in files:
        # Read at most limit+1 bytes so oversized uploads are rejected without
        # buffering the entire file.
        content = await upload.read(settings.max_upload_bytes + 1)
        if len(content) > settings.max_upload_bytes:
            error = FileTooLargeError(
                f"'{upload.filename}' exceeds the {settings.max_upload_size_mb:g} MB limit."
            )
            results.append(
                UploadResult(
                    filename=upload.filename or "document",
                    ok=False,
                    error_code=error.code,
                    error_message=error.message,
                )
            )
            continue
        payloads.append((upload.filename, content))

    # Ingestion is CPU-bound (embedding); run it off the event loop.
    from starlette.concurrency import run_in_threadpool

    outcomes = await run_in_threadpool(container.documents.upload_many, collection_id, payloads)
    for outcome in outcomes:
        document_ok = outcome.document is not None and outcome.document.status == "indexed"
        results.append(
            UploadResult(
                filename=outcome.filename,
                ok=document_ok,
                document=to_out(outcome.document) if outcome.document else None,
                error_code=outcome.error_code
                or (outcome.document.error_code if outcome.document else None),
                error_message=outcome.error_message
                or (outcome.document.error_message if outcome.document else None),
            )
        )
    succeeded = sum(r.ok for r in results)
    return UploadResponse(results=results, succeeded=succeeded, failed=len(results) - succeeded)


@router.get("/documents", response_model=list[DocumentOut])
def list_documents(
    collection_id: str | None = Query(default=None, max_length=64),
    container: Container = Depends(get_container),
) -> list[DocumentOut]:
    return [to_out(d) for d in container.documents.list_documents(collection_id)]


@router.get("/documents/{document_id}", response_model=DocumentOut)
def get_document(document_id: str, container: Container = Depends(get_container)) -> DocumentOut:
    return to_out(container.documents.get_document(document_id))


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: str,
    container: Container = Depends(get_container),
    user: CurrentUser | None = Depends(current_user),
) -> Response:
    UserService.require_admin(user)
    container.documents.delete_document(document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
