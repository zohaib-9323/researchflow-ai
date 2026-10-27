"""Data-access functions. Services own transactions; repositories only query/mutate."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.persistence.models import (
    Chunk,
    Collection,
    Conversation,
    Document,
    Report,
    ResearchTurn,
    utcnow,
)

DEFAULT_COLLECTION_NAME = "Default"


class CollectionRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, collection_id: str) -> Collection | None:
        return self.session.get(Collection, collection_id)

    def get_by_name(self, name: str) -> Collection | None:
        return self.session.scalar(select(Collection).where(Collection.name == name))

    def list(self) -> Sequence[Collection]:
        return self.session.scalars(select(Collection).order_by(Collection.created_at)).all()

    def create(self, name: str) -> Collection:
        collection = Collection(name=name)
        self.session.add(collection)
        self.session.flush()
        return collection

    def get_or_create_default(self) -> Collection:
        return self.get_by_name(DEFAULT_COLLECTION_NAME) or self.create(DEFAULT_COLLECTION_NAME)

    def indexed_document_count(self, collection_id: str) -> int:
        return (
            self.session.scalar(
                select(func.count(Document.id)).where(
                    Document.collection_id == collection_id, Document.status == "indexed"
                )
            )
            or 0
        )


class DocumentRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, document_id: str) -> Document | None:
        return self.session.get(Document, document_id)

    def find_by_hash(self, collection_id: str, content_hash: str) -> Document | None:
        return self.session.scalar(
            select(Document).where(
                Document.collection_id == collection_id, Document.content_hash == content_hash
            )
        )

    def list(self, collection_id: str | None = None) -> Sequence[Document]:
        query = select(Document).where(Document.status != "deleting")
        if collection_id is not None:
            query = query.where(Document.collection_id == collection_id)
        return self.session.scalars(query.order_by(Document.created_at.desc())).all()

    def list_indexed(self, collection_id: str | None = None) -> Sequence[Document]:
        query = select(Document).where(Document.status == "indexed")
        if collection_id is not None:
            query = query.where(Document.collection_id == collection_id)
        return self.session.scalars(query).all()

    def list_by_status(self, collection_id: str, status: str) -> Sequence[Document]:
        return self.session.scalars(
            select(Document).where(
                Document.collection_id == collection_id, Document.status == status
            )
        ).all()

    def add(self, document: Document) -> Document:
        self.session.add(document)
        self.session.flush()
        return document

    def delete(self, document: Document) -> None:
        self.session.delete(document)


class ChunkRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add_many(self, chunks: list[Chunk]) -> list[Chunk]:
        self.session.add_all(chunks)
        self.session.flush()
        return chunks

    def ids_for_document(self, document_id: str) -> list[int]:
        return list(self.session.scalars(select(Chunk.id).where(Chunk.document_id == document_id)))

    def for_document(self, document_id: str) -> Sequence[Chunk]:
        return self.session.scalars(
            select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.chunk_index)
        ).all()

    def delete_for_document(self, document_id: str) -> None:
        self.session.execute(delete(Chunk).where(Chunk.document_id == document_id))

    def searchable_by_ids(self, collection_id: str, ids: list[int]) -> dict[int, tuple[Chunk, str]]:
        """Return chunks (with their document filename) that are still searchable.

        This join is the safety net that keeps deleted, failed or foreign-collection
        content out of results even if the vector index is stale.
        """
        if not ids:
            return {}
        rows = self.session.execute(
            select(Chunk, Document.filename)
            .join(Document, Chunk.document_id == Document.id)
            .where(
                Chunk.id.in_(ids),
                Chunk.collection_id == collection_id,
                Document.collection_id == collection_id,
                Document.status == "indexed",
            )
        ).all()
        return {chunk.id: (chunk, filename) for chunk, filename in rows}


class ConversationRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, conversation_id: str) -> Conversation | None:
        return self.session.get(Conversation, conversation_id)

    def list(self, limit: int = 100, owner_email: str | None = None) -> Sequence[Conversation]:
        query = select(Conversation)
        if owner_email is not None:
            query = query.where(Conversation.owner_email == owner_email)
        return self.session.scalars(
            query.order_by(Conversation.updated_at.desc()).limit(limit)
        ).all()

    def create(
        self, collection_id: str, title: str, owner_email: str | None = None
    ) -> Conversation:
        conversation = Conversation(
            collection_id=collection_id, title=title[:300], owner_email=owner_email
        )
        self.session.add(conversation)
        self.session.flush()
        return conversation

    def turn_count(self, conversation_id: str) -> int:
        return (
            self.session.scalar(
                select(func.count(ResearchTurn.id)).where(
                    ResearchTurn.conversation_id == conversation_id
                )
            )
            or 0
        )

    def add_turn(self, conversation: Conversation, **fields: Any) -> ResearchTurn:
        turn = ResearchTurn(conversation_id=conversation.id, **fields)
        self.session.add(turn)
        conversation.updated_at = utcnow()
        self.session.flush()
        return turn

    def get_turn(self, turn_id: str) -> ResearchTurn | None:
        return self.session.get(ResearchTurn, turn_id)


class ReportRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, report_id: str) -> Report | None:
        return self.session.get(Report, report_id)

    def add(self, report: Report) -> Report:
        self.session.add(report)
        self.session.flush()
        return report

    def list_for_conversation(self, conversation_id: str) -> Sequence[Report]:
        return self.session.scalars(
            select(Report)
            .where(Report.conversation_id == conversation_id)
            .order_by(Report.created_at)
        ).all()
