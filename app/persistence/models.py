"""ORM models for application data (documents, chunks, conversations, reports)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.persistence.database import Base


def new_id() -> str:
    return uuid.uuid4().hex


def utcnow() -> datetime:
    return datetime.now(UTC)


class Collection(Base):
    __tablename__ = "collections"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(200), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    documents: Mapped[list[Document]] = relationship(back_populates="collection")


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("collection_id", "content_hash", name="uq_doc_hash"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=new_id)
    collection_id: Mapped[str] = mapped_column(ForeignKey("collections.id"), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    file_type: Mapped[str] = mapped_column(String(16))
    size_bytes: Mapped[int] = mapped_column(Integer)
    content_hash: Mapped[str] = mapped_column(String(64))
    # Stored file name inside the upload dir (never exposed through the API).
    storage_name: Mapped[str] = mapped_column(String(128))
    # processing -> indexed | failed ; indexed -> deleting (hidden) -> removed
    status: Mapped[str] = mapped_column(String(32), default="processing", index=True)
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    page_count: Mapped[int | None] = mapped_column(Integer)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    collection: Mapped[Collection] = relationship(back_populates="documents")
    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )


class Chunk(Base):
    """A searchable passage. Its integer `id` is the vector id inside FAISS."""

    __tablename__ = "chunks"
    # Never reuse ids of deleted chunks: a stale vector must not point at new text.
    __table_args__ = {"sqlite_autoincrement": True}  # noqa: RUF012

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chunk_uid: Mapped[str] = mapped_column(String(64), unique=True, default=new_id)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    collection_id: Mapped[str] = mapped_column(String(64), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    page_number: Mapped[int | None] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)

    document: Mapped[Document] = relationship(back_populates="chunks")


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=new_id)
    collection_id: Mapped[str] = mapped_column(ForeignKey("collections.id"), index=True)
    title: Mapped[str] = mapped_column(String(300))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    turns: Mapped[list[ResearchTurn]] = relationship(
        back_populates="conversation", order_by="ResearchTurn.created_at"
    )


class ResearchTurn(Base):
    """One question and its final, validated response."""

    __tablename__ = "research_turns"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=new_id)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), index=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32))
    claims: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    citations: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    evidence: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    evaluation: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    limitations: Mapped[list[str]] = mapped_column(JSON, default=list)
    execution: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    conversation: Mapped[Conversation] = relationship(back_populates="turns")


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=new_id)
    turn_id: Mapped[str] = mapped_column(ForeignKey("research_turns.id"), index=True)
    conversation_id: Mapped[str] = mapped_column(String(64), index=True)
    title: Mapped[str] = mapped_column(String(300))
    content: Mapped[dict[str, Any]] = mapped_column(JSON)
    markdown: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
