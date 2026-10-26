"""Semantic retrieval over a collection, exposed as a LangChain retriever."""

from __future__ import annotations

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from app.ingestion.indexing import FaissIndexManager
from app.persistence.database import Database
from app.persistence.repositories import ChunkRepository
from app.schemas.research import EvidencePassage


class CollectionRetriever(BaseRetriever):
    """LangChain retriever bound to one collection.

    FAISS returns candidate vector ids; they are resolved against the database so only
    chunks of `indexed` documents in this collection can ever be returned.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: FaissIndexManager
    database: Database
    collection_id: str
    k: int = 5
    min_score: float = 0.0

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        # Over-fetch so filtered-out (deleted/stale) vectors do not starve results.
        hits = self.index.search(self.collection_id, query, self.k * 3)
        hits = [(vector_id, score) for vector_id, score in hits if score >= self.min_score]
        with self.database.session() as session:
            resolved = ChunkRepository(session).searchable_by_ids(
                self.collection_id, [vector_id for vector_id, _ in hits]
            )
        documents: list[Document] = []
        for vector_id, score in hits:
            if vector_id not in resolved:
                continue
            chunk, filename = resolved[vector_id]
            documents.append(
                Document(
                    page_content=chunk.text,
                    metadata={
                        "chunk_id": chunk.chunk_uid,
                        "document_id": chunk.document_id,
                        "document_name": filename,
                        "page": chunk.page_number,
                        "score": round(score, 4),
                    },
                )
            )
            if len(documents) >= self.k:
                break
        return documents


class RetrievalService:
    def __init__(
        self, index: FaissIndexManager, database: Database, top_k: int, min_score: float
    ) -> None:
        self.index = index
        self.database = database
        self.top_k = top_k
        self.min_score = min_score

    def retriever(self, collection_id: str, k: int | None = None) -> CollectionRetriever:
        return CollectionRetriever(
            index=self.index,
            database=self.database,
            collection_id=collection_id,
            k=k or self.top_k,
            min_score=self.min_score,
        )

    def retrieve(self, collection_id: str, query: str, k: int | None = None) -> list[Document]:
        return self.retriever(collection_id, k).invoke(query)


def merge_evidence(
    existing: list[EvidencePassage], documents: list[Document], limit: int
) -> list[EvidencePassage]:
    """Add newly retrieved chunks to the evidence set with stable source labels.

    Existing passages keep their `S<n>` label across retries, new chunks get the next
    free label, duplicates are ignored, and the set is capped at `limit` passages
    (keeping existing ones first, then the highest-scoring new ones).
    """
    merged = list(existing)
    known = {passage.chunk_id for passage in merged}
    next_number = (
        max((int(p.source_id[1:]) for p in merged if p.source_id[1:].isdigit()), default=0) + 1
    )
    ranked = sorted(documents, key=lambda doc: doc.metadata.get("score", 0.0), reverse=True)
    for document in ranked:
        if len(merged) >= limit:
            break
        meta = document.metadata
        if meta["chunk_id"] in known:
            continue
        merged.append(
            EvidencePassage(
                source_id=f"S{next_number}",
                chunk_id=meta["chunk_id"],
                document_id=meta["document_id"],
                document_name=meta["document_name"],
                page=meta.get("page"),
                text=document.page_content,
                score=float(meta.get("score", 0.0)),
            )
        )
        known.add(meta["chunk_id"])
        next_number += 1
    return merged
