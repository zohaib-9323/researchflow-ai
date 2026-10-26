from __future__ import annotations

import pytest
from langchain_core.documents import Document

from app.container import Container
from app.core.exceptions import DuplicateDocumentError, IndexingError, NotFoundError
from app.ingestion.indexing import FaissIndexManager, IndexModelMismatchError
from app.llm.embeddings import HashingEmbeddings
from app.persistence.repositories import ChunkRepository, CollectionRepository
from app.schemas.research import EvidencePassage
from app.services.retrieval_service import merge_evidence
from tests.conftest import make_pdf, sample_bytes


def test_upload_indexes_and_retrieves_with_metadata(
    container: Container, collection_id: str, indexed_samples: dict[str, str]
) -> None:
    results = container.retrieval.retrieve(collection_id, "nitrate limit wells Hollow Creek")
    assert results
    top = results[0]
    assert top.metadata["document_name"] == "larkspur_water_quality.txt"
    assert top.metadata["document_id"] == indexed_samples["larkspur_water_quality.txt"]
    assert top.metadata["page"] is None
    assert any("nitrate" in r.page_content.lower() for r in results)


def test_pdf_page_numbers_survive_to_retrieval(container: Container, collection_id: str) -> None:
    pdf = make_pdf(["Introduction to the survey.", "The turbine produced 42 megawatts."])
    document = container.documents.upload(collection_id, "turbines.pdf", pdf)
    assert (document.status, document.page_count) == ("indexed", 2)
    hit = container.retrieval.retrieve(collection_id, "turbine megawatts")[0]
    assert (hit.metadata["document_name"], hit.metadata["page"]) == ("turbines.pdf", 2)


def test_duplicate_upload_rejected(container: Container, collection_id: str) -> None:
    content = sample_bytes("larkspur_council_memo.txt")
    container.documents.upload(collection_id, "memo.txt", content)
    with pytest.raises(DuplicateDocumentError):
        container.documents.upload(collection_id, "memo-copy.txt", content)


def test_failed_extraction_is_recorded_not_indexed(
    container: Container, collection_id: str
) -> None:
    document = container.documents.upload(collection_id, "scan.pdf", make_pdf(["", ""]))
    assert document.status == "failed"
    assert document.error_code == "ocr_required"
    assert document.chunk_count == 0
    with container.database.session() as session:
        assert ChunkRepository(session).ids_for_document(document.id) == []


def test_indexing_failure_rolls_back(
    container: Container, collection_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(*args: object, **kwargs: object) -> None:
        raise IndexingError("disk full")

    monkeypatch.setattr(container.index, "add", explode)
    document = container.documents.upload(
        collection_id, "w.txt", sample_bytes("larkspur_water_quality.txt")
    )
    assert (document.status, document.error_code) == ("failed", "indexing_failed")
    with container.database.session() as session:
        assert ChunkRepository(session).ids_for_document(document.id) == []
        assert CollectionRepository(session).indexed_document_count(collection_id) == 0


def test_failed_document_can_be_retried(
    container: Container, collection_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = sample_bytes("larkspur_water_quality.txt")
    original_add = container.index.add
    monkeypatch.setattr(
        container.index, "add", lambda *a, **k: (_ for _ in ()).throw(IndexingError("x"))
    )
    assert container.documents.upload(collection_id, "w.txt", content).status == "failed"
    monkeypatch.setattr(container.index, "add", original_add)
    assert container.documents.upload(collection_id, "w.txt", content).status == "indexed"
    assert len(container.documents.list_documents(collection_id)) == 1


def test_deleted_document_is_no_longer_searchable(
    container: Container, collection_id: str, indexed_samples: dict[str, str]
) -> None:
    doc_id = indexed_samples["larkspur_water_quality.txt"]
    container.documents.delete_document(doc_id)
    results = container.retrieval.retrieve(collection_id, "nitrate wells Hollow Creek")
    assert all(r.metadata["document_id"] != doc_id for r in results)
    with pytest.raises(NotFoundError):
        container.documents.get_document(doc_id)


def test_delete_with_index_failure_hides_document(
    container: Container,
    collection_id: str,
    indexed_samples: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc_id = indexed_samples["larkspur_water_quality.txt"]

    def fail_remove(*args: object, **kwargs: object) -> None:
        raise IndexingError("The vector index could not be updated.")

    monkeypatch.setattr(container.index, "remove", fail_remove)
    with pytest.raises(IndexingError):
        container.documents.delete_document(doc_id)
    # Vectors are still in FAISS, but the database filter keeps them out of results.
    results = container.retrieval.retrieve(collection_id, "nitrate wells Hollow Creek")
    assert all(r.metadata["document_id"] != doc_id for r in results)
    assert doc_id not in {d.id for d in container.documents.list_documents()}
    # A rebuild completes the interrupted deletion.
    monkeypatch.undo()
    container.documents.rebuild_index(collection_id)
    assert container.index.count(collection_id) == sum(
        d.chunk_count for d in container.documents.list_documents(collection_id)
    )


def test_collections_are_isolated(
    container: Container, collection_id: str, indexed_samples: dict[str, str]
) -> None:
    with container.database.session() as session:
        other = CollectionRepository(session).create("Other").id
    container.documents.upload(other, "turbines.txt", b"Wind turbines produced 42 megawatts.")
    assert all(
        r.metadata["document_name"] == "turbines.txt"
        for r in container.retrieval.retrieve(other, "nitrate wells solar pilot")
    )
    assert all(
        r.metadata["document_name"] != "turbines.txt"
        for r in container.retrieval.retrieve(collection_id, "wind turbines megawatts")
    )


def test_rebuild_restores_search(
    container: Container, collection_id: str, indexed_samples: dict[str, str]
) -> None:
    container.index.reset(collection_id)
    assert container.retrieval.retrieve(collection_id, "nitrate") == []
    rebuilt = container.documents.rebuild_index(collection_id)
    assert rebuilt == container.index.count(collection_id) > 0
    assert container.retrieval.retrieve(collection_id, "nitrate wells")


def test_index_model_mismatch_detected(tmp_path) -> None:
    first = FaissIndexManager(tmp_path, HashingEmbeddings(), "model-a")
    first.add("col", [1], ["hello world"])
    second = FaissIndexManager(tmp_path, HashingEmbeddings(), "model-b")
    with pytest.raises(IndexModelMismatchError):
        second.search("col", "hello", 3)


def test_index_rejects_unsafe_collection_ids(tmp_path) -> None:
    manager = FaissIndexManager(tmp_path, HashingEmbeddings(), "m")
    with pytest.raises(IndexingError):
        manager.add("../escape", [1], ["x"])


def _doc(chunk_id: str, score: float) -> Document:
    return Document(
        page_content=f"text {chunk_id}",
        metadata={
            "chunk_id": chunk_id,
            "document_id": "d",
            "document_name": "n.txt",
            "page": None,
            "score": score,
        },
    )


def test_merge_evidence_keeps_labels_stable_and_dedupes() -> None:
    first = merge_evidence([], [_doc("a", 0.5), _doc("b", 0.9)], limit=5)
    assert [(p.source_id, p.chunk_id) for p in first] == [("S1", "b"), ("S2", "a")]
    second = merge_evidence(first, [_doc("a", 0.5), _doc("c", 0.4)], limit=5)
    assert [(p.source_id, p.chunk_id) for p in second] == [("S1", "b"), ("S2", "a"), ("S3", "c")]
    capped = merge_evidence(second, [_doc("d", 0.99), _doc("e", 0.1)], limit=4)
    assert [p.chunk_id for p in capped] == ["b", "a", "c", "d"]


def test_merge_evidence_continues_numbering_after_pruning() -> None:
    kept = [
        EvidencePassage(
            source_id="S3", chunk_id="x", document_id="d", document_name="n", text="t", score=0.5
        )
    ]
    merged = merge_evidence(kept, [_doc("y", 0.6)], limit=5)
    assert [p.source_id for p in merged] == ["S3", "S4"]
