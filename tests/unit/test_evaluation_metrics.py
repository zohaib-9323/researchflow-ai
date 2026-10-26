from __future__ import annotations

from app.evaluation.run import (
    citation_precision,
    load_dataset,
    reciprocal_rank,
    retrieval_hit,
    term_recall,
)


def test_dataset_is_well_formed() -> None:
    data = load_dataset()
    assert len(data) >= 10
    assert len({e.id for e in data}) == len(data)
    assert any(not e.answerable for e in data)
    assert all(e.expected_documents for e in data if e.answerable)


def test_retrieval_metrics() -> None:
    assert retrieval_hit(["a", "b"], ["b"]) is True
    assert retrieval_hit(["a"], ["b"]) is False
    assert reciprocal_rank(["a", "b", "c"], ["c"]) == 1 / 3
    assert reciprocal_rank([], ["c"]) == 0.0


def test_answer_metrics() -> None:
    assert citation_precision(["a", "a", "x"], ["a"]) == 2 / 3
    assert citation_precision([], ["a"]) is None
    assert term_recall("Costs were 1.2 MILLION dollars", ["1.2 million", "grant"]) == 0.5
    assert term_recall("anything", []) is None
