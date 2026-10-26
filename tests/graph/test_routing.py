from __future__ import annotations

import pytest

from app.graph import routing
from app.graph.state import ResearchState


def state(**fields: object) -> ResearchState:
    base: dict[str, object] = {
        "status": "running",
        "retry_count": 0,
        "max_retries": 2,
        "retrieved_documents": [{"source_id": "S1"}],
    }
    base.update(fields)
    return base  # type: ignore[return-value]


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({}, routing.RETRIEVE),
        ({"status": "rejected"}, routing.FINALIZE),
        ({"status": "not_applicable"}, routing.FINALIZE),
        ({"status": "insufficient_evidence"}, routing.FINALIZE),
        ({"status": "error", "error": {"code": "x"}}, routing.FINALIZE),
    ],
)
def test_after_classify(fields: dict, expected: str) -> None:
    assert routing.route_after_classify(state(**fields)) == expected


def test_after_retrieve() -> None:
    assert routing.route_after_retrieve(state()) == routing.GENERATE
    assert routing.route_after_retrieve(state(retrieved_documents=[])) == routing.REVISE_SEARCH
    exhausted = state(retrieved_documents=[], retry_count=2)
    assert routing.route_after_retrieve(exhausted) == routing.FINALIZE
    assert routing.route_after_retrieve(state(error={"code": "x"})) == routing.FINALIZE


@pytest.mark.parametrize(
    ("outcome", "retry_count", "expected"),
    [
        ("sufficient", 0, routing.FINALIZE),
        ("sufficient", 2, routing.FINALIZE),
        ("insufficient_evidence", 0, routing.REVISE_SEARCH),
        ("irrelevant_evidence", 1, routing.REVISE_SEARCH),
        ("unsupported_claims", 0, routing.REVISE_ANSWER),
        ("insufficient_evidence", 2, routing.FINALIZE),  # budget exhausted
        ("unsupported_claims", 2, routing.FINALIZE),
    ],
)
def test_after_evaluate(outcome: str, retry_count: int, expected: str) -> None:
    current = state(evaluation={"outcome": outcome}, retry_count=retry_count)
    assert routing.route_after_evaluate(current) == expected


def test_zero_retry_budget_never_loops() -> None:
    current = state(evaluation={"outcome": "insufficient_evidence"}, max_retries=0)
    assert routing.route_after_evaluate(current) == routing.FINALIZE
    assert routing.route_after_retrieve(state(retrieved_documents=[], max_retries=0)) == (
        routing.FINALIZE
    )


def test_errors_short_circuit_every_router() -> None:
    errored = state(status="error", error={"code": "provider_error"})
    for router in (
        routing.route_after_generate,
        routing.route_after_evaluate,
        routing.route_after_revise_search,
        routing.route_after_revise_answer,
    ):
        assert router(errored) == routing.FINALIZE
