"""Conditional edges. Pure functions of state, so every decision is unit-testable.

Termination: each loop (revise_search -> retrieve -> ... and revise_answer -> evaluate)
increments `retry_count`, and every loop entry is guarded by
`retry_count < max_retries`, so every path reaches finalize_response.
"""

from __future__ import annotations

from typing import Final, Literal

from app.graph.state import ResearchState

CLASSIFY: Final = "classify_question"
RETRIEVE: Final = "retrieve_documents"
GENERATE: Final = "generate_answer"
EVALUATE: Final = "evaluate_evidence"
REVISE_SEARCH: Final = "revise_search"
REVISE_ANSWER: Final = "revise_answer"
FINALIZE: Final = "finalize_response"

_TERMINAL_STATUSES = {"rejected", "not_applicable", "insufficient_evidence", "error"}


def has_error(state: ResearchState) -> bool:
    return state.get("status") == "error" or bool(state.get("error"))


def retries_remaining(state: ResearchState) -> bool:
    return state.get("retry_count", 0) < state.get("max_retries", 0)


def route_after_classify(
    state: ResearchState,
) -> Literal["retrieve_documents", "finalize_response"]:
    if has_error(state) or state.get("status") in _TERMINAL_STATUSES:
        return FINALIZE
    return RETRIEVE


def route_after_retrieve(
    state: ResearchState,
) -> Literal["generate_answer", "revise_search", "finalize_response"]:
    if has_error(state):
        return FINALIZE
    if not state.get("retrieved_documents"):
        return REVISE_SEARCH if retries_remaining(state) else FINALIZE
    return GENERATE


def route_after_generate(state: ResearchState) -> Literal["evaluate_evidence", "finalize_response"]:
    return FINALIZE if has_error(state) else EVALUATE


def route_after_evaluate(
    state: ResearchState,
) -> Literal["revise_search", "revise_answer", "finalize_response"]:
    if has_error(state):
        return FINALIZE
    evaluation = state.get("evaluation") or {}
    outcome = evaluation.get("outcome")
    if outcome == "sufficient":
        return FINALIZE
    if not retries_remaining(state):
        return FINALIZE
    if outcome == "unsupported_claims":
        # The evidence is there but the answer misuses it: fix the answer.
        return REVISE_ANSWER
    # insufficient_evidence / irrelevant_evidence: look for better evidence.
    return REVISE_SEARCH


def route_after_revise_search(
    state: ResearchState,
) -> Literal["retrieve_documents", "finalize_response"]:
    return FINALIZE if has_error(state) else RETRIEVE


def route_after_revise_answer(
    state: ResearchState,
) -> Literal["evaluate_evidence", "finalize_response"]:
    return FINALIZE if has_error(state) else EVALUATE
