"""Typed state for the research workflow.

The state is checkpointed per conversation thread. `messages` accumulates across turns
(conversation memory, via the add_messages reducer); every other field is per-turn and
is reset by `initial_turn_state()` when a new question starts.

Complex values are stored as plain dicts (model_dump of the schemas) so checkpoints
stay JSON-friendly and version-tolerant.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages


class ResearchState(TypedDict, total=False):
    # conversation memory (persists across turns of the same thread)
    messages: Annotated[list[AnyMessage], add_messages]

    # request
    question: str
    collection_id: str
    standalone_question: str
    question_type: str

    # retrieval
    search_query: str
    search_queries: list[str]
    retrieved_documents: list[dict[str, Any]]  # EvidencePassage dumps

    # answer + evaluation
    draft_claims: list[dict[str, Any]]  # AnswerClaim dumps
    insufficient_flag: bool
    missing_information: str
    evaluation: dict[str, Any] | None  # EvaluationResult dump
    best_attempt: dict[str, Any] | None

    # retry budget (enforced by routing, not by prompts)
    retry_count: int
    max_retries: int
    retrieval_attempts: int
    answer_revisions: int

    # output
    answer: str
    claims: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    limitations: list[str]
    status: str
    error: dict[str, Any] | None
    final_reason: str
    steps: list[dict[str, str]]


def initial_turn_state(question: str, collection_id: str, max_retries: int) -> ResearchState:
    """Input for a new turn: resets every per-turn field, appends the user message."""
    return ResearchState(
        messages=[HumanMessage(content=question)],
        question=question,
        collection_id=collection_id,
        standalone_question=question,
        question_type="",
        search_query="",
        search_queries=[],
        retrieved_documents=[],
        draft_claims=[],
        insufficient_flag=False,
        missing_information="",
        evaluation=None,
        best_attempt=None,
        retry_count=0,
        max_retries=max_retries,
        retrieval_attempts=0,
        answer_revisions=0,
        answer="",
        claims=[],
        citations=[],
        limitations=[],
        status="running",
        error=None,
        final_reason="",
        steps=[],
    )
