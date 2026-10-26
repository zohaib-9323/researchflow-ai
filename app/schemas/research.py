"""API and workflow schemas for research questions, evidence and citations."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

ResearchStatus = Literal[
    "answered",  # passed evaluation
    "partial",  # best supported answer after the retry budget, with limitations
    "insufficient_evidence",  # documents do not support an answer
    "not_applicable",  # not a document question (e.g. greeting)
    "rejected",  # invalid request
    "error",  # unrecoverable error (provider/configuration)
]


class EvidencePassage(BaseModel):
    """A retrieved chunk, traceable to an actual indexed passage."""

    source_id: str  # per-answer label shown to the model and user, e.g. "S1"
    chunk_id: str  # stable chunk uid in the database
    document_id: str
    document_name: str
    page: int | None = None
    text: str
    score: float


class Citation(BaseModel):
    source_id: str
    chunk_id: str
    document_id: str
    document_name: str
    page: int | None = None
    excerpt: str


class AnswerClaim(BaseModel):
    text: str
    citations: list[str] = Field(default_factory=list, description="source_ids")
    supported: bool = True


class EvaluationResult(BaseModel):
    outcome: Literal[
        "sufficient", "insufficient_evidence", "unsupported_claims", "irrelevant_evidence"
    ]
    explanation: str
    relevant_source_ids: list[str] = Field(default_factory=list)
    unsupported_claims: list[int] = Field(default_factory=list)
    missing_information: str | None = None
    deterministic_issues: list[str] = Field(default_factory=list)
    llm_judged: bool = True


class ExecutionStep(BaseModel):
    node: str
    summary: str


class ExecutionSummary(BaseModel):
    steps: list[ExecutionStep] = Field(default_factory=list)
    retrieval_attempts: int = 0
    answer_revisions: int = 0
    retries_used: int = 0
    max_retries: int = 0
    passed_evaluation: bool = False
    final_reason: str = ""
    search_queries: list[str] = Field(default_factory=list)
    models_used: list[str] = Field(default_factory=list)


class ResearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    collection_id: str | None = Field(default=None, max_length=64)
    conversation_id: str | None = Field(default=None, max_length=64)

    @field_validator("question")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class ResearchResponse(BaseModel):
    conversation_id: str
    turn_id: str | None
    question: str
    status: ResearchStatus
    answer: str
    claims: list[AnswerClaim] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    evidence: list[EvidencePassage] = Field(default_factory=list)
    evaluation: EvaluationResult | None = None
    limitations: list[str] = Field(default_factory=list)
    execution: ExecutionSummary
    error: dict | None = None
    created_at: datetime | None = None


class ConversationSummary(BaseModel):
    id: str
    title: str
    collection_id: str
    turn_count: int
    created_at: datetime
    updated_at: datetime


class ConversationDetail(ConversationSummary):
    turns: list[ResearchResponse]
