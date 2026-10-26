"""Pydantic schemas for machine-readable model output (validated structured output).

Kept deliberately flat (strings, lists, literals) so they translate cleanly into the
provider's JSON-schema mode.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class QuestionClassification(BaseModel):
    question_type: Literal[
        "document_question", "follow_up", "greeting_or_smalltalk", "out_of_scope"
    ] = Field(description="Category of the user's message.")
    needs_retrieval: bool = Field(
        description="True when answering requires searching the uploaded documents."
    )
    standalone_question: str = Field(
        description="The question rewritten to be understandable without the conversation."
    )
    search_query: str = Field(description="A concise semantic search query for the documents.")
    reason: str = Field(description="One short sentence explaining the classification.")


class DraftClaim(BaseModel):
    statement: str = Field(description="One factual statement answering part of the question.")
    source_ids: list[str] = Field(
        description="Labels of the evidence passages (e.g. 'S2') that directly support it."
    )


class DraftAnswer(BaseModel):
    claims: list[DraftClaim] = Field(
        description="The answer as a list of claims, each backed by evidence passages."
    )
    insufficient_evidence: bool = Field(
        description="True if the passages do not contain enough information to answer."
    )
    missing_information: str = Field(
        default="", description="What information is missing, if anything."
    )


class LLMEvaluation(BaseModel):
    outcome: Literal[
        "sufficient", "insufficient_evidence", "unsupported_claims", "irrelevant_evidence"
    ] = Field(
        description=(
            "sufficient: every claim is supported and the question is answered; "
            "insufficient_evidence: passages are relevant but incomplete; "
            "unsupported_claims: some claims are not backed by their cited passages; "
            "irrelevant_evidence: passages do not relate to the question."
        )
    )
    explanation: str = Field(description="Brief, evidence-based justification.")
    relevant_source_ids: list[str] = Field(description="Passages relevant to the question.")
    unsupported_claim_numbers: list[int] = Field(
        description="1-based numbers of claims not supported by their cited passages."
    )
    missing_information: str = Field(default="", description="What is still missing.")
    suggested_search_query: str = Field(
        default="", description="A better search query if more evidence is needed."
    )


class RevisedSearch(BaseModel):
    search_query: str = Field(description="A reformulated search query.")
    rationale: str = Field(description="Why this query should find better evidence.")


class ReportFinding(BaseModel):
    statement: str
    source_ids: list[str]
    kind: Literal["supported_finding", "interpretation"] = Field(
        description="supported_finding: directly stated in sources; interpretation: inferred."
    )


class ReportDraft(BaseModel):
    title: str
    executive_summary: str
    key_findings: list[ReportFinding]
    agreements: list[str] = Field(description="Points where sources agree (cite labels).")
    disagreements: list[str] = Field(description="Points where sources differ (cite labels).")
    limitations: list[str] = Field(description="Limitations and missing evidence.")
    open_questions: list[str] = Field(description="Unresolved questions.")
    conclusion: str
