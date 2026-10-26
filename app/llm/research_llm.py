"""Research-specific LLM operations used by the graph and report service.

`ResearchLLM` is the seam between the workflow and the model: the graph depends on
this protocol only, so tests inject a scripted fake instead of calling Gemini.
"""

from __future__ import annotations

from typing import Protocol

from app.llm.models import LLMGateway
from app.llm.prompts import (
    ANSWER_PROMPT,
    CLASSIFY_PROMPT,
    EVALUATE_PROMPT,
    REPORT_PROMPT,
    REVISE_ANSWER_PROMPT,
    REVISE_SEARCH_PROMPT,
    UNTRUSTED_DATA_RULES,
)
from app.llm.structured_outputs import (
    DraftAnswer,
    LLMEvaluation,
    QuestionClassification,
    ReportDraft,
    RevisedSearch,
)
from app.schemas.research import AnswerClaim, EvidencePassage
from app.utils.citations import format_claims, format_evidence


class ResearchLLM(Protocol):
    @property
    def configured(self) -> bool: ...

    def classify(self, question: str, history: str) -> QuestionClassification: ...

    def draft_answer(
        self, question: str, evidence: list[EvidencePassage], feedback: str = ""
    ) -> DraftAnswer: ...

    def revise_answer(
        self,
        question: str,
        evidence: list[EvidencePassage],
        draft: list[AnswerClaim],
        feedback: str,
    ) -> DraftAnswer: ...

    def evaluate(
        self, question: str, evidence: list[EvidencePassage], claims: list[AnswerClaim]
    ) -> LLMEvaluation: ...

    def revise_search(
        self, question: str, previous_queries: list[str], missing: str
    ) -> RevisedSearch: ...

    def write_report(
        self, question: str, answer: str, evidence: list[EvidencePassage]
    ) -> ReportDraft: ...


class GeminiResearchLLM:
    """LangChain-backed implementation (Gemini via the rotating key pool)."""

    def __init__(self, gateway: LLMGateway) -> None:
        self.gateway = gateway

    @property
    def configured(self) -> bool:
        return self.gateway.configured

    def classify(self, question: str, history: str) -> QuestionClassification:
        return self.gateway.structured(
            CLASSIFY_PROMPT,
            QuestionClassification,
            {"question": question, "history": history or "(none)", **_rules()},
            task="classify_question",
        )

    def draft_answer(
        self, question: str, evidence: list[EvidencePassage], feedback: str = ""
    ) -> DraftAnswer:
        return self.gateway.structured(
            ANSWER_PROMPT,
            DraftAnswer,
            {
                "question": question,
                "evidence": format_evidence(evidence),
                "feedback": f"Notes from a previous attempt:\n{feedback}" if feedback else "",
                **_rules(),
            },
            task="generate_answer",
        )

    def revise_answer(
        self,
        question: str,
        evidence: list[EvidencePassage],
        draft: list[AnswerClaim],
        feedback: str,
    ) -> DraftAnswer:
        return self.gateway.structured(
            REVISE_ANSWER_PROMPT,
            DraftAnswer,
            {
                "question": question,
                "evidence": format_evidence(evidence),
                "draft": format_claims(draft),
                "feedback": feedback or "(none)",
                **_rules(),
            },
            task="revise_answer",
        )

    def evaluate(
        self, question: str, evidence: list[EvidencePassage], claims: list[AnswerClaim]
    ) -> LLMEvaluation:
        return self.gateway.structured(
            EVALUATE_PROMPT,
            LLMEvaluation,
            {
                "question": question,
                "evidence": format_evidence(evidence),
                "claims": format_claims(claims),
                **_rules(),
            },
            task="evaluate_evidence",
        )

    def revise_search(
        self, question: str, previous_queries: list[str], missing: str
    ) -> RevisedSearch:
        return self.gateway.structured(
            REVISE_SEARCH_PROMPT,
            RevisedSearch,
            {
                "question": question,
                "previous_queries": "; ".join(previous_queries) or "(none)",
                "missing": missing or "(not specified)",
                **_rules(),
            },
            task="revise_search",
        )

    def write_report(
        self, question: str, answer: str, evidence: list[EvidencePassage]
    ) -> ReportDraft:
        return self.gateway.structured(
            REPORT_PROMPT,
            ReportDraft,
            {
                "question": question,
                "answer": answer or "(no validated answer)",
                "evidence": format_evidence(evidence),
                **_rules(),
            },
            task="write_report",
        )


def _rules() -> dict[str, str]:
    return {"security_rules": UNTRUSTED_DATA_RULES}
