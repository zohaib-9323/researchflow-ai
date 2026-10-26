"""Test doubles: a scripted ResearchLLM and a fake chat model for gateway tests."""

from __future__ import annotations

from collections import deque
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatResult
from langchain_core.runnables import RunnableLambda

from app.llm.structured_outputs import (
    DraftAnswer,
    DraftClaim,
    LLMEvaluation,
    QuestionClassification,
    ReportDraft,
    ReportFinding,
    RevisedSearch,
)
from app.schemas.research import AnswerClaim, EvidencePassage
from app.utils.citations import content_tokens


def first_matching_sentence(question: str, evidence: list[EvidencePassage]) -> DraftClaim | None:
    """Deterministic 'answer': the evidence sentence sharing most words with the question."""
    wanted = content_tokens(question)
    best: tuple[int, str, str] | None = None
    for passage in evidence:
        for sentence in passage.text.replace("\n", " ").split(". "):
            overlap = len(content_tokens(sentence) & wanted)
            if overlap and (best is None or overlap > best[0]):
                best = (overlap, sentence.strip().rstrip("."), passage.source_id)
    if best is None:
        return None
    return DraftClaim(statement=best[1] + ".", source_ids=[best[2]])


class ScriptedLLM:
    """A ResearchLLM whose responses can be scripted per method.

    Each `script[method]` is a list of results or exceptions consumed in order; when a
    script is empty the deterministic default behaviour is used.
    """

    def __init__(self, configured: bool = True, **script: list[Any]) -> None:
        self._configured = configured
        self.script: dict[str, deque] = {name: deque(items) for name, items in script.items()}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @property
    def configured(self) -> bool:
        return self._configured

    def _scripted(self, method: str, **kwargs: Any) -> Any:
        self.calls.append((method, kwargs))
        queue = self.script.get(method)
        if queue:
            item = queue.popleft()
            if isinstance(item, BaseException):
                raise item
            return item(**kwargs) if callable(item) else item
        return None

    def call_names(self) -> list[str]:
        return [name for name, _ in self.calls]

    # ResearchLLM protocol -------------------------------------------------------------
    def classify(self, question: str, history: str) -> QuestionClassification:
        result = self._scripted("classify", question=question, history=history)
        return result or QuestionClassification(
            question_type="document_question",
            needs_retrieval=True,
            standalone_question=question,
            search_query=question,
            reason="default",
        )

    def draft_answer(
        self, question: str, evidence: list[EvidencePassage], feedback: str = ""
    ) -> DraftAnswer:
        result = self._scripted(
            "draft_answer", question=question, evidence=evidence, feedback=feedback
        )
        if result is not None:
            return result
        claim = first_matching_sentence(question, evidence)
        return DraftAnswer(claims=[claim] if claim else [], insufficient_evidence=claim is None)

    def revise_answer(
        self,
        question: str,
        evidence: list[EvidencePassage],
        draft: list[AnswerClaim],
        feedback: str,
    ) -> DraftAnswer:
        result = self._scripted(
            "revise_answer", question=question, evidence=evidence, draft=draft, feedback=feedback
        )
        if result is not None:
            return result
        return DraftAnswer(
            claims=[
                DraftClaim(statement=c.text, source_ids=c.citations) for c in draft if c.supported
            ],
            insufficient_evidence=False,
        )

    def evaluate(
        self, question: str, evidence: list[EvidencePassage], claims: list[AnswerClaim]
    ) -> LLMEvaluation:
        result = self._scripted("evaluate", question=question, evidence=evidence, claims=claims)
        return result or LLMEvaluation(
            outcome="sufficient",
            explanation="All claims are supported.",
            relevant_source_ids=[p.source_id for p in evidence],
            unsupported_claim_numbers=[],
        )

    def revise_search(
        self, question: str, previous_queries: list[str], missing: str
    ) -> RevisedSearch:
        result = self._scripted(
            "revise_search", question=question, previous_queries=previous_queries, missing=missing
        )
        return result or RevisedSearch(search_query=f"{question} details", rationale="default")

    def write_report(
        self, question: str, answer: str, evidence: list[EvidencePassage]
    ) -> ReportDraft:
        result = self._scripted("write_report", question=question, answer=answer, evidence=evidence)
        return result or ReportDraft(
            title="Report",
            executive_summary="Summary.",
            key_findings=[
                ReportFinding(
                    statement="Finding.",
                    source_ids=[evidence[0].source_id],
                    kind="supported_finding",
                )
            ],
            agreements=[],
            disagreements=[],
            limitations=[],
            open_questions=[],
            conclusion="Conclusion.",
        )


def evaluation(outcome: str, **kwargs: Any) -> LLMEvaluation:
    return LLMEvaluation(
        outcome=outcome,  # type: ignore[arg-type]
        explanation=kwargs.pop("explanation", f"judged {outcome}"),
        relevant_source_ids=kwargs.pop("relevant_source_ids", []),
        unsupported_claim_numbers=kwargs.pop("unsupported_claim_numbers", []),
        **kwargs,
    )


class FakeStructuredChatModel(BaseChatModel):
    """Chat model whose structured-output runnable replays scripted results.

    Items are exceptions (raised), dicts (returned as include_raw results) or pydantic
    objects (wrapped as a successful parse).
    """

    responses: list[Any]
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-structured"

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:  # pragma: no cover
        raise NotImplementedError

    def with_structured_output(self, schema: Any, *, include_raw: bool = False, **kwargs: Any):
        def run(_input: Any) -> Any:
            item = self.responses[min(self.calls, len(self.responses) - 1)]
            self.calls += 1
            if isinstance(item, BaseException):
                raise item
            if isinstance(item, dict):
                return item
            return {"raw": None, "parsed": item, "parsing_error": None}

        return RunnableLambda(run)
