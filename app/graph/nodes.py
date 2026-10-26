"""Graph node implementations.

Nodes are methods of `ResearchNodes`, which receives its dependencies explicitly so
the graph can be tested with a scripted LLM and an in-memory index. Each node returns a
partial state update. Domain errors (provider, configuration, indexing) are caught and
turned into an `error` state that routing sends straight to finalization.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from app.core.exceptions import ResearchFlowError
from app.core.logging import get_logger
from app.graph.state import ResearchState
from app.llm.research_llm import ResearchLLM
from app.llm.structured_outputs import DraftAnswer
from app.schemas.research import AnswerClaim, EvaluationResult, EvidencePassage
from app.services.retrieval_service import merge_evidence
from app.utils.citations import build_citations, render_answer, validate_claims

logger = get_logger(__name__)

NodeUpdate = dict[str, Any]

NO_DOCUMENTS_MESSAGE = (
    "This collection has no indexed documents yet. Upload PDF or TXT files in the "
    "Document Library, then ask your question again."
)
NOT_APPLICABLE_MESSAGE = (
    "I answer research questions using the documents in your collection. Ask me "
    "something about your uploaded documents."
)
INSUFFICIENT_MESSAGE = (
    "The uploaded documents do not contain sufficient evidence to answer this question."
)


@dataclass
class GraphDependencies:
    llm: ResearchLLM
    retrieve: Callable[[str, str, int], list[Document]]  # (collection_id, query, k)
    indexed_document_count: Callable[[str], int]
    top_k: int = 5
    max_evidence: int = 8
    max_question_chars: int = 2000
    max_history_turns: int = 6


def _step(state: ResearchState, node: str, summary: str) -> list[dict[str, str]]:
    return [*state.get("steps", []), {"node": node, "summary": summary}]


def _guard(node: str) -> Callable:
    """Convert domain errors raised inside a node into a terminal error state."""

    def decorator(fn: Callable[[Any, ResearchState], NodeUpdate]) -> Callable:
        @wraps(fn)
        def wrapper(self: Any, state: ResearchState) -> NodeUpdate:
            try:
                return fn(self, state)
            except ResearchFlowError as exc:
                logger.warning(
                    "graph_node_failed",
                    extra={
                        "node": node,
                        "error_code": exc.code,
                        "retry_count": state.get("retry_count", 0),
                    },
                )
                return {
                    "status": "error",
                    "error": {"code": exc.code, "message": exc.message, "node": node},
                    "steps": _step(state, node, f"Stopped: {exc.code}"),
                }

        return wrapper

    return decorator


def _evidence(state: ResearchState) -> list[EvidencePassage]:
    return [EvidencePassage.model_validate(item) for item in state.get("retrieved_documents", [])]


def _claims(state: ResearchState) -> list[AnswerClaim]:
    return [AnswerClaim.model_validate(item) for item in state.get("draft_claims", [])]


def _history(messages: list[AnyMessage], max_turns: int) -> str:
    previous = messages[:-1]  # last message is the current question
    lines = []
    for message in previous[-max_turns * 2 :] if max_turns else []:
        role = "User" if isinstance(message, HumanMessage) else "Assistant"
        content = message.content if isinstance(message.content, str) else str(message.content)
        lines.append(f"{role}: {content[:800]}")
    return "\n".join(lines)


class ResearchNodes:
    def __init__(self, deps: GraphDependencies) -> None:
        self.deps = deps

    # 1 ---------------------------------------------------------------------------------
    @_guard("classify_question")
    def classify_question(self, state: ResearchState) -> NodeUpdate:
        node = "classify_question"
        question = (state.get("question") or "").strip()
        if not question:
            return {
                "status": "rejected",
                "answer": "Please enter a question.",
                "final_reason": "Empty question.",
                "steps": _step(state, node, "Rejected: empty"),
            }
        if len(question) > self.deps.max_question_chars:
            return {
                "status": "rejected",
                "answer": f"Questions are limited to {self.deps.max_question_chars} characters.",
                "final_reason": "Question too long.",
                "steps": _step(state, node, "Rejected: too long"),
            }
        if self.deps.indexed_document_count(state["collection_id"]) == 0:
            return {
                "status": "insufficient_evidence",
                "answer": NO_DOCUMENTS_MESSAGE,
                "final_reason": "No indexed documents in the collection.",
                "limitations": ["No documents are indexed in this collection."],
                "steps": _step(state, node, "No indexed documents; nothing to search"),
            }
        history = _history(state.get("messages", []), self.deps.max_history_turns)
        result = self.deps.llm.classify(question, history)
        if not result.needs_retrieval or result.question_type in (
            "greeting_or_smalltalk",
            "out_of_scope",
        ):
            return {
                "status": "not_applicable",
                "question_type": result.question_type,
                "answer": NOT_APPLICABLE_MESSAGE,
                "final_reason": f"Not a document question ({result.question_type}).",
                "steps": _step(state, node, f"Classified as {result.question_type}"),
            }
        standalone = result.standalone_question.strip() or question
        query = result.search_query.strip() or standalone
        return {
            "question_type": result.question_type,
            "standalone_question": standalone,
            "search_query": query,
            "steps": _step(state, node, f"Classified as {result.question_type}; retrieval needed"),
        }

    # 2 ---------------------------------------------------------------------------------
    @_guard("retrieve_documents")
    def retrieve_documents(self, state: ResearchState) -> NodeUpdate:
        query = state.get("search_query") or state.get("standalone_question") or state["question"]
        documents = self.deps.retrieve(state["collection_id"], query, self.deps.top_k)
        previous = _evidence(state)
        evidence = merge_evidence(previous, documents, self.deps.max_evidence)
        new_count = len(evidence) - len(previous)
        attempts = state.get("retrieval_attempts", 0) + 1
        return {
            "retrieved_documents": [p.model_dump() for p in evidence],
            "search_queries": [*state.get("search_queries", []), query],
            "retrieval_attempts": attempts,
            "steps": _step(
                state,
                "retrieve_documents",
                f"Attempt {attempts}: {len(documents)} passages found, {new_count} new "
                f"({len(evidence)} in evidence set)",
            ),
        }

    # 3 ---------------------------------------------------------------------------------
    @_guard("generate_answer")
    def generate_answer(self, state: ResearchState) -> NodeUpdate:
        evidence = _evidence(state)
        feedback = ""
        if evaluation := state.get("evaluation"):
            feedback = (
                f"{evaluation.get('explanation', '')} {evaluation.get('missing_information') or ''}"
            )
        draft = self.deps.llm.draft_answer(state["standalone_question"], evidence, feedback.strip())
        return self._apply_draft(state, draft, "generate_answer", "Drafted")

    def _apply_draft(
        self, state: ResearchState, draft: DraftAnswer, node: str, verb: str
    ) -> NodeUpdate:
        claims, issues = validate_claims(draft.claims, _evidence(state))
        supported = sum(claim.supported for claim in claims)
        summary = f"{verb} {len(claims)} claims ({supported} with verified citations)"
        if issues:
            summary += f"; {len(issues)} citation issue(s)"
        return {
            "draft_claims": [c.model_dump() for c in claims],
            "insufficient_flag": draft.insufficient_evidence,
            "missing_information": draft.missing_information,
            "steps": _step(state, node, summary),
        }

    # 4 ---------------------------------------------------------------------------------
    @_guard("evaluate_evidence")
    def evaluate_evidence(self, state: ResearchState) -> NodeUpdate:
        node = "evaluate_evidence"
        evidence = _evidence(state)
        claims = _claims(state)
        issues = [
            f"Claim {n} lacks verified support."
            for n, claim in enumerate(claims, start=1)
            if not claim.supported
        ]
        # Deterministic checks first; the LLM judge is only used when they pass.
        if not evidence:
            evaluation = EvaluationResult(
                outcome="insufficient_evidence",
                explanation="No relevant passages were retrieved.",
                deterministic_issues=["No evidence retrieved."],
                llm_judged=False,
            )
        elif not claims:
            evaluation = EvaluationResult(
                outcome="insufficient_evidence",
                explanation="The answer step found no supportable claims in the evidence.",
                missing_information=state.get("missing_information") or None,
                deterministic_issues=["No claims produced."],
                llm_judged=False,
            )
        elif issues:
            evaluation = EvaluationResult(
                outcome="unsupported_claims",
                explanation="Some claims are not backed by valid citations to retrieved passages.",
                unsupported_claims=[
                    n for n, claim in enumerate(claims, start=1) if not claim.supported
                ],
                deterministic_issues=issues,
                llm_judged=False,
            )
        else:
            judged = self.deps.llm.evaluate(state["standalone_question"], evidence, claims)
            known = {p.source_id for p in evidence}
            flagged = sorted({n for n in judged.unsupported_claim_numbers if 1 <= n <= len(claims)})
            outcome = judged.outcome
            if outcome == "sufficient" and (flagged or state.get("insufficient_flag")):
                # A "sufficient" verdict that contradicts its own details is not trusted.
                outcome = "unsupported_claims" if flagged else "insufficient_evidence"
            evaluation = EvaluationResult(
                outcome=outcome,
                explanation=judged.explanation,
                relevant_source_ids=[s for s in judged.relevant_source_ids if s in known],
                unsupported_claims=flagged,
                missing_information=judged.missing_information or None,
                llm_judged=True,
            )
            for number in flagged:
                claims[number - 1].supported = False

        best = self._best_attempt(state, claims, evaluation)
        return {
            "evaluation": evaluation.model_dump(),
            "draft_claims": [c.model_dump() for c in claims],
            "best_attempt": best,
            "steps": _step(
                state,
                node,
                f"Outcome: {evaluation.outcome}"
                + (" (deterministic check)" if not evaluation.llm_judged else " (LLM judge)"),
            ),
        }

    @staticmethod
    def _best_attempt(
        state: ResearchState, claims: list[AnswerClaim], evaluation: EvaluationResult
    ) -> dict[str, Any]:
        score = sum(claim.supported for claim in claims) + (
            100 if evaluation.outcome == "sufficient" else 0
        )
        current = state.get("best_attempt")
        if current is not None and current["score"] >= score:
            return current
        return {
            "score": score,
            "claims": [c.model_dump() for c in claims],
            "evaluation": evaluation.model_dump(),
        }

    # 5 ---------------------------------------------------------------------------------
    @_guard("revise_search")
    def revise_search(self, state: ResearchState) -> NodeUpdate:
        evaluation = state.get("evaluation") or {}
        missing = evaluation.get("missing_information") or evaluation.get("explanation") or ""
        result = self.deps.llm.revise_search(
            state.get("standalone_question") or state["question"],
            state.get("search_queries", []),
            missing,
        )
        query = result.search_query.strip() or state.get("standalone_question", "")
        # Keep only passages the evaluator found relevant (if it said which).
        relevant = set(evaluation.get("relevant_source_ids") or [])
        evidence = state.get("retrieved_documents", [])
        if relevant:
            evidence = [p for p in evidence if p["source_id"] in relevant]
        retry = state.get("retry_count", 0) + 1
        return {
            "search_query": query,
            "retrieved_documents": evidence,
            "retry_count": retry,
            "steps": _step(
                state,
                "revise_search",
                f"Retry {retry}/{state.get('max_retries', 0)}: new search query",
            ),
        }

    # 6 ---------------------------------------------------------------------------------
    @_guard("revise_answer")
    def revise_answer(self, state: ResearchState) -> NodeUpdate:
        evaluation = state.get("evaluation") or {}
        feedback_parts = [evaluation.get("explanation", "")]
        feedback_parts += evaluation.get("deterministic_issues") or []
        if unsupported := evaluation.get("unsupported_claims"):
            feedback_parts.append(f"Unsupported claim numbers: {unsupported}")
        draft = self.deps.llm.revise_answer(
            state["standalone_question"],
            _evidence(state),
            _claims(state),
            "\n".join(part for part in feedback_parts if part),
        )
        retry = state.get("retry_count", 0) + 1
        update = self._apply_draft(
            {**state, "retry_count": retry},
            draft,
            "revise_answer",
            f"Retry {retry}/{state.get('max_retries', 0)}: revised to",
        )
        update["retry_count"] = retry
        update["answer_revisions"] = state.get("answer_revisions", 0) + 1
        return update

    # 7 ---------------------------------------------------------------------------------
    def finalize_response(self, state: ResearchState) -> NodeUpdate:
        status = state.get("status", "running")
        evidence = _evidence(state)
        limitations = list(state.get("limitations", []))
        retries = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 0)

        if status == "error" or state.get("error"):
            error = state.get("error") or {"code": "internal_error", "message": "Unknown error"}
            answer = f"The research request could not be completed: {error['message']}"
            return self._final(
                state,
                "error",
                answer,
                [],
                [],
                limitations,
                f"Stopped after an error in {error.get('node', 'the workflow')}.",
            )

        if status in ("rejected", "not_applicable", "insufficient_evidence") and not evidence:
            return self._final(
                state,
                status,
                state.get("answer", ""),
                [],
                [],
                limitations,
                state.get("final_reason", ""),
            )

        evaluation = state.get("evaluation") or {}
        if evaluation.get("outcome") == "sufficient":
            claims = [c for c in _claims(state) if c.supported]
            reason = "Evidence evaluation passed" + (
                f" after {retries} retr{'y' if retries == 1 else 'ies'}." if retries else "."
            )
            final_status = "answered"
        else:
            best = state.get("best_attempt") or {"claims": [], "evaluation": evaluation}
            claims = [
                c for c in (AnswerClaim.model_validate(x) for x in best["claims"]) if c.supported
            ]
            best_eval = best.get("evaluation") or {}
            if not evidence:
                reason = "No relevant passages were found"
            else:
                reason = f"Evidence remained {best_eval.get('outcome', 'insufficient')}"
            reason += f" after {retries} of {max_retries} retries."
            missing = best_eval.get("missing_information") or state.get("missing_information")
            if missing:
                limitations.append(f"Missing evidence: {missing}")
            if best_eval.get("explanation"):
                limitations.append(f"Evaluation: {best_eval['explanation']}")
            final_status = "partial" if claims else "insufficient_evidence"
            if claims:
                limitations.insert(
                    0,
                    "Only the claims that could be verified against the documents are shown; "
                    "the answer may be incomplete.",
                )

        citations = build_citations(claims, evidence)
        answer = render_answer(claims) if claims else INSUFFICIENT_MESSAGE
        return self._final(state, final_status, answer, claims, citations, limitations, reason)

    def _final(
        self,
        state: ResearchState,
        status: str,
        answer: str,
        claims: list[AnswerClaim],
        citations: list,
        limitations: list[str],
        reason: str,
    ) -> NodeUpdate:
        logger.info(
            "research_finalized",
            extra={
                "status": status,
                "retry_count": state.get("retry_count", 0),
                "retrieval_attempts": state.get("retrieval_attempts", 0),
                "error_category": (state.get("error") or {}).get("code"),
            },
        )
        return {
            "status": status,
            "answer": answer,
            "claims": [c.model_dump() for c in claims],
            "citations": [c.model_dump() for c in citations],
            "limitations": limitations,
            "final_reason": reason,
            "messages": [AIMessage(content=answer)],
            "steps": _step(state, "finalize_response", f"Final status: {status}. {reason}"),
        }
