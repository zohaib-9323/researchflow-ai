"""End-to-end graph runs with a scripted LLM and real (hashing) retrieval."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langgraph.checkpoint.memory import InMemorySaver

from app.core.exceptions import InvalidModelOutputError, ProviderUnavailableError
from app.graph.nodes import NO_DOCUMENTS_MESSAGE, GraphDependencies
from app.graph.state import initial_turn_state
from app.graph.workflow import build_research_graph, thread_config
from app.llm.structured_outputs import DraftAnswer, DraftClaim, QuestionClassification
from app.persistence.checkpointer import create_sqlite_checkpointer
from tests.fakes import ScriptedLLM, evaluation

PASSAGES = {
    "solar": "Participating households reduced grid electricity consumption by 18 percent.",
    "cost": "The pilot cost 1.2 million dollars in total.",
}


def fake_retrieve(corpus: dict[str, str] | None = None, empty_first: int = 0):
    calls = {"n": 0}
    corpus = PASSAGES if corpus is None else corpus

    def retrieve(collection_id: str, query: str, k: int) -> list[Document]:
        calls["n"] += 1
        if calls["n"] <= empty_first:
            return []
        return [
            Document(
                page_content=text,
                metadata={
                    "chunk_id": f"{collection_id}-{key}",
                    "document_id": f"doc-{key}",
                    "document_name": f"{key}.pdf",
                    "page": 2,
                    "score": 0.9,
                },
            )
            for key, text in corpus.items()
        ][:k]

    retrieve.calls = calls  # type: ignore[attr-defined]
    return retrieve


def run(
    llm: ScriptedLLM,
    question: str = "How much did grid consumption fall?",
    retrieve=None,
    max_retries: int = 2,
    indexed: int = 3,
    thread: str = "t1",
    checkpointer=None,
):
    graph = build_research_graph(
        GraphDependencies(
            llm=llm,
            retrieve=retrieve or fake_retrieve(),
            indexed_document_count=lambda cid: indexed,
        ),
        checkpointer=checkpointer or InMemorySaver(),
    )
    return graph.invoke(initial_turn_state(question, "col", max_retries), thread_config(thread))


def nodes_run(result: dict) -> list[str]:
    return [step["node"] for step in result["steps"]]


def test_successful_answer_with_verified_citations() -> None:
    result = run(ScriptedLLM())
    assert result["status"] == "answered"
    assert nodes_run(result) == [
        "classify_question",
        "retrieve_documents",
        "generate_answer",
        "evaluate_evidence",
        "finalize_response",
    ]
    assert result["citations"][0]["document_name"] == "solar.pdf"
    assert result["citations"][0]["page"] == 2
    assert "[S1]" in result["answer"]
    assert result["retry_count"] == 0


def test_insufficient_evidence_triggers_new_search_then_passes() -> None:
    llm = ScriptedLLM(
        evaluate=[evaluation("insufficient_evidence", missing_information="cost figures")]
    )
    result = run(llm)
    assert result["status"] == "answered"
    assert nodes_run(result).count("retrieve_documents") == 2
    assert "revise_search" in nodes_run(result)
    assert result["retry_count"] == 1
    assert result["retrieval_attempts"] == 2
    revise_call = next(kw for name, kw in llm.calls if name == "revise_search")
    assert revise_call["missing"] == "cost figures"


def test_unsupported_claims_trigger_answer_revision() -> None:
    fixed = DraftAnswer(
        claims=[
            DraftClaim(statement="The pilot cost 1.2 million dollars in total.", source_ids=["S2"])
        ],
        insufficient_evidence=False,
    )
    llm = ScriptedLLM(
        evaluate=[evaluation("unsupported_claims", unsupported_claim_numbers=[1])],
        revise_answer=[fixed],
    )
    result = run(llm)
    assert result["status"] == "answered"
    assert "1.2 million" in result["answer"]
    assert "revise_answer" in nodes_run(result)
    assert "revise_search" not in nodes_run(result)
    assert result["answer_revisions"] == 1


def test_retry_budget_is_enforced_by_the_graph() -> None:
    llm = ScriptedLLM(evaluate=[evaluation("insufficient_evidence")] * 10)
    result = run(llm, max_retries=2)
    assert result["retry_count"] == 2
    assert result["retrieval_attempts"] == 3  # initial attempt + 2 retries
    assert llm.call_names().count("evaluate") == 3
    assert nodes_run(result)[-1] == "finalize_response"
    # Claims that passed deterministic checks are kept, but flagged as partial.
    assert result["status"] == "partial"
    assert any("may be incomplete" in limitation for limitation in result["limitations"])


def test_exhausted_budget_without_supported_claims_reports_insufficient_evidence() -> None:
    no_answer = DraftAnswer(
        claims=[], insufficient_evidence=True, missing_information="No data on battery costs."
    )
    llm = ScriptedLLM(draft_answer=[no_answer] * 5)
    result = run(llm, question="What did batteries cost?")
    assert result["status"] == "insufficient_evidence"
    assert result["citations"] == []
    assert "do not contain sufficient evidence" in result["answer"]
    assert any("battery costs" in limitation for limitation in result["limitations"])
    assert "evaluate" not in llm.call_names()  # deterministic check, no LLM judge needed


def test_empty_retrieval_retries_then_finalizes() -> None:
    retrieve = fake_retrieve(corpus={})
    result = run(ScriptedLLM(), retrieve=retrieve, max_retries=2)
    assert result["status"] == "insufficient_evidence"
    assert retrieve.calls["n"] == 3
    assert "generate_answer" not in nodes_run(result)


def test_empty_first_retrieval_recovers_after_revised_query() -> None:
    retrieve = fake_retrieve(empty_first=1)
    result = run(ScriptedLLM(), retrieve=retrieve)
    assert result["status"] == "answered"
    assert result["retry_count"] == 1


def test_fabricated_citations_are_never_returned() -> None:
    fabricated = DraftAnswer(
        claims=[
            DraftClaim(
                statement="Households reduced grid electricity consumption by 18 percent.",
                source_ids=["S1", "S99"],
            ),
            DraftClaim(statement="The mayor resigned in protest.", source_ids=["S42"]),
        ],
        insufficient_evidence=False,
    )
    llm = ScriptedLLM(draft_answer=[fabricated])
    result = run(llm)
    labels = {c["source_id"] for c in result["citations"]}
    assert labels <= {"S1", "S2"}
    assert "mayor" not in result["answer"]


def test_invalid_model_output_terminates_with_error() -> None:
    llm = ScriptedLLM(draft_answer=[InvalidModelOutputError()])
    result = run(llm)
    assert result["status"] == "error"
    assert result["error"]["code"] == "invalid_model_output"
    assert result["error"]["node"] == "generate_answer"
    assert nodes_run(result)[-1] == "finalize_response"
    assert result["citations"] == []


def test_provider_failure_terminates_safely() -> None:
    result = run(ScriptedLLM(classify=[ProviderUnavailableError()]))
    assert result["status"] == "error"
    assert result["error"]["code"] == "provider_unavailable"
    assert nodes_run(result) == ["classify_question", "finalize_response"]


def test_greeting_skips_retrieval() -> None:
    greeting = QuestionClassification(
        question_type="greeting_or_smalltalk",
        needs_retrieval=False,
        standalone_question="hi",
        search_query="",
        reason="greeting",
    )
    retrieve = fake_retrieve()
    result = run(ScriptedLLM(classify=[greeting]), question="hi!", retrieve=retrieve)
    assert result["status"] == "not_applicable"
    assert retrieve.calls["n"] == 0


def test_no_indexed_documents_short_circuits_without_llm() -> None:
    llm = ScriptedLLM()
    result = run(llm, indexed=0)
    assert result["status"] == "insufficient_evidence"
    assert result["answer"] == NO_DOCUMENTS_MESSAGE
    assert llm.calls == []


def test_overlong_question_rejected() -> None:
    llm = ScriptedLLM()
    result = run(llm, question="x" * 5000)
    assert result["status"] == "rejected"
    assert llm.calls == []


def test_threads_are_isolated_and_history_is_carried(tmp_path: Path) -> None:
    saver = create_sqlite_checkpointer(tmp_path / "cp.db")
    llm = ScriptedLLM()
    run(llm, question="How much did grid consumption fall?", thread="A", checkpointer=saver)
    run(llm, question="What did the pilot cost?", thread="B", checkpointer=saver)
    run(llm, question="And what about the cost?", thread="A", checkpointer=saver)

    histories = [kw["history"] for name, kw in llm.calls if name == "classify"]
    assert histories[:2] == ["", ""]  # new threads start empty
    assert "grid consumption" in histories[2]  # thread A remembers its own turn
    assert "pilot cost" not in histories[2]  # ...and nothing from thread B


def test_conversation_resumes_after_restart(tmp_path: Path) -> None:
    path = tmp_path / "cp.db"
    run(
        ScriptedLLM(),
        question="How much did grid consumption fall?",
        thread="T",
        checkpointer=create_sqlite_checkpointer(path),
    )
    # Simulate a process restart: brand-new checkpointer + graph on the same file.
    llm = ScriptedLLM()
    result = run(
        llm,
        question="What did the pilot cost?",
        thread="T",
        checkpointer=create_sqlite_checkpointer(path),
    )
    history = next(kw["history"] for name, kw in llm.calls if name == "classify")
    assert "grid consumption" in history
    assert len(result["messages"]) == 4  # 2 user + 2 assistant messages
    # Per-turn fields were reset: only this turn's steps are present.
    assert [s["node"] for s in result["steps"]].count("finalize_response") == 1
    assert result["question"] == "What did the pilot cost?"
