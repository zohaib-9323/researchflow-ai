"""End-to-end agent runs through the real graph, tools, retrieval, checkpointer and DB,
with a scripted AgentLLM (no network, no keys)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.agent.schemas import (
    AgentReportDraft,
    AssessedFinding,
    PlanDraft,
    PlannedTask,
    ReportItem,
    TaskAssessment,
)
from app.agent.tools import WebResult
from app.agent.workflow import agent_config
from app.container import Container
from app.core.exceptions import ProviderUnavailableError
from app.main import create_app
from app.persistence.models import ResearchRun
from tests.agent.conftest import agent_container
from tests.agent.fakes import FakeWebSearch, ScriptedAgentLLM
from tests.conftest import sample_bytes

COMPARE = "Compare the savings reported by the pilot report and the council memo"
SIMPLE = "How many rooftop solar systems did the pilot install?"


def _load_samples(container: Container) -> str:
    from app.persistence.repositories import CollectionRepository

    with container.database.session() as session:
        collection_id = CollectionRepository(session).get_or_create_default().id
    for name in ("larkspur_solar_pilot_report.txt", "larkspur_council_memo.txt"):
        container.documents.upload(collection_id, name, sample_bytes(name))
    return collection_id


@pytest.fixture
def loaded(container: Container) -> Container:
    _load_samples(container)
    return container


def test_simple_question_skips_planner(loaded: Container, agent_llm: ScriptedAgentLLM) -> None:
    run = loaded.agent.start(SIMPLE)
    assert run.status == "completed", run.error
    assert run.mode == "simple"
    assert agent_llm.count("plan") == 0
    assert [t["status"] for t in run.tasks] == ["completed"]
    assert run.tool_calls[0]["tool"] == "search_documents"
    report = loaded.agent.report(run.id)
    assert "240 rooftop solar systems" in report.markdown
    assert report.content["citations"][0]["title"] == "larkspur_solar_pilot_report.txt"


def test_planned_run_success_path(loaded: Container, agent_llm: ScriptedAgentLLM) -> None:
    run = loaded.agent.start(COMPARE)
    assert run.status == "completed", run.error
    assert run.mode == "planned"
    assert agent_llm.count("plan") == 1
    assert [t["id"] for t in run.tasks] == ["T1", "T2"]
    kinds = [e["kind"] for e in run.events]
    assert kinds[0] == "plan"
    assert "verify" in kinds
    assert "report" in kinds
    assert kinds[-1] == "stop"
    assert run.verification["verified"] >= 2
    assert run.counters["tool_calls"] == 2
    # the report itself is produced through the create_research_report tool
    assert any(c["tool"] == "create_research_report" for c in run.tool_calls)
    assert run.has_report
    assert not run.resumable
    # snapshots never carry full document text
    assert all("text" not in s and len(s["excerpt"]) <= 300 for s in run.sources)


def test_only_task_relevant_tools_are_exposed(
    loaded: Container, agent_llm: ScriptedAgentLLM
) -> None:
    agent_llm.script["plan"].extend(
        [
            PlanDraft(
                tasks=[
                    PlannedTask(description="Find pilot savings", approach="document_search"),
                    PlannedTask(description="Compute the difference", approach="calculation"),
                ],
                rationale="r",
            )
        ]
    )
    loaded.agent.start(COMPARE)
    offered = [kw["tool_names"] for name, kw in agent_llm.calls if name == "choose_tools"]
    assert offered[0] == ["search_documents", "inspect_source"]
    assert offered[1] == ["calculate", "search_documents"]


def test_model_chooses_tools_dynamically(loaded: Container, agent_llm: ScriptedAgentLLM) -> None:
    agent_llm.script["plan"].append(
        PlanDraft(
            tasks=[PlannedTask(description="Compare savings figures", approach="comparison")],
            rationale="r",
        )
    )
    agent_llm.script["choose_tools"].extend(
        [
            [
                {"name": "search_documents", "args": {"query": "reduction grid electricity"}},
                {"name": "calculate", "args": {"expression": "18 - 12"}},  # not offered
            ],
        ]
    )
    agent_llm.script["assess"].append(
        TaskAssessment(outcome="needs_more_evidence", findings=[], summary="need a comparison")
    )
    agent_llm.script["choose_tools"].append(
        lambda task, tool_names, sources: [
            {
                "name": "compare_sources",
                "args": {"source_ids": [s.id for s in sources[:2]], "aspect": "savings"},
            }
        ]
    )
    run = loaded.agent.start(COMPARE)
    names = [(c["tool"], c["ok"]) for c in run.tool_calls]
    assert names[:3] == [
        ("search_documents", True),
        ("calculate", False),
        ("compare_sources", True),
    ]
    assert any("disagree" in gap for gap in run.verification["contradictions"])
    report = loaded.agent.report(run.id)
    assert "disagree" in report.markdown


def test_no_tool_call_falls_back_to_document_search(
    loaded: Container, agent_llm: ScriptedAgentLLM
) -> None:
    agent_llm.script["choose_tools"].append([])
    run = loaded.agent.start(SIMPLE)
    assert run.tool_calls[0]["tool"] == "search_documents"
    assert run.tool_calls[0]["ok"]
    assert any("falling back" in e["message"] for e in run.events)
    assert run.status == "completed"


def test_invalid_tool_arguments_from_model_are_recorded(
    loaded: Container, agent_llm: ScriptedAgentLLM
) -> None:
    agent_llm.script["choose_tools"].extend(
        [[{"name": "search_documents", "args": {"query": "x", "k": 99}}]]
    )
    run = loaded.agent.start(SIMPLE)
    first = run.tool_calls[0]
    assert not first["ok"]
    assert "Invalid arguments" in first["summary"]
    # the task had no sources, so it was retried and then succeeded
    assert run.tasks[0]["attempts"] == 2
    assert run.status == "completed"


def test_insufficient_evidence_then_retry_succeeds(
    loaded: Container, agent_llm: ScriptedAgentLLM
) -> None:
    agent_llm.script["assess"].append(
        TaskAssessment(
            outcome="needs_more_evidence",
            findings=[],
            missing_information="No installation count.",
            suggested_query="rooftop systems installed",
            summary="partial",
        )
    )
    run = loaded.agent.start(SIMPLE)
    assert run.status == "completed"
    assert run.tasks[0]["attempts"] == 2
    assert run.counters["retries"] == 1
    assert [e["kind"] for e in run.events].count("retry") == 1
    retry_task = [kw["task"] for name, kw in agent_llm.calls if name == "choose_tools"][1]
    assert "rooftop systems installed" in retry_task.notes


def test_retry_budget_is_enforced(tmp_path: Path) -> None:
    llm = ScriptedAgentLLM()
    unsure = TaskAssessment(outcome="needs_more_evidence", findings=[], summary="unsure")
    llm.script["assess"].extend([unsure] * 10)
    container = agent_container(tmp_path, llm, agent_max_task_retries=2)
    _load_samples(container)
    run = container.agent.start(SIMPLE)
    assert run.tasks[0]["attempts"] == 3  # 1 + 2 retries, then given up
    assert run.tasks[0]["status"] == "insufficient"
    assert run.status == "insufficient_evidence"


def test_not_answerable_produces_insufficient_evidence_report(
    loaded: Container, agent_llm: ScriptedAgentLLM
) -> None:
    agent_llm.script["assess"].append(
        TaskAssessment(outcome="not_answerable", findings=[], summary="Not in the documents.")
    )
    run = loaded.agent.start("What was the hourly wage of the installers?")
    assert run.status == "insufficient_evidence"
    assert agent_llm.count("write_report") == 0  # nothing to write from: no model call
    report = loaded.agent.report(run.id)
    assert report.content["key_findings"] == []
    assert report.content["citations"] == []
    assert "Contradictory or missing evidence" in report.markdown


def test_tool_call_limit_stops_the_run(tmp_path: Path) -> None:
    llm = ScriptedAgentLLM()
    container = agent_container(tmp_path, llm, agent_max_tool_calls=1)
    _load_samples(container)
    run = container.agent.start(COMPARE)
    assert run.status == "limit_reached"
    assert run.stop_reason == "tool_calls"
    assert run.counters["tool_calls"] == 1
    assert [t["status"] for t in run.tasks] == ["completed", "skipped"]
    assert any("limit" in item for item in container.agent.report(run.id).content["limitations"])


def test_tool_calls_per_step_are_capped_by_remaining_budget(tmp_path: Path) -> None:
    llm = ScriptedAgentLLM()
    llm.script["choose_tools"].append(
        [{"name": "search_documents", "args": {"query": f"solar pilot {i}"}} for i in range(5)]
    )
    container = agent_container(tmp_path, llm, agent_max_tool_calls=2)
    _load_samples(container)
    run = container.agent.start(SIMPLE)
    assert run.counters["tool_calls"] == 2
    assert len([c for c in run.tool_calls if c["tool"] == "search_documents"]) == 2


def test_time_limit_stops_the_run(tmp_path: Path) -> None:
    llm = ScriptedAgentLLM()
    container = agent_container(tmp_path, llm, agent_max_seconds=0.001)
    _load_samples(container)
    run = container.agent.start(COMPARE)
    assert run.stop_reason == "time"
    assert run.counters["tool_calls"] == 0
    assert run.status == "insufficient_evidence"


def test_plan_is_capped_at_max_tasks(tmp_path: Path) -> None:
    llm = ScriptedAgentLLM()
    llm.script["plan"].append(
        PlanDraft(
            tasks=[
                PlannedTask(description=f"Task {i}", approach="document_search") for i in range(8)
            ],
            rationale="r",
        )
    )
    container = agent_container(tmp_path, llm, agent_max_tasks=3)
    _load_samples(container)
    run = container.agent.start(COMPARE)
    assert len(run.tasks) == 3
    assert "capped at 3 tasks" in " ".join(run.limitations)


def test_citations_are_validated(loaded: Container, agent_llm: ScriptedAgentLLM) -> None:
    def assessment(task: object, sources: list) -> TaskAssessment:
        real = sources[0].id
        return TaskAssessment(
            outcome="complete",
            findings=[
                AssessedFinding(
                    statement="The pilot installed 240 rooftop solar systems.",
                    source_ids=[real, "D99"],
                ),
                AssessedFinding(statement="Invented claim.", source_ids=["W5"]),
                AssessedFinding(
                    statement="Penguins operate the nuclear reactor downtown.", source_ids=[real]
                ),
            ],
            summary="s",
        )

    agent_llm.script["assess"].append(assessment)
    agent_llm.script["write_report"].append(
        lambda findings, sources: AgentReportDraft(
            title="T",
            executive_summary="E",
            key_findings=[
                ReportItem(text=findings[0].statement, source_ids=findings[0].source_ids),
                ReportItem(text="A fabricated statistic.", source_ids=["D42"]),
            ],
            comparisons=[],
            contradictions_or_gaps=[],
            limitations=[],
            conclusion="C",
        )
    )
    run = loaded.agent.start(SIMPLE)
    # unknown ids dropped at assessment; unsupported statement rejected at verification
    assert run.verification == {**run.verification, "verified": 1, "rejected": 1}
    content = loaded.agent.report(run.id).content
    assert [f["text"] for f in content["key_findings"]] == [
        "The pilot installed 240 rooftop solar systems."
    ]
    cited = {c["source_id"] for c in content["citations"]}
    assert "D99" not in cited
    assert "D42" not in cited
    assert any(e["kind"] == "report" and "removed" in e["message"] for e in run.events)


def test_report_writer_failure_falls_back_to_verified_findings(
    loaded: Container, agent_llm: ScriptedAgentLLM
) -> None:
    from app.agent.tools import ToolFailedError

    agent_llm.script["write_report"].append(ToolFailedError("writer broke"))
    run = loaded.agent.start(SIMPLE)
    content = loaded.agent.report(run.id).content
    assert content["key_findings"]
    assert content["citations"]
    assert run.status == "completed"


def test_provider_failure_is_resumable(loaded: Container, agent_llm: ScriptedAgentLLM) -> None:
    agent_llm.script["assess"].append(ProviderUnavailableError())
    run = loaded.agent.start(COMPARE)
    assert run.status == "error"
    assert run.error["code"] == "provider_unavailable"
    assert run.resumable
    assert not run.has_report
    calls_before = run.counters["tool_calls"]

    resumed = loaded.agent.resume(run.id)
    assert resumed.status == "completed", resumed.error
    # resumed from the checkpoint: the first tool call was not repeated
    assert resumed.counters["tool_calls"] == calls_before + 1
    assert agent_llm.count("plan") == 1
    assert resumed.has_report


def test_interrupted_run_is_detected_and_resumed(
    loaded: Container, agent_llm: ScriptedAgentLLM
) -> None:
    from app.persistence.repositories import CollectionRepository

    with loaded.database.session() as session:
        cid = CollectionRepository(session).get_or_create_default().id
        run = ResearchRun(collection_id=cid, objective=SIMPLE, status="running")
        session.add(run)
        session.flush()
        run_id = run.id
    detail = loaded.agent.get(run_id)
    assert detail.status == "interrupted"
    assert detail.resumable
    resumed = loaded.agent.resume(run_id)
    assert resumed.status == "completed"


def test_completed_run_cannot_be_resumed(loaded: Container) -> None:
    from app.core.exceptions import ValidationFailedError

    run = loaded.agent.start(SIMPLE)
    with pytest.raises(ValidationFailedError):
        loaded.agent.resume(run.id)


def test_runs_use_isolated_threads(loaded: Container, agent_llm: ScriptedAgentLLM) -> None:
    first = loaded.agent.start(SIMPLE)
    second = loaded.agent.start("What did the pilot cost in total?")
    assert first.id != second.id
    limits = loaded.agent.limits()
    a = loaded.agent.graph.get_state(agent_config(first.id, limits)).values
    b = loaded.agent.graph.get_state(agent_config(second.id, limits)).values
    assert a["objective"] == SIMPLE
    assert b["objective"].startswith("What did the pilot cost")
    assert a["run_id"] == first.id
    assert b["run_id"] == second.id
    assert len(a["tool_calls"]) == len(first.tool_calls)


def test_web_search_disabled_behaviour(loaded: Container, agent_llm: ScriptedAgentLLM) -> None:
    agent_llm.script["plan"].append(
        PlanDraft(
            tasks=[PlannedTask(description="Find recent solar news", approach="web_search")],
            rationale="r",
        )
    )
    run = loaded.agent.start(COMPARE)
    assert run.tasks[0]["approach"] == "document_search"
    offered = [kw["tool_names"] for name, kw in agent_llm.calls if name == "choose_tools"]
    assert all("search_web" not in names for names in offered)
    assert not run.web_available
    assert any("Web search is not configured" in item for item in run.limitations)
    assert not any(s["kind"] == "web" for s in run.sources)


def test_web_search_enabled_uses_provider(tmp_path: Path) -> None:
    llm = ScriptedAgentLLM()
    web = FakeWebSearch(
        [
            WebResult(
                title="Solar news",
                url="https://news.example/solar",
                content="Rooftop solar systems were installed in 240 homes.",
            )
        ]
    )
    llm.script["plan"].append(
        PlanDraft(
            tasks=[
                PlannedTask(description="Find rooftop solar systems news", approach="web_search")
            ],
            rationale="r",
        )
    )
    llm.script["choose_tools"].append([{"name": "search_web", "args": {"query": "rooftop solar"}}])
    container = agent_container(tmp_path, llm, web_search=web)
    _load_samples(container)
    run = container.agent.start(COMPARE)
    assert run.web_available
    assert web.queries == ["rooftop solar"]
    web_sources = [s for s in run.sources if s["kind"] == "web"]
    assert web_sources[0]["id"] == "W1"
    assert web_sources[0]["url"] == "https://news.example/solar"
    citations = container.agent.report(run.id).content["citations"]
    assert citations[0]["url"] == "https://news.example/solar"


def test_empty_collection_without_web_is_rejected(container: Container) -> None:
    run = container.agent.start(SIMPLE)
    assert run.status == "error"
    assert run.error["code"] == "no_documents"
    assert not run.resumable


# -- API ---------------------------------------------------------------------------------
def test_agent_api_endpoints(loaded: Container) -> None:
    with TestClient(create_app(loaded)) as client:
        assert client.get("/health").json()["web_search_configured"] is False
        created = client.post("/agent/runs", json={"objective": SIMPLE})
        assert created.status_code == 202
        run_id = created.json()["id"]
        detail = client.get(f"/agent/runs/{run_id}").json()
        assert detail["status"] == "completed"
        assert detail["tasks"]
        assert client.get("/agent/runs").json()[0]["id"] == run_id
        report = client.get(f"/agent/runs/{run_id}/report").json()
        assert report["content"]["key_findings"]
        download = client.get(f"/agent/runs/{run_id}/report/download")
        assert download.headers["content-type"].startswith("text/markdown")
        assert "attachment" in download.headers["content-disposition"]
        assert download.text.startswith("# ")
        assert client.get("/agent/runs/nope").status_code == 404
        assert client.post("/agent/runs", json={"objective": "x"}).status_code == 422
        assert client.post(f"/agent/runs/{run_id}/resume").status_code == 422


def test_agent_requires_llm(tmp_path: Path) -> None:
    from app.container import build_container
    from app.core.exceptions import LLMNotConfiguredError
    from app.llm.embeddings import HashingEmbeddings
    from tests.conftest import make_settings

    container = build_container(
        make_settings(tmp_path, google_api_key_1=None),
        embeddings=HashingEmbeddings(),
        agent_background=False,
    )
    with pytest.raises(LLMNotConfiguredError):
        container.agent.start(SIMPLE)
