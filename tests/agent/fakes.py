"""A deterministic, scriptable AgentLLM and a fake web-search provider."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from langchain_core.tools import BaseTool

from app.agent.schemas import (
    AgentReportDraft,
    AssessedFinding,
    ComparisonDraft,
    ComparisonPoint,
    Finding,
    PlanDraft,
    PlannedTask,
    ReportItem,
    ResearchTask,
    SourceRecord,
    TaskAssessment,
)
from app.agent.tools import WebResult
from app.utils.citations import content_tokens


def best_sentence(text: str, sources: list[SourceRecord]) -> tuple[str, str] | None:
    """The source sentence sharing most content words with `text`: (sentence, source id)."""
    wanted = content_tokens(text)
    best: tuple[int, str, str] | None = None
    for source in sources:
        for sentence in source.text.replace("\n", " ").split(". "):
            overlap = len(content_tokens(sentence) & wanted)
            if overlap and (best is None or overlap > best[0]):
                best = (overlap, sentence.strip().rstrip(".") + ".", source.id)
    return None if best is None else (best[1], best[2])


class ScriptedAgentLLM:
    """Each `script[method]` is consumed in order (values, callables or exceptions);
    when empty, a deterministic default based on the real sources is used."""

    def __init__(self, configured: bool = True, **script: list[Any]) -> None:
        self._configured = configured
        self.script: defaultdict[str, deque] = defaultdict(deque)
        for name, items in script.items():
            self.script[name].extend(items)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @property
    def configured(self) -> bool:
        return self._configured

    def _scripted(self, method: str, **kwargs: Any) -> Any:
        self.calls.append((method, kwargs))
        queue = self.script.get(method)
        if not queue:
            return None
        item = queue.popleft()
        if isinstance(item, BaseException):
            raise item
        return item(**kwargs) if callable(item) else item

    def count(self, method: str) -> int:
        return sum(1 for name, _ in self.calls if name == method)

    def plan(self, objective: str, max_tasks: int, web_available: bool) -> PlanDraft:
        scripted = self._scripted(
            "plan", objective=objective, max_tasks=max_tasks, web_available=web_available
        )
        return scripted or PlanDraft(
            tasks=[
                PlannedTask(
                    description="Find the savings reported by the solar pilot",
                    approach="document_search",
                ),
                PlannedTask(
                    description="Find the council memo assessment of savings",
                    approach="document_search",
                ),
            ],
            rationale="Two sources to compare.",
        )

    def choose_tools(
        self,
        objective: str,
        task: ResearchTask,
        sources: list[SourceRecord],
        tools: list[BaseTool],
    ) -> list[dict[str, Any]]:
        scripted = self._scripted(
            "choose_tools", task=task, tool_names=[t.name for t in tools], sources=sources
        )
        if scripted is not None:
            return scripted
        return [{"name": "search_documents", "args": {"query": task.description}, "id": "c1"}]

    def assess(
        self,
        objective: str,
        task: ResearchTask,
        tool_results: str,
        sources: list[SourceRecord],
    ) -> TaskAssessment:
        scripted = self._scripted("assess", task=task, sources=sources)
        if scripted is not None:
            return scripted
        match = best_sentence(task.description, sources)
        if match is None:
            return TaskAssessment(outcome="not_answerable", findings=[], summary="Nothing found.")
        sentence, source_id = match
        return TaskAssessment(
            outcome="complete",
            findings=[AssessedFinding(statement=sentence, source_ids=[source_id])],
            summary=sentence,
        )

    def compare(self, aspect: str, sources: list[SourceRecord]) -> ComparisonDraft:
        scripted = self._scripted("compare", aspect=aspect, sources=sources)
        return scripted or ComparisonDraft(
            points=[
                ComparisonPoint(
                    point=f"The sources give different figures for {aspect}.",
                    source_ids=[s.id for s in sources],
                    relation="disagree",
                )
            ],
            summary="They disagree.",
        )

    def write_report(
        self,
        objective: str,
        findings: list[Finding],
        gaps: list[str],
        sources: list[SourceRecord],
    ) -> AgentReportDraft:
        scripted = self._scripted("write_report", findings=findings, sources=sources)
        return scripted or AgentReportDraft(
            title="Solar savings review",
            executive_summary="Summary of verified findings.",
            key_findings=[ReportItem(text=f.statement, source_ids=f.source_ids) for f in findings],
            comparisons=[],
            contradictions_or_gaps=[],
            limitations=["Based on three short documents."],
            conclusion="See findings.",
        )


class FakeWebSearch:
    name = "fake-web"

    def __init__(self, results: list[WebResult] | None = None) -> None:
        self.results = results or []
        self.queries: list[str] = []

    def search(self, query: str, max_results: int) -> list[WebResult]:
        self.queries.append(query)
        return self.results[:max_results]
