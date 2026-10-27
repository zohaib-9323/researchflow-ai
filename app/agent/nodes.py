"""Node implementations of the research agent graph.

Each node returns a partial state update and appends user-visible events (what was done
and what came out of it — never hidden reasoning). Provider failures are not caught
here: they propagate out of the graph, leaving the last checkpoint intact so the run can
be resumed (see app/agent/service.py).
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document

from app.agent import routing
from app.agent.llm import AgentLLM
from app.agent.report import build_report, citations_for, empty_report, render_markdown
from app.agent.schemas import (
    AgentEvent,
    Approach,
    Finding,
    PlanDraft,
    ResearchTask,
    SourceRecord,
)
from app.agent.state import AgentState
from app.agent.tools import ToolContext, ToolExecutor, WebSearchProvider, build_tools, tools_for
from app.core.logging import get_logger
from app.utils.citations import MIN_LEXICAL_SUPPORT, lexical_support

logger = get_logger(__name__)

NodeUpdate = dict[str, Any]

MAX_CALLS_PER_STEP = 3  # tool calls the model may request for one task attempt
MAX_OBJECTIVE_CHARS = 4000

# Objectives that need several steps; anything else short is answered with one search.
_COMPLEX_MARKERS = re.compile(
    r"\b(compare|comparison|versus|vs\.?|differ|difference|contrast|trend|analy[sz]e|"
    r"evaluate|assess|pros|cons|impact|investigate|overview|summari[sz]e|report|"
    r"calculate|how much|percentage|both|each|all)\b",
    re.IGNORECASE,
)
_OUTCOME_LABELS = {
    "completed": "complete",
    "pending": "needs more evidence",
    "insufficient": "insufficient evidence",
}
_WEB_MARKERS = re.compile(r"\b(latest|current|today|news|recent|this year|online)\b", re.I)


@dataclass
class AgentDependencies:
    llm: AgentLLM
    retrieve: Callable[[str, str, int], list[Document]]  # (collection_id, query, k)
    indexed_document_count: Callable[[str], int]
    executor: ToolExecutor
    web_search: WebSearchProvider | None = None
    document_info: Callable[[str], dict[str, Any] | None] = field(default=lambda _doc_id: None)
    max_context_chars: int = 14000
    search_k: int = 5


def _event(kind: str, message: str, task_id: str | None = None) -> list[dict[str, Any]]:
    return [AgentEvent(kind=kind, message=message, task_id=task_id).model_dump()]  # type: ignore[arg-type]


def _tasks(state: AgentState) -> list[ResearchTask]:
    return [ResearchTask.model_validate(t) for t in state.get("tasks", [])]


def _sources(state: AgentState) -> dict[str, SourceRecord]:
    return {sid: SourceRecord.model_validate(s) for sid, s in state.get("sources", {}).items()}


def _dump_tasks(tasks: list[ResearchTask]) -> list[dict[str, Any]]:
    return [t.model_dump() for t in tasks]


def is_simple_objective(objective: str) -> bool:
    """A short, single question that one search task can answer (planner skipped)."""
    words = objective.split()
    return (
        len(words) <= 18
        and objective.count("?") <= 1
        and objective.count(".") <= 1
        and not _COMPLEX_MARKERS.search(objective)
    )


class AgentNodes:
    def __init__(self, deps: AgentDependencies) -> None:
        self.deps = deps

    # -- plan_research ------------------------------------------------------------------
    def plan_research(self, state: AgentState) -> NodeUpdate:
        objective = " ".join(state.get("objective", "").split())
        limits = state["limits"]
        web = state.get("web_available", False)
        limitations: list[str] = []
        if not web:
            limitations.append("Web search is not configured; only uploaded documents were used.")
        if not objective or len(objective) > MAX_OBJECTIVE_CHARS:
            return self._fail("invalid_objective", "The research objective is empty or too long.")
        if self.deps.indexed_document_count(state["collection_id"]) == 0 and not web:
            return self._fail(
                "no_documents",
                "This collection has no indexed documents and web search is not configured.",
            )

        if is_simple_objective(objective):
            approach: Approach = (
                "web_search" if web and _WEB_MARKERS.search(objective) else "document_search"
            )
            tasks = [ResearchTask(id="T1", description=objective, approach=approach)]
            return {
                "mode": "simple",
                "tasks": _dump_tasks(tasks),
                "limitations": limitations,
                "events": _event("plan", "Simple question: planning skipped, one search task."),
            }

        draft: PlanDraft = self.deps.llm.plan(objective, limits["max_tasks"], web)
        tasks = []
        for planned in draft.tasks[: limits["max_tasks"]]:
            description = " ".join(planned.description.split())[:500]
            if not description:
                continue
            planned_approach: Approach = planned.approach
            if planned_approach == "web_search" and not web:
                planned_approach = "document_search"
            tasks.append(
                ResearchTask(
                    id=f"T{len(tasks) + 1}",
                    description=description,
                    approach=planned_approach,
                    source_requirements=planned.source_requirements[:300],
                )
            )
        if len(draft.tasks) > limits["max_tasks"]:
            limitations.append(f"The plan was capped at {limits['max_tasks']} tasks.")
        if not tasks:  # an empty plan still gets one honest attempt
            tasks = [ResearchTask(id="T1", description=objective, approach="document_search")]
        summary = "; ".join(f"{t.id}: {t.description}" for t in tasks)
        return {
            "mode": "planned",
            "tasks": _dump_tasks(tasks),
            "limitations": limitations,
            "events": _event("plan", f"Planned {len(tasks)} task(s) — {summary}"),
        }

    # -- select_next_task ---------------------------------------------------------------
    def select_next_task(self, state: AgentState) -> NodeUpdate:
        tasks = _tasks(state)
        exceeded = routing.limit_exceeded(state)
        if exceeded is not None:
            label = {"tool_calls": "tool-call", "time": "time"}[exceeded]
            for task in tasks:
                if task.status in ("pending", "in_progress"):
                    task.status = "skipped"
            return {
                "tasks": _dump_tasks(tasks),
                "current_task_id": None,
                "stop_reason": exceeded,
                "events": _event("stop", f"Stopped researching: the {label} limit was reached."),
            }
        for task in tasks:
            if task.status == "pending":
                task.status = "in_progress"
                task.attempts += 1
                attempt = f" (attempt {task.attempts})" if task.attempts > 1 else ""
                return {
                    "tasks": _dump_tasks(tasks),
                    "current_task_id": task.id,
                    "last_tool_results": "",
                    "evaluation": None,
                    "events": _event(
                        "task", f"Started {task.id}{attempt}: {task.description}", task.id
                    ),
                }
        return {"current_task_id": None, "events": _event("task", "All tasks processed.")}

    # -- execute_research_tool ----------------------------------------------------------
    def execute_research_tool(self, state: AgentState) -> NodeUpdate:
        tasks = _tasks(state)
        task = next(t for t in tasks if t.id == state["current_task_id"])
        sources = _sources(state)
        limits = state["limits"]
        ctx = ToolContext(
            collection_id=state["collection_id"],
            task_id=task.id,
            sources=sources,
            max_sources=limits["max_sources"],
            retrieve=self.deps.retrieve,
            compare=self.deps.llm.compare,
            web_search=self.deps.web_search if state.get("web_available") else None,
            document_info=self.deps.document_info,
        )
        all_tools = build_tools(ctx)
        allowed = tools_for(task.approach, bool(state.get("web_available")))
        tools = {name: all_tools[name] for name in allowed}

        calls = self.deps.llm.choose_tools(
            state["objective"], task, list(sources.values()), list(tools.values())
        )
        events: list[dict[str, Any]] = []
        if not calls:
            query = task.notes.split("Try: ", 1)[-1] if "Try: " in task.notes else task.description
            calls = [{"name": "search_documents", "args": {"query": query[:300]}}]
            events += _event(
                "tool", "The model requested no tool; falling back to document search.", task.id
            )

        budget = limits["max_tool_calls"] - state.get("tool_call_count", 0)
        if len(calls) > min(budget, MAX_CALLS_PER_STEP):
            events += _event(
                "tool",
                f"Ran {min(budget, MAX_CALLS_PER_STEP)} of {len(calls)} requested tool calls "
                "(limit).",
                task.id,
            )
        calls = calls[: max(0, min(budget, MAX_CALLS_PER_STEP))]

        records = []
        touched: list[str] = []  # sources returned to this task (new or already known)
        results = []
        comparisons = []
        for call in calls:
            name = str(call.get("name", ""))
            raw_args = call.get("args")
            args: dict[str, Any] = raw_args if isinstance(raw_args, dict) else {}
            if name == "search_documents":
                args = {"k": self.deps.search_k, **args}
            record, output = self.deps.executor.run(tools, name, args, task.id)
            if name != "compare_sources":
                touched += output.source_ids
            records.append(record.model_dump())
            status = "ok" if record.ok else "failed"
            events += _event("tool", f"{name} → {status}: {record.summary}", task.id)
            results.append(
                {"tool": name, "ok": output.ok, "summary": output.summary, "data": output.data}
            )
            if name == "compare_sources" and output.ok and isinstance(output.data, dict):
                comparisons += [{**p, "task_id": task.id} for p in output.data.get("points", [])]

        task.source_ids = list(dict.fromkeys([*task.source_ids, *touched]))
        return {
            "tasks": _dump_tasks(tasks),
            "sources": {sid: s.model_dump() for sid, s in ctx.sources.items()},
            "tool_calls": records,
            "tool_call_count": state.get("tool_call_count", 0) + len(calls),
            "last_tool_results": json.dumps(results, default=str)[: self.deps.max_context_chars],
            "comparisons": comparisons,
            "events": events,
        }

    # -- evaluate_research_progress -----------------------------------------------------
    def evaluate_research_progress(self, state: AgentState) -> NodeUpdate:
        tasks = _tasks(state)
        task = next(t for t in tasks if t.id == state["current_task_id"])
        sources = _sources(state)
        task_sources = [sources[sid] for sid in task.source_ids if sid in sources]
        can_retry = task.attempts <= state["limits"]["max_task_retries"]

        findings: list[Finding] = []
        if not task_sources:  # nothing to assess: no LLM call needed
            outcome, missing, query = "needs_more_evidence", "No sources found.", ""
            summary = "No sources were found for this task."
        else:
            assessment = self.deps.llm.assess(
                state["objective"], task, state.get("last_tool_results", ""), task_sources
            )
            outcome = assessment.outcome
            missing, query, summary = (
                assessment.missing_information,
                assessment.suggested_query,
                assessment.summary,
            )
            for item in assessment.findings:
                ids = [sid for sid in dict.fromkeys(item.source_ids) if sid in sources]
                if ids and item.statement.strip():
                    findings.append(
                        Finding(statement=item.statement.strip(), source_ids=ids, task_id=task.id)
                    )

        retry = outcome == "needs_more_evidence" and can_retry
        if outcome == "complete" and findings:
            task.status = "completed"
        elif retry:
            task.status = "pending"
            task.notes = missing + (f" Try: {query}" if query else "")
        else:
            task.status = "insufficient"
            task.notes = missing or "The sources do not answer this task."
        task.summary = summary[:500]
        label = _OUTCOME_LABELS[task.status]
        return {
            "tasks": _dump_tasks(tasks),
            "findings": [f.model_dump() for f in findings],
            "evaluation": {"task_id": task.id, "outcome": outcome, "retry": retry},
            "events": _event(
                "evaluation",
                f"{task.id}: {label} — {len(findings)} finding(s), {len(task_sources)} source(s).",
                task.id,
            ),
        }

    # -- revise_research_plan -----------------------------------------------------------
    def revise_research_plan(self, state: AgentState) -> NodeUpdate:
        tasks = _tasks(state)
        task = next(t for t in tasks if t.id == state["current_task_id"])
        # The task stays pending with notes on what is missing; its next attempt gets the
        # notes (and the suggested query) in its prompt.
        return {
            "events": _event(
                "retry", f"Retrying {task.id}: {task.notes or 'more evidence needed'}", task.id
            ),
        }

    # -- verify_evidence ----------------------------------------------------------------
    def verify_evidence(self, state: AgentState) -> NodeUpdate:
        sources = _sources(state)
        verified: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for raw in state.get("findings", []):
            finding = Finding.model_validate(raw)
            ids = [sid for sid in finding.source_ids if sid in sources]
            support = lexical_support(finding.statement, [sources[sid].text for sid in ids])
            finding.support = round(support, 3)
            finding.verified = bool(ids) and support >= MIN_LEXICAL_SUPPORT
            (verified if finding.verified else rejected).append(finding.model_dump())

        contradictions = [
            f"Sources {', '.join(c['source_ids'])} disagree: {c['point']}"
            for c in state.get("comparisons", [])
            if c.get("relation") == "disagree"
        ]
        gaps = [
            f"{t.id} ({t.description}): {t.notes or t.status}"
            for t in _tasks(state)
            if t.status in ("insufficient", "failed", "skipped")
        ]
        message = f"Verified {len(verified)} of {len(verified) + len(rejected)} finding(s)."
        if rejected:
            message += f" {len(rejected)} rejected as unsupported by their cited sources."
        if contradictions:
            message += f" {len(contradictions)} contradiction(s) flagged."
        return {
            "verification": {
                "verified": verified,
                "rejected": rejected,
                "contradictions": contradictions,
                "gaps": gaps,
            },
            "events": _event("verify", message),
        }

    # -- generate_report ----------------------------------------------------------------
    def generate_report(self, state: AgentState) -> NodeUpdate:
        verification = state.get("verification") or {}
        verified = [Finding.model_validate(f) for f in verification.get("verified", [])]
        gaps = [*verification.get("contradictions", []), *verification.get("gaps", [])]
        limitations = list(state.get("limitations", []))
        if verification.get("rejected"):
            limitations.append(
                f"{len(verification['rejected'])} draft finding(s) were dropped because their "
                "cited sources did not support them."
            )
        if state.get("stop_reason"):
            limitations.append("Research stopped early at a configured limit.")
        tasks = state.get("tasks", [])

        if not verified:
            content = empty_report(state["objective"], gaps, limitations)
            return {
                "report": {"content": content, "markdown": render_markdown(content, tasks)},
                "events": _event(
                    "report", "No verified findings: wrote an insufficient-evidence report."
                ),
            }

        sources = _sources(state)
        cited = {sid for f in verified for sid in f.source_ids}
        used = [sources[sid] for sid in sorted(cited)]

        def write(objective: str) -> dict[str, Any]:
            draft = self.deps.llm.write_report(objective, verified, gaps, used)
            content, problems = build_report(
                draft,
                objective=objective,
                sources={s.id: s for s in used},
                gaps=gaps,
                limitations=limitations,
            )
            return {"content": content, "problems": problems}

        ctx = ToolContext(
            collection_id=state["collection_id"],
            task_id="report",
            sources=sources,
            max_sources=state["limits"]["max_sources"],
            retrieve=self.deps.retrieve,
            compare=self.deps.llm.compare,
            write_report=write,
        )
        tools = {"create_research_report": build_tools(ctx)["create_research_report"]}
        record, output = self.deps.executor.run(
            tools, "create_research_report", {"objective": state["objective"]}, "report"
        )
        if not output.ok or not isinstance(output.data, dict):
            # Fall back to a report assembled from verified findings only (no model text).
            content = empty_report(state["objective"], gaps, limitations)
            content["title"] = "Research findings"
            content["executive_summary"] = "The report writer failed; verified findings are listed."
            content["key_findings"] = [
                {"text": f.statement, "source_ids": f.source_ids} for f in verified
            ]
            content["citations"] = citations_for(sources, cited)
            content["conclusion"] = "See the verified findings above."
            message = f"Report writer failed ({record.summary}); listed verified findings."
        else:
            content = output.data["content"]
            message = f"Report generated with {len(content['key_findings'])} finding(s)."
            if output.data.get("problems"):
                message += f" {len(output.data['problems'])} uncited item(s) removed."
        return {
            "report": {"content": content, "markdown": render_markdown(content, tasks)},
            "tool_calls": [record.model_dump()],
            "events": _event("report", message),
        }

    # -- finalize_research --------------------------------------------------------------
    def finalize_research(self, state: AgentState) -> NodeUpdate:
        if state.get("error"):
            return {"status": "error", "current_task_id": None}
        report = state.get("report") or {}
        findings = (report.get("content") or {}).get("key_findings", [])
        statuses = {t["status"] for t in state.get("tasks", [])}
        if not findings:
            status = "insufficient_evidence"
        elif state.get("stop_reason"):
            status = "limit_reached"
        elif statuses <= {"completed"}:
            status = "completed"
        else:
            status = "partial"
        elapsed = time.time() - state.get("started_at", time.time())
        logger.info(
            "agent_run_finished",
            extra={
                "run_id": state.get("run_id"),
                "status": status,
                "tool_calls": state.get("tool_call_count", 0),
                "tasks": len(state.get("tasks", [])),
                "elapsed_s": round(elapsed, 1),
            },
        )
        return {
            "status": status,
            "current_task_id": None,
            "events": _event("stop", f"Run finished: {status.replace('_', ' ')}."),
        }

    # -- helpers ------------------------------------------------------------------------
    @staticmethod
    def _fail(code: str, message: str) -> NodeUpdate:
        return {
            "error": {"code": code, "message": message},
            "events": _event("stop", message),
        }
