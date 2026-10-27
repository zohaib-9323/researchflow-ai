"""Research-run lifecycle: start, progress snapshots, resume and reports.

A run executes the agent graph on its own checkpoint thread, in a background thread
(or inline, for tests). After every graph step the service stores a progress snapshot
in the database, which the API/UI polls — progress shown to users is real graph state.

If the process stops mid-run, or a provider failure aborts a step, the LangGraph
checkpoint keeps the last completed step; `resume()` continues from there.
"""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime
from typing import Any

from langgraph.graph.state import CompiledStateGraph
from sqlalchemy import select

from app.agent.state import AgentLimits, AgentState, initial_agent_state
from app.agent.workflow import agent_config
from app.config import Settings
from app.core.exceptions import (
    LLMNotConfiguredError,
    NotFoundError,
    ResearchFlowError,
    ValidationFailedError,
)
from app.core.logging import get_logger
from app.persistence.database import Database
from app.persistence.models import AgentReport, ResearchRun
from app.persistence.repositories import CollectionRepository
from app.schemas.agent import AgentReportOut, RunDetail, RunSummary

logger = get_logger(__name__)

ACTIVE = {"queued", "running"}
RESUMABLE = {"interrupted", "error"}
MAX_CONCURRENT_RUNS = 2
SNAPSHOT_TEXT_CHARS = 300  # source excerpt kept in snapshots (never full documents)


def _now() -> datetime:
    return datetime.now(UTC)


class AgentService:
    def __init__(
        self,
        settings: Settings,
        database: Database,
        graph: CompiledStateGraph,
        *,
        llm_configured: bool,
        web_available: bool,
        background: bool = True,
    ) -> None:
        self.settings = settings
        self.database = database
        self.graph = graph
        self.llm_configured = llm_configured
        self.web_available = web_available
        self.background = background
        self._active: set[str] = set()
        self._lock = threading.Lock()
        self._threads: dict[str, threading.Thread] = {}
        self._mark_orphaned_runs()

    # -- public API ---------------------------------------------------------------------
    def limits(self) -> AgentLimits:
        s = self.settings
        return AgentLimits(
            max_tasks=s.agent_max_tasks,
            max_tool_calls=s.agent_max_tool_calls,
            max_task_retries=s.agent_max_task_retries,
            max_seconds=s.agent_max_seconds,
            max_sources=s.agent_max_sources,
        )

    def start(self, objective: str, collection_id: str | None = None) -> RunDetail:
        if not self.llm_configured:
            raise LLMNotConfiguredError()
        objective = " ".join(objective.split())
        if len(objective) < 3:
            raise ValidationFailedError("Describe the research objective.")
        with self.database.session() as session:
            repo = CollectionRepository(session)
            collection = repo.get(collection_id) if collection_id else repo.get_or_create_default()
            if collection is None:
                raise NotFoundError("Collection not found.")
            run = ResearchRun(collection_id=collection.id, objective=objective, status="queued")
            session.add(run)
            session.flush()
            run_id = run.id
        self._claim(run_id)
        state = initial_agent_state(
            run_id=run_id,
            objective=objective,
            collection_id=collection.id,
            web_available=self.web_available,
            limits=self.limits(),
        )
        self._launch(run_id, state)
        return self.get(run_id)

    def resume(self, run_id: str) -> RunDetail:
        run = self._load(run_id)
        if run.status not in RESUMABLE or not self._resumable(run_id):
            raise ValidationFailedError("This run cannot be resumed.")
        if not self.llm_configured:
            raise LLMNotConfiguredError()
        self._claim(run_id)
        config = agent_config(run_id, self.limits())
        if self.graph.get_state(config).values:
            # The time budget applies per session of work, so restart its clock.
            self.graph.update_state(config, {"started_at": time.time(), "error": None})
            self._launch(run_id, None)
        else:  # stopped before the first checkpoint: start over with the same input
            state = initial_agent_state(
                run_id=run_id,
                objective=run.objective,
                collection_id=run.collection_id,
                web_available=self.web_available,
                limits=self.limits(),
            )
            self._launch(run_id, state)
        return self.get(run_id)

    def list(self, limit: int = 50) -> list[RunSummary]:
        with self.database.session() as session:
            runs = session.scalars(
                select(ResearchRun).order_by(ResearchRun.created_at.desc()).limit(limit)
            ).all()
            reports = set(
                session.scalars(
                    select(AgentReport.run_id).where(AgentReport.run_id.in_([r.id for r in runs]))
                ).all()
            )
            return [self._summary(run, run.id in reports) for run in runs]

    def get(self, run_id: str) -> RunDetail:
        run = self._load(run_id)
        with self.database.session() as session:
            has_report = (
                session.scalar(select(AgentReport.id).where(AgentReport.run_id == run_id))
                is not None
            )
        snap = run.snapshot or {}
        summary = self._summary(run, has_report)
        return RunDetail(
            **summary.model_dump(),
            mode=snap.get("mode"),
            current_task_id=snap.get("current_task_id"),
            stop_reason=snap.get("stop_reason"),
            tasks=snap.get("tasks", []),
            events=snap.get("events", []),
            tool_calls=snap.get("tool_calls", []),
            sources=snap.get("sources", []),
            counters=snap.get("counters", {}),
            limits=snap.get("limits", {}),
            verification=snap.get("verification", {}),
            limitations=snap.get("limitations", []),
            web_available=snap.get("web_available", self.web_available),
            error=run.error,
        )

    def report(self, run_id: str) -> AgentReportOut:
        self._load(run_id)
        with self.database.session() as session:
            report = session.scalar(select(AgentReport).where(AgentReport.run_id == run_id))
            if report is None:
                raise NotFoundError("This run has no report yet.")
            return AgentReportOut(
                id=report.id,
                run_id=report.run_id,
                title=report.title,
                content=report.content,
                markdown=report.markdown,
                created_at=report.created_at,
            )

    def wait(self, run_id: str, timeout: float = 30.0) -> None:
        """Block until a background run finishes (tests and scripts)."""
        thread = self._threads.get(run_id)
        if thread is not None:
            thread.join(timeout)

    # -- execution ----------------------------------------------------------------------
    def _claim(self, run_id: str) -> None:
        with self._lock:
            if run_id in self._active:
                raise ValidationFailedError("This run is already in progress.")
            if len(self._active) >= MAX_CONCURRENT_RUNS:
                raise ValidationFailedError(
                    "Too many research runs are in progress. Try again when one finishes."
                )
            self._active.add(run_id)

    def _launch(self, run_id: str, state: AgentState | None) -> None:
        self._update(run_id, status="running", error=None)
        if not self.background:
            self._execute(run_id, state)
            return
        thread = threading.Thread(
            target=self._execute, args=(run_id, state), name=f"agent-{run_id[:8]}", daemon=True
        )
        self._threads[run_id] = thread
        thread.start()

    def _execute(self, run_id: str, state: AgentState | None) -> None:
        config = agent_config(run_id, self.limits())
        final: dict[str, Any] = {}
        try:
            for values in self.graph.stream(state, config, stream_mode="values"):
                final = values
                self._update(run_id, snapshot=self._snapshot(values))
            status = final.get("status", "error")
            error = final.get("error")
            self._store_report(run_id, final)
            self._update(run_id, status=status, error=error)
        except ResearchFlowError as exc:
            logger.warning("agent_run_failed", extra={"run_id": run_id, "error_code": exc.code})
            self._update(run_id, status="error", error={"code": exc.code, "message": exc.message})
        except Exception as exc:  # never leave a run "running" after a crash
            logger.exception("agent_run_crashed", extra={"error_type": type(exc).__name__})
            self._update(
                run_id,
                status="error",
                error={"code": "internal_error", "message": "The research run failed."},
            )
        finally:
            with self._lock:
                self._active.discard(run_id)

    def _store_report(self, run_id: str, state: dict[str, Any]) -> None:
        report = state.get("report")
        if not report:
            return
        content = report["content"]
        with self.database.session() as session:
            existing = session.scalar(select(AgentReport).where(AgentReport.run_id == run_id))
            if existing is None:
                existing = AgentReport(run_id=run_id, title="", content={}, markdown="")
                session.add(existing)
            existing.title = content["title"][:300]
            existing.content = content
            existing.markdown = report["markdown"]

    # -- persistence helpers ------------------------------------------------------------
    def _update(self, run_id: str, **fields: Any) -> None:
        with self.database.session() as session:
            run = session.get(ResearchRun, run_id)
            if run is None:
                return
            for key, value in fields.items():
                setattr(run, key, value)
            run.updated_at = _now()

    def _load(self, run_id: str) -> ResearchRun:
        with self.database.session() as session:
            run = session.get(ResearchRun, run_id)
            if run is None:
                raise NotFoundError("Research run not found.")
            if run.status in ACTIVE and run_id not in self._active:
                # The process that ran it is gone (restart / crash).
                run.status = "interrupted"
                run.updated_at = _now()
            return run

    def _mark_orphaned_runs(self) -> None:
        with self.database.session() as session:
            for run in session.scalars(
                select(ResearchRun).where(ResearchRun.status.in_(ACTIVE))
            ).all():
                run.status = "interrupted"
                run.updated_at = _now()

    def _resumable(self, run_id: str) -> bool:
        state = self.graph.get_state(agent_config(run_id, self.limits()))
        return not state.values or bool(state.next)

    def _summary(self, run: ResearchRun, has_report: bool) -> RunSummary:
        status = run.status
        if status in ACTIVE and run.id not in self._active:
            status = "interrupted"
        return RunSummary(
            id=run.id,
            collection_id=run.collection_id,
            objective=run.objective,
            status=status,
            resumable=status in RESUMABLE and self._resumable(run.id),
            has_report=has_report,
            created_at=run.created_at,
            updated_at=run.updated_at,
        )

    @staticmethod
    def _snapshot(state: dict[str, Any]) -> dict[str, Any]:
        """User-facing progress: no full document text, no prompts, no secrets."""
        sources = [
            {
                **{k: v for k, v in source.items() if k != "text"},
                "excerpt": str(source.get("text", ""))[:SNAPSHOT_TEXT_CHARS],
            }
            for source in state.get("sources", {}).values()
        ]
        verification = state.get("verification") or {}
        tasks = state.get("tasks", [])
        limits = state.get("limits", {})
        return {
            "mode": state.get("mode"),
            "current_task_id": state.get("current_task_id"),
            "stop_reason": state.get("stop_reason"),
            "tasks": tasks,
            "events": state.get("events", []),
            "tool_calls": state.get("tool_calls", []),
            "sources": sources,
            "limits": dict(limits),
            "web_available": state.get("web_available", False),
            "limitations": state.get("limitations", []),
            "counters": {
                "tool_calls": state.get("tool_call_count", 0),
                "max_tool_calls": limits.get("max_tool_calls"),
                "tasks_total": len(tasks),
                "tasks_done": sum(
                    1 for t in tasks if t["status"] not in ("pending", "in_progress")
                ),
                "retries": sum(max(0, t.get("attempts", 0) - 1) for t in tasks),
                "sources": len(sources),
                "findings": len(state.get("findings", [])),
                "elapsed_seconds": round(time.time() - state.get("started_at", time.time()), 1),
            },
            "verification": {
                "verified": len(verification.get("verified", [])),
                "rejected": len(verification.get("rejected", [])),
                "contradictions": verification.get("contradictions", []),
                "gaps": verification.get("gaps", []),
            }
            if verification
            else {},
            "report_ready": bool(state.get("report")),
        }
