"""Typed state of one research-agent run (one LangGraph thread per run).

Records are stored as plain dicts (model_dump of app/agent/schemas.py) so checkpoints
stay JSON-friendly. The activity log, tool calls and findings are append-only (reducers);
everything else is replaced by the node that owns it.
"""

from __future__ import annotations

import operator
import time
from typing import Annotated, Any, TypedDict


class AgentLimits(TypedDict):
    max_tasks: int
    max_tool_calls: int
    max_task_retries: int
    max_seconds: float
    max_sources: int


class AgentState(TypedDict, total=False):
    # request
    run_id: str
    objective: str
    collection_id: str
    web_available: bool
    limits: AgentLimits
    started_at: float  # wall-clock seconds; reset when a run is resumed

    # plan and progress
    mode: str  # "simple" (planner skipped) | "planned"
    tasks: list[dict[str, Any]]  # ResearchTask dumps
    current_task_id: str | None
    tool_call_count: int
    last_tool_results: str  # compact text of the current task's tool outputs
    evaluation: dict[str, Any] | None  # outcome of the latest task assessment
    stop_reason: str | None  # set when a limit stopped the research loop

    # evidence
    sources: dict[str, dict[str, Any]]  # SourceRecord dumps by id
    comparisons: Annotated[list[dict[str, Any]], operator.add]
    findings: Annotated[list[dict[str, Any]], operator.add]  # Finding dumps (unverified)
    verification: dict[str, Any] | None

    # activity log (user-visible; no hidden reasoning)
    events: Annotated[list[dict[str, Any]], operator.add]
    tool_calls: Annotated[list[dict[str, Any]], operator.add]

    # output
    report: dict[str, Any] | None
    limitations: list[str]
    status: str
    error: dict[str, Any] | None


def initial_agent_state(
    *,
    run_id: str,
    objective: str,
    collection_id: str,
    web_available: bool,
    limits: AgentLimits,
) -> AgentState:
    return AgentState(
        run_id=run_id,
        objective=objective,
        collection_id=collection_id,
        web_available=web_available,
        limits=limits,
        started_at=time.time(),
        mode="planned",
        tasks=[],
        current_task_id=None,
        tool_call_count=0,
        last_tool_results="",
        evaluation=None,
        stop_reason=None,
        sources={},
        comparisons=[],
        findings=[],
        verification=None,
        events=[],
        tool_calls=[],
        report=None,
        limitations=[],
        status="running",
        error=None,
    )
