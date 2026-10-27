"""Conditional edges of the research agent. Pure functions of state.

Termination: the research loop (select -> execute -> evaluate -> [revise] -> select)
consumes one task attempt per pass. Tasks are capped at `max_tasks`, attempts per task at
`max_task_retries + 1`, tool calls at `max_tool_calls` and wall time at `max_seconds`;
select_next_task leaves the loop when no attempt is left or a limit is hit. Every path
then runs verify_evidence -> generate_report -> finalize_research -> END, and the graph's
recursion limit (see workflow.recursion_limit) is a final safety net.
"""

from __future__ import annotations

import time
from typing import Final, Literal

from app.agent.state import AgentState

PLAN: Final = "plan_research"
SELECT: Final = "select_next_task"
EXECUTE: Final = "execute_research_tool"
EVALUATE: Final = "evaluate_research_progress"
REVISE: Final = "revise_research_plan"
VERIFY: Final = "verify_evidence"
REPORT: Final = "generate_report"
FINALIZE: Final = "finalize_research"


def has_error(state: AgentState) -> bool:
    return bool(state.get("error"))


def limit_exceeded(state: AgentState) -> str | None:
    """Name of the first exhausted run budget, or None."""
    limits = state["limits"]
    if state.get("tool_call_count", 0) >= limits["max_tool_calls"]:
        return "tool_calls"
    if time.time() - state.get("started_at", time.time()) >= limits["max_seconds"]:
        return "time"
    return None


def route_after_plan(state: AgentState) -> Literal["select_next_task", "finalize_research"]:
    return FINALIZE if has_error(state) or not state.get("tasks") else SELECT


def route_after_select(
    state: AgentState,
) -> Literal["execute_research_tool", "verify_evidence", "finalize_research"]:
    if has_error(state):
        return FINALIZE
    if state.get("current_task_id") is None or state.get("stop_reason"):
        return VERIFY
    return EXECUTE


def route_after_execute(
    state: AgentState,
) -> Literal["evaluate_research_progress", "finalize_research"]:
    return FINALIZE if has_error(state) else EVALUATE


def route_after_evaluate(
    state: AgentState,
) -> Literal["revise_research_plan", "select_next_task", "finalize_research"]:
    if has_error(state):
        return FINALIZE
    evaluation = state.get("evaluation") or {}
    if evaluation.get("retry") and limit_exceeded(state) is None:
        return REVISE
    return SELECT


def route_after_verify(state: AgentState) -> Literal["generate_report", "finalize_research"]:
    return FINALIZE if has_error(state) else REPORT
