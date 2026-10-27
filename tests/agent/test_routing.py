from __future__ import annotations

import time
from typing import Any

from app.agent import routing
from app.agent.nodes import is_simple_objective
from app.agent.state import AgentLimits
from app.agent.workflow import recursion_limit

LIMITS = AgentLimits(
    max_tasks=3, max_tool_calls=5, max_task_retries=1, max_seconds=60, max_sources=10
)


def state(**values: Any) -> Any:
    return {"limits": LIMITS, "started_at": time.time(), "tool_call_count": 0, **values}


def test_plan_routes_to_select_or_finalize() -> None:
    assert routing.route_after_plan(state(tasks=[{"id": "T1"}])) == routing.SELECT
    assert routing.route_after_plan(state(tasks=[])) == routing.FINALIZE
    assert routing.route_after_plan(state(tasks=[{}], error={"code": "x"})) == routing.FINALIZE


def test_select_routes() -> None:
    assert routing.route_after_select(state(current_task_id="T1")) == routing.EXECUTE
    assert routing.route_after_select(state(current_task_id=None)) == routing.VERIFY
    assert (
        routing.route_after_select(state(current_task_id="T1", stop_reason="time"))
        == routing.VERIFY
    )


def test_evaluate_routes_to_revise_only_with_retry_and_budget() -> None:
    retry = {"retry": True}
    assert routing.route_after_evaluate(state(evaluation=retry)) == routing.REVISE
    assert routing.route_after_evaluate(state(evaluation={"retry": False})) == routing.SELECT
    assert (
        routing.route_after_evaluate(state(evaluation=retry, tool_call_count=5)) == routing.SELECT
    )
    assert (
        routing.route_after_evaluate(state(evaluation=retry, started_at=time.time() - 120))
        == routing.SELECT
    )
    assert routing.route_after_evaluate(state(error={"code": "x"})) == routing.FINALIZE


def test_limit_exceeded() -> None:
    assert routing.limit_exceeded(state()) is None
    assert routing.limit_exceeded(state(tool_call_count=5)) == "tool_calls"
    assert routing.limit_exceeded(state(started_at=time.time() - 61)) == "time"


def test_recursion_limit_covers_every_attempt() -> None:
    # 3 tasks x 2 attempts x 4 nodes + plan/final select/verify/report/finalize
    assert recursion_limit(LIMITS) >= 3 * 2 * 4 + 5


def test_simple_objective_heuristic() -> None:
    assert is_simple_objective("How many homes did the solar pilot include?")
    assert not is_simple_objective("Compare the pilot report and the council memo savings")
    assert not is_simple_objective(" ".join(["word"] * 25))
