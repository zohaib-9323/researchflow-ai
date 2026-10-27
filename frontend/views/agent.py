"""Deep research: start a planned, tool-using research run and watch its real progress.

Progress comes from GET /agent/runs/{id}, which returns the graph state stored after
every step — the page polls it while a run is active and never invents progress.
"""

from __future__ import annotations

from typing import Any

import streamlit as st
from ui_common import (
    ApiError,
    chip,
    collection_selector,
    esc,
    friendly_time,
    get_client,
    hero,
    html_block,
    plural,
    section,
    setup_page,
    stat_cards,
)

client = get_client()
setup_page()

ACTIVE = {"queued", "running"}
# run status -> (tone, label)
RUN_STATUS = {
    "queued": ("warn", "Queued"),
    "running": ("warn", "Researching"),
    "completed": ("ok", "Completed"),
    "partial": ("warn", "Partial: some tasks lacked evidence"),
    "insufficient_evidence": ("info", "No verifiable evidence found"),
    "limit_reached": ("warn", "Stopped at a limit"),
    "interrupted": ("bad", "Interrupted"),
    "error": ("bad", "Failed"),
}
TASK_STATUS = {
    "pending": ("", "Pending"),
    "in_progress": ("warn", "In progress"),
    "completed": ("ok", "Done"),
    "insufficient": ("info", "Insufficient evidence"),
    "failed": ("bad", "Failed"),
    "skipped": ("", "Skipped (limit)"),
}
EVENT_ICONS = {
    "plan": "🗺️",
    "task": "▶️",
    "tool": "🔧",
    "evaluation": "🧪",
    "retry": "🔁",
    "verify": "✅",
    "report": "📝",
    "stop": "⏹️",
}
EXAMPLE_OBJECTIVES = [
    "Compare the savings reported by the solar pilot report and the council memo, "
    "and explain why they differ.",
    "Summarize the costs, funding and expansion plans for the Larkspur solar programme.",
    "How many rooftop solar systems did the pilot install?",
]

hero(
    "Autonomous research agent",
    'Deep <span class="rf-grad">research</span>',
    "Give the agent a research objective. It plans tasks, chooses tools (document search, "
    "source inspection, comparison, calculation, and web search if configured), checks its "
    "evidence, and writes a cited report.",
)


def status_chip(status: str) -> str:
    tone, label = RUN_STATUS.get(status, ("", status))
    return chip(label, tone, dot=True)


def start_form() -> None:
    try:
        health = client.get("/health")
    except ApiError as exc:
        st.error(exc.message)
        return
    if not health.get("llm_configured"):
        st.warning("No Gemini API key is configured, so the agent cannot run.")
        return
    if not health.get("web_search_configured"):
        st.caption("Web search is not configured: the agent will use your documents only.")
    collection = collection_selector("agent_collection")
    objective = st.text_area(
        "Research objective",
        key="agent_objective",
        placeholder="What should the agent investigate?",
        max_chars=4000,
        height=110,
    )
    with st.expander("Examples"):
        for i, example in enumerate(EXAMPLE_OBJECTIVES):
            if st.button(example, key=f"agent_example_{i}", type="tertiary"):
                st.session_state["agent_objective"] = example
                st.rerun()
    if st.button("Start research", type="primary", icon=":material/travel_explore:"):
        if len(objective.strip()) < 3:
            st.warning("Describe what the agent should research.")
            return
        try:
            run = client.post(
                "/agent/runs",
                {"objective": objective, "collection_id": collection["id"] if collection else None},
            )
        except ApiError as exc:
            st.error(exc.message)
            return
        st.session_state["agent_run_id"] = run["id"]
        st.rerun()


def render_tasks(run: dict[str, Any]) -> None:
    section("Plan", "Simple question: planning skipped" if run.get("mode") == "simple" else "")
    if not run["tasks"]:
        st.caption("Planning…" if run["status"] in ACTIVE else "No tasks were planned.")
        return
    for task in run["tasks"]:
        tone, label = TASK_STATUS.get(task["status"], ("", task["status"]))
        current = " ← current" if task["id"] == run.get("current_task_id") else ""
        attempts = f" · attempt {task['attempts']}" if task.get("attempts", 0) > 1 else ""
        html_block(
            '<div class="rf-row" style="margin:4px 0">'
            + chip(task["id"], "brand")
            + chip(label, tone, dot=True)
            + chip(task["approach"].replace("_", " "))
            + f"<span>{esc(task['description'])}{esc(current)}{esc(attempts)}</span></div>"
        )
        if task.get("summary") and task["status"] != "in_progress":
            st.caption(task["summary"])


def render_activity(run: dict[str, Any]) -> None:
    section("Activity", plural(len(run["events"]), "step"))
    for event in run["events"][-40:]:
        icon = EVENT_ICONS.get(event["kind"], "•")
        st.markdown(f"{icon} {esc(event['message'])}", unsafe_allow_html=True)


def render_tools(run: dict[str, Any]) -> None:
    calls = run["tool_calls"]
    with st.expander(f"Tool calls ({len(calls)})"):
        for call in calls:
            args = ", ".join(f"{k}={v}" for k, v in call["arguments"].items())
            mark = "✅" if call["ok"] else "⚠️"
            st.markdown(
                f"{mark} `{esc(call['tool'])}({esc(args)})` · {esc(call['summary'])} "
                f"· {call['result_count']} result(s) · {call['duration_ms']} ms",
                unsafe_allow_html=True,
            )


def render_sources(run: dict[str, Any]) -> None:
    sources = run["sources"]
    if not sources:
        return
    with st.expander(f"Sources gathered ({len(sources)})"):
        for source in sources:
            where = source.get("url") or (
                f"page {source['page']}" if source.get("page") is not None else ""
            )
            st.markdown(f"**[{source['id']}] {esc(source['title'])}** {esc(where)}")
            st.caption(f"{source['excerpt']} — via {source['provenance']}")


def render_report(run: dict[str, Any]) -> None:
    try:
        report = client.get(f"/agent/runs/{run['id']}/report")
    except ApiError as exc:
        st.error(exc.message)
        return
    section("Report")
    read, download = st.columns([3, 1], vertical_alignment="top")
    download.download_button(
        "Download Markdown",
        report["markdown"],
        file_name="research-report.md",
        mime="text/markdown",
        icon=":material/download:",
        key=f"agent_dl_{run['id']}",
    )
    with read.container(border=True):
        st.markdown(report["markdown"])


def render_run(run: dict[str, Any]) -> None:
    html_block(
        '<div class="rf-row">'
        + status_chip(run["status"])
        + f'<span class="rf-muted">{esc(friendly_time(run["updated_at"]))}</span></div>'
    )
    st.markdown(f"**Objective:** {esc(run['objective'])}", unsafe_allow_html=True)
    counters = run.get("counters") or {}
    limits = run.get("limits") or {}
    stat_cards(
        [
            ("Tasks done", f"{counters.get('tasks_done', 0)}/{counters.get('tasks_total', 0)}"),
            (
                "Tool calls",
                f"{counters.get('tool_calls', 0)}/{limits.get('max_tool_calls', '-')}",
            ),
            ("Retries", counters.get("retries", 0)),
            ("Sources", counters.get("sources", 0)),
        ]
    )
    verification = run.get("verification") or {}
    if verification:
        st.caption(
            f"Evidence check: {verification.get('verified', 0)} finding(s) verified, "
            f"{verification.get('rejected', 0)} rejected."
        )
    if run.get("error"):
        st.error(run["error"].get("message", "The run failed."))
    if run.get("resumable") and st.button("Resume run", icon=":material/play_arrow:"):
        try:
            client.post(f"/agent/runs/{run['id']}/resume", {})
        except ApiError as exc:
            st.error(exc.message)
        st.rerun()
    left, right = st.columns([1, 1])
    with left:
        render_tasks(run)
    with right:
        render_activity(run)
    render_tools(run)
    render_sources(run)
    if run.get("has_report"):
        render_report(run)


@st.fragment(run_every=2)
def live_run(run_id: str) -> None:
    """Polls the run while it is active; stops refreshing once it has finished."""
    try:
        run = client.get(f"/agent/runs/{run_id}")
    except ApiError as exc:
        st.error(exc.message)
        return
    render_run(run)
    if run["status"] not in ACTIVE:
        st.session_state["agent_finished"] = run_id
        st.rerun()  # a full rerun renders the final state without the polling fragment


def static_run(run_id: str) -> None:
    try:
        render_run(client.get(f"/agent/runs/{run_id}"))
    except ApiError as exc:
        st.error(exc.message)


def past_runs() -> None:
    try:
        runs = client.get("/agent/runs")
    except ApiError:
        return
    if not runs:
        return
    section("Past runs")
    for run in runs[:20]:
        text_col, button_col = st.columns([5, 1], vertical_alignment="center")
        text_col.markdown(
            f"{esc(run['objective'][:140])}<br>"
            + status_chip(run["status"])
            + f' <span class="rf-muted">{esc(friendly_time(run["created_at"]))}</span>',
            unsafe_allow_html=True,
        )
        if button_col.button("Open", key=f"open_run_{run['id']}"):
            st.session_state["agent_run_id"] = run["id"]
            st.rerun()


run_id = st.session_state.get("agent_run_id")
if run_id:
    if st.button("New research", icon=":material/add:", type="tertiary"):
        st.session_state.pop("agent_run_id", None)
        st.rerun()
    try:
        status = client.get(f"/agent/runs/{run_id}")["status"]
    except ApiError as exc:
        st.error(exc.message)
        st.session_state.pop("agent_run_id", None)
        st.stop()
    if status in ACTIVE:
        live_run(run_id)
    else:
        static_run(run_id)
else:
    start_form()
    past_runs()
