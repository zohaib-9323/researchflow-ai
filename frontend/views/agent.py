"""Deep research: start a planned, tool-using research run and watch its real progress.

Progress comes from GET /agent/runs/{id}, which returns the graph state stored after
every step — the page polls it while a run is active and never invents progress. The
run's activity log is shown as a plain-language research journal.
"""

from __future__ import annotations

import re
from typing import Any

import streamlit as st
from ui_common import (
    ApiError,
    chip,
    collection_selector,
    credits_left,
    esc,
    friendly_time,
    get_client,
    hero,
    html_block,
    icon,
    orb,
    out_of_credits_notice,
    plural,
    section,
    setup_page,
)

client = get_client()
setup_page()

ACTIVE = {"queued", "running"}
# run status -> (tone, label, one-line explanation)
RUN_STATUS = {
    "queued": ("warn", "Getting ready", "Your research is about to start."),
    "running": ("brand", "Researching", "Working through the plan. This page updates live."),
    "completed": ("ok", "Report ready", "Every step was answered with checked evidence."),
    "partial": ("warn", "Report ready, with gaps", "Some steps couldn't be fully answered."),
    "insufficient_evidence": (
        "info",
        "Not enough evidence",
        "Your documents don't contain enough to answer this.",
    ),
    "limit_reached": ("warn", "Stopped early", "It reached its time or step limit."),
    "interrupted": ("bad", "Interrupted", "The run stopped unexpectedly. You can resume it."),
    "error": ("bad", "Something went wrong", "The AI service failed. You can resume the run."),
}
TASK_STATE = {
    "pending": "pending",
    "in_progress": "active",
    "completed": "completed",
    "insufficient": "insufficient",
    "failed": "failed",
    "skipped": "skipped",
}
TASK_NOTE = {
    "insufficient": "Not enough evidence found",
    "failed": "Failed",
    "skipped": "Skipped (limit reached)",
}
# tool -> (icon, what it means in plain words)
TOOL_WORDS = {
    "search_documents": ("search", "Searched your documents"),
    "search_web": ("globe", "Searched the web"),
    "inspect_source": ("eye", "Read a source closely"),
    "compare_sources": ("compare", "Compared sources"),
    "calculate": ("calc", "Did the math"),
    "create_research_report": ("pen", "Wrote the report"),
}
KIND_ICON = {
    "plan": "map",
    "task": "play",
    "evaluation": "check",
    "retry": "retry",
    "verify": "shield",
    "report": "pen",
    "stop": "flag",
}
EXAMPLE_OBJECTIVES = [
    "Compare the savings reported by the solar pilot report and the council memo, "
    "and explain why they differ.",
    "Summarize the costs, funding and expansion plans for the Larkspur solar programme.",
    "How many rooftop solar systems did the pilot install?",
]
_TOOL_EVENT = re.compile(r"^(\w+) → (ok|failed): (.*)$", re.S)

hero(
    "Your research agent",
    'Deep <span class="rf-grad">research</span>',
    "Give me a question that needs real digging. I'll plan the steps, read and compare your "
    "sources, check every finding, and write you a cited report.",
)


def humanize(text: str) -> str:
    """Task ids (T1) become 'step 1'; tidy the backend's short notes."""
    text = re.sub(r"\bT(\d+)\b", r"step \1", text)
    return text.replace("(s)", "s")


def status_chip(status: str) -> str:
    tone, label, _ = RUN_STATUS.get(status, ("", status, ""))
    return chip(label, "brand live" if status in ACTIVE else tone, dot=True)


# -- start ----------------------------------------------------------------------------------
def use_example(example: str) -> None:
    # Runs before the text area is created, so it may set the widget's value.
    st.session_state["agent_objective"] = example


def start_form() -> None:
    try:
        health = client.get("/health")
    except ApiError as exc:
        st.error(exc.message)
        return
    if not health.get("llm_configured"):
        st.warning("No Gemini API key is configured, so the agent can't run yet.")
        return
    with st.container(key="agentform"):
        objective = st.text_area(
            "What would you like me to research?",
            key="agent_objective",
            placeholder="e.g. Compare what the reports say about costs and explain any "
            "differences…",
            max_chars=4000,
            height=120,
        )
        left, right = st.columns([3, 2], vertical_alignment="bottom")
        with left:
            collection = collection_selector("agent_collection")
        remaining = credits_left("agent_run")
        with right:
            label = "Start researching" + (
                f" · {remaining} free left" if remaining not in (None, 0) else ""
            )
            clicked = st.button(
                label,
                type="primary",
                icon=":material/arrow_forward:",
                use_container_width=True,
                disabled=remaining == 0,
            )
        if not health.get("web_search_configured"):
            st.caption("I'll use the documents in your library (web search is off).")
    if remaining == 0:
        out_of_credits_notice("agent_run")
    section("Need an idea?")
    with st.container(key="agentexamples"):
        columns = st.columns(3)
        for i, example in enumerate(EXAMPLE_OBJECTIVES):
            columns[i].button(
                example,
                key=f"agent_example_{i}",
                icon=":material/lightbulb:",
                on_click=use_example,
                args=(example,),
            )
    if clicked:
        if len(objective.strip()) < 3:
            st.warning("Tell me what to research first.")
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


# -- run view -------------------------------------------------------------------------------
def render_header(run: dict[str, Any]) -> None:
    explanation = RUN_STATUS.get(run["status"], ("", "", ""))[2]
    active = run["status"] in ACTIVE
    counters = run.get("counters") or {}
    total = counters.get("tasks_total", 0)
    done = counters.get("tasks_done", 0)
    # while running, leave room at the end for verification and the report
    pct = 100 if not active else (max(6, round(done / total * 88) + 6) if total else 4)
    verification = run.get("verification") or {}
    meta = [
        f"<span><b>{done}</b> of <b>{total}</b> steps</span>" if total else "",
        f"<span><b>{counters.get('sources', 0)}</b> sources read</span>",
        f"<span><b>{counters.get('tool_calls', 0)}</b> actions</span>",
    ]
    if counters.get("retries"):
        meta.append(f"<span><b>{counters['retries']}</b> retried</span>")
    if verification:
        meta.append(f"<span><b>{verification.get('verified', 0)}</b> findings verified</span>")
    with st.container(key="runhead"):
        html_block(
            '<div class="rf-row">'
            + status_chip(run["status"])
            + f'<span class="rf-muted">{esc(explanation)} · {esc(friendly_time(run["updated_at"]))}'
            "</span></div>"
            f'<div class="rf-objective">{esc(run["objective"])}</div>'
            f'<div class="rf-progress{" live" if active else ""}">'
            f'<b style="width:{pct}%"></b></div>'
            f'<div class="rf-progress-meta">{"".join(meta)}</div>'
        )
        current = next((t for t in run["tasks"] if t["id"] == run.get("current_task_id")), None)
        if active:
            doing = humanize(current["description"]) if current else "Making a research plan…"
            html_block(
                f'<div class="rf-now">{orb(34, live=True)}<div><small>Working on</small>'
                f"<span>{esc(doing)}</span></div></div>"
            )
        if run.get("error"):
            st.error(run["error"].get("message", "The run failed."))
        if run.get("resumable") and st.button(
            "Resume where it stopped", icon=":material/play_arrow:", type="primary"
        ):
            try:
                client.post(f"/agent/runs/{run['id']}/resume", {})
            except ApiError as exc:
                st.error(exc.message)
            st.rerun()


def render_plan(run: dict[str, Any]) -> None:
    note = "A single quick search was enough" if run.get("mode") == "simple" else ""
    section("The plan", note)
    if not run["tasks"]:
        html_block(
            '<div class="rf-now" style="margin-top:0">'
            + orb(28, live=run["status"] in ACTIVE)
            + "<div><span>Thinking about how to approach this…</span></div></div>"
        )
        return
    items = []
    for number, task in enumerate(run["tasks"], start=1):
        state = TASK_STATE.get(task["status"], "pending")
        mark = icon("check") if state == "completed" else ""
        detail = ""
        if task["status"] in TASK_NOTE:
            detail = TASK_NOTE[task["status"]]
        if task.get("summary") and task["status"] in ("completed", "insufficient"):
            detail = task["summary"] if state == "completed" else f"{detail}. {task['summary']}"
        retry = " · second attempt" if task.get("attempts", 0) > 1 else ""
        items.append(
            f'<li class="{state}"><span class="rf-plan-mark">{mark}</span><div>'
            f'<div class="rf-plan-n">Step {number}{esc(retry)}</div>'
            f'<div class="rf-plan-d">{esc(task["description"])}</div>'
            + (f'<div class="rf-plan-s">{esc(humanize(detail))}</div>' if detail else "")
            + "</div></li>"
        )
    html_block(f'<ul class="rf-plan">{"".join(items)}</ul>')


def journal_entry(event: dict[str, Any]) -> str:
    kind, message = event["kind"], event["message"]
    css, ico, title, detail = f"k-{kind}", KIND_ICON.get(kind, "spark"), message, ""
    if kind == "tool" and (match := _TOOL_EVENT.match(message)):
        tool, outcome, summary = match.groups()
        ico, title = TOOL_WORDS.get(tool, ("spark", tool.replace("_", " ").capitalize()))
        detail = summary
        if outcome == "failed":
            css += " failed"
            title += " (didn't work)"
    elif kind == "plan":
        # the steps themselves are listed in the plan, so the journal only notes the plan
        title = message.partition(" — ")[0]
        title = title.replace("Planned", "Made a plan with").replace("task(s)", "steps")
        if message.startswith("Simple question"):
            title = "A single search should answer this"
    elif kind == "task":
        title = humanize(message.split(": ", 1)[0])
        detail = message.split(": ", 1)[1] if ": " in message else ""
        if message == "All tasks processed.":
            title, detail = "Finished all steps", ""
    elif kind == "evaluation":
        head, _, rest = message.partition(" — ")
        title = "Checked the evidence for " + humanize(head).replace(":", " →")
        detail = humanize(rest)
    else:
        title = humanize(message)
    title = title[:1].upper() + title[1:]
    return (
        f'<li class="{css}"><span class="rf-journal-ico">{icon(ico)}</span><div>'
        f'<div class="rf-journal-t">{esc(title)}</div>'
        + (f'<div class="rf-journal-d">{esc(humanize(detail))}</div>' if detail else "")
        + "</div></li>"
    )


def render_journal(run: dict[str, Any]) -> None:
    section("Research journal", plural(len(run["events"]), "entry", "entries"))
    events = run["events"][-30:]
    if not events:
        st.caption("The journal will fill in as I work.")
        return
    html_block(f'<ul class="rf-journal">{"".join(journal_entry(e) for e in events)}</ul>')


def render_sources(run: dict[str, Any]) -> None:
    sources = run["sources"]
    if not sources:
        return
    with st.expander(f"Sources I read ({len(sources)})", icon=":material/menu_book:"):
        cards = []
        for source in sources:
            where = (
                f'<a href="{esc(source["url"])}" target="_blank" rel="noopener">link</a>'
                if source.get("url")
                else (f"p. {source['page']}" if source.get("page") is not None else "")
            )
            cards.append(
                f'<div class="rf-src"><div class="rf-src-head">'
                f'<span class="rf-src-id">{esc(source["id"])}</span>'
                f'<span class="rf-src-name">{esc(source["title"])}</span>'
                f'<span class="rf-src-page">{where}</span></div>'
                f'<div class="rf-src-quote">“{esc(source["excerpt"])}”</div></div>'
            )
        html_block(f'<div class="rf-sources">{"".join(cards)}</div>')


def render_actions(run: dict[str, Any]) -> None:
    calls = run["tool_calls"]
    if not calls:
        return
    with st.expander(f"Behind the scenes ({len(calls)} actions)", icon=":material/build:"):
        for call in calls:
            words = TOOL_WORDS.get(call["tool"], ("", call["tool"]))[1]
            args = ", ".join(f"{k}: {v}" for k, v in call["arguments"].items())
            mark = "✓" if call["ok"] else "✕"
            st.markdown(
                f"**{mark} {esc(words)}** · {esc(args)}  \n"
                f"<span class='rf-muted'>{esc(call['summary'])} · "
                f"{call['duration_ms'] / 1000:.1f} s</span>",
                unsafe_allow_html=True,
            )


def render_report(run: dict[str, Any]) -> None:
    try:
        report = client.get(f"/agent/runs/{run['id']}/report")
    except ApiError as exc:
        st.error(exc.message)
        return
    content = report["content"]
    section("Your report", plural(len(content.get("citations", [])), "source") + " cited")
    with st.container(key="agentreport"):
        top, download = st.columns([4, 1], vertical_alignment="center")
        top.markdown(
            '<div class="rf-row">'
            + chip(plural(len(content.get("key_findings", [])), "key finding"), "brand")
            + chip(plural(len(content.get("citations", [])), "citation"))
            + "</div>",
            unsafe_allow_html=True,
        )
        download.download_button(
            "Download",
            report["markdown"],
            file_name="research-report.md",
            mime="text/markdown",
            icon=":material/download:",
            key=f"agent_dl_{run['id']}",
            use_container_width=True,
        )
        with st.container(key="reportdoc"):
            st.markdown(report["markdown"])


def render_run(run: dict[str, Any]) -> None:
    render_header(run)
    left, right = st.columns([1, 1], gap="large")
    with left:
        render_plan(run)
    with right:
        render_journal(run)
    if run.get("has_report"):
        render_report(run)
    render_sources(run)
    render_actions(run)


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
        st.rerun()  # a full rerun renders the final state without the polling fragment


def past_runs() -> None:
    try:
        runs = client.get("/agent/runs")
    except ApiError:
        return
    if not runs:
        return
    section("Your past research", plural(len(runs), "run"))
    for run in runs[:20]:
        with st.container(key=f"runrow_{run['id']}"):
            text_col, button_col = st.columns([5, 1], vertical_alignment="center")
            text_col.markdown(
                f'<div class="rf-run-title">{esc(run["objective"][:160])}</div>'
                f'<div class="rf-row" style="margin-top:8px">{status_chip(run["status"])}'
                f'<span class="rf-muted">{esc(friendly_time(run["created_at"]))}</span></div>',
                unsafe_allow_html=True,
            )
            if button_col.button("Open", key=f"open_run_{run['id']}", use_container_width=True):
                st.session_state["agent_run_id"] = run["id"]
                st.rerun()


run_id = st.session_state.get("agent_run_id")
if run_id:
    if st.button("New research", icon=":material/arrow_back:", type="tertiary"):
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
        try:
            render_run(client.get(f"/agent/runs/{run_id}"))
        except ApiError as exc:
            st.error(exc.message)
else:
    start_form()
    past_runs()
