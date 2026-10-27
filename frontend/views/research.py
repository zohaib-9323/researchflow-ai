from __future__ import annotations

import time
from typing import Any

import streamlit as st
from ui_common import (
    RETRY_NODES,
    STAGES,
    STATUS_INFO,
    ApiError,
    can_manage_documents,
    chip,
    cite_links,
    collection_selector,
    credits_left,
    esc,
    examples_for,
    file_badge,
    get_client,
    hero,
    html_block,
    load_sample_documents,
    out_of_credits_notice,
    plural,
    section,
    setup_page,
)

client = get_client()
info = setup_page()
state = st.session_state

hero(
    "Research assistant",
    'Ask your <span class="rf-grad">documents</span>',
    "Answers grounded in your own PDFs and text files. Every claim links to the exact passage "
    "it came from, is fact-checked by an agent, and the assistant says so when the documents "
    "don't know.",
)

# -- current conversation -------------------------------------------------------------------
conversation: dict[str, Any] | None = None
if state.get("conversation_id"):
    try:
        conversation = client.get(f"/conversations/{state['conversation_id']}")
        saved = client.get(f"/conversations/{state['conversation_id']}/reports")
        state["reports"] = {r["turn_id"]: r for r in saved}
    except ApiError as exc:
        st.warning(f"Could not reopen that conversation: {exc.message}")
        state.pop("conversation_id", None)

left, right = st.columns([4, 1], vertical_alignment="bottom")
with left:
    collection = collection_selector(
        "research_collection",
        disabled=conversation is not None,  # a conversation stays with one collection
        preselect=conversation["collection_id"] if conversation else state.get("collection_id"),
    )
with right:
    if st.button(
        "New research",
        icon=":material/add:",
        use_container_width=True,
        disabled=conversation is None,
    ):
        for key in ("conversation_id", "reports"):
            state.pop(key, None)
        st.rerun()

collection_id = collection["id"] if collection else None
doc_count = collection["indexed_documents"] if collection else 0
llm_ready = bool(info and info["llm_configured"])

if conversation is not None:
    html_block(
        '<div class="rf-row" style="margin-top:6px">'
        + chip("Conversation in progress", "brand", dot=True)
        + chip(plural(conversation["turn_count"], "question"))
        + '<span class="rf-muted">Follow-up questions can refer to earlier answers.</span></div>'
    )


# -- rendering helpers ----------------------------------------------------------------------
def answer_body(turn: dict[str, Any], anchor: str) -> str:
    if turn["status"] == "error":
        return (
            f'<div class="rf-callout bad"><b>The AI service failed</b>{esc(turn["answer"])}'
            "<br>Nothing was made up: no answer is shown when the AI service fails. Try again "
            "in a minute.</div>"
        )
    claims = [c for c in turn.get("claims") or [] if c.get("supported", True)]
    if claims and turn["status"] in ("answered", "partial"):
        items = "".join(
            f"<li>{esc(c['text'])}"
            + "".join(
                f'<a class="rf-cite" href="#{anchor}-{esc(sid)}">{esc(sid)}</a>'
                for sid in c["citations"]
            )
            + "</li>"
            for c in claims
        )
        return f'<ol class="rf-claims">{items}</ol>'
    text = turn["answer"].strip() or "No answer."
    return f'<div class="rf-answer-text">{cite_links(text, anchor)}</div>'


def pipeline_track(execution: dict[str, Any]) -> str:
    nodes = []
    for step in execution["steps"]:
        short = STAGES.get(step["node"], (step["node"], ""))[0]
        if step["node"] in RETRY_NODES:
            nodes.append(f'<span class="rf-node retry"><i>↻</i>{esc(short)}</span>')
        else:
            nodes.append(f'<span class="rf-node"><i>✓</i>{esc(short)}</span>')
    arrow = '<span class="rf-arrow">→</span>'
    return (
        '<div class="rf-track"><span class="rf-track-label">Agent path</span>'
        + arrow.join(nodes)
        + "</div>"
    )


def sources_grid(turn: dict[str, Any], anchor: str) -> str:
    cards = []
    for cite in turn["citations"]:
        page = f"p. {cite['page']}" if cite["page"] is not None else ""
        cards.append(
            f'<div class="rf-src" id="{anchor}-{esc(cite["source_id"])}">'
            f'<div class="rf-src-head"><span class="rf-src-id">{esc(cite["source_id"])}</span>'
            f'<span class="rf-src-name" title="{esc(cite["document_name"])}">'
            f"{esc(cite['document_name'])}</span>"
            f'<span class="rf-src-page">{esc(page)}</span></div>'
            f'<div class="rf-src-quote">“{esc(cite["excerpt"])}”</div></div>'
        )
    return f'<div class="rf-sources">{"".join(cards)}</div>'


def render_trace(turn: dict[str, Any]) -> None:
    execution = turn["execution"]
    evaluation = turn.get("evaluation")
    with st.expander("Agent trace — how this answer was produced", icon=":material/account_tree:"):
        passed = "Passed" if execution["passed_evaluation"] else "Not passed"
        html_block(
            '<div class="rf-mini">'
            f"<div><small>Searches</small><strong>{execution['retrieval_attempts']}</strong></div>"
            f"<div><small>Automatic retries</small><strong>{execution['retries_used']} of "
            f"{execution['max_retries']}</strong></div>"
            f"<div><small>Fact-check</small><strong>{passed}</strong></div></div>"
        )
        steps = "".join(
            f'<li class="{"retry" if s["node"] in RETRY_NODES else ""}">'
            f"<b>{esc(STAGES.get(s['node'], ('', s['node']))[1])}</b> "
            f"<span>— {esc(s['summary'])}</span></li>"
            for s in execution["steps"]
        )
        html_block(f'<ul class="rf-trace">{steps}</ul>')
        if evaluation:
            checker = (
                "AI fact-checker (an estimate, not a guarantee)"
                if evaluation["llm_judged"]
                else "automatic citation check"
            )
            html_block(
                f'<div class="rf-muted"><b>Fact-check</b> · {esc(checker)}</div>'
                f'<div style="font-size:14px;line-height:1.6;margin:4px 0 12px">'
                f"{esc(evaluation['explanation'])}</div>"
            )
        if execution["search_queries"]:
            html_block(
                '<div class="rf-row"><span class="rf-muted"><b>Searched for</b></span>'
                + "".join(chip(q, "brand") for q in execution["search_queries"])
                + "</div>"
            )
        if execution.get("models_used"):
            st.caption("Model: " + ", ".join(execution["models_used"]))


def render_passages(turn: dict[str, Any]) -> None:
    if not turn["evidence"]:
        return
    cited = {c["source_id"] for c in turn["citations"]}
    with st.expander(
        f"All passages found ({len(turn['evidence'])})", icon=":material/format_quote:"
    ):
        st.caption("Everything the search returned, best match first. ★ = used in the answer.")
        blocks = []
        for passage in turn["evidence"]:
            page = f" · p. {passage['page']}" if passage["page"] is not None else ""
            star = " ★" if passage["source_id"] in cited else ""
            width = max(0, min(100, round(passage["score"] * 100)))
            blocks.append(
                '<div class="rf-passage"><div class="rf-passage-head">'
                f'<span class="rf-src-id">{esc(passage["source_id"])}</span>'
                f"<b>{esc(passage['document_name'])}</b>{esc(page)}{star}"
                f'<span style="flex:1"></span><span class="rf-bar"><span style="width:{width}%">'
                f"</span></span>{width}%</div>"
                f'<div class="rf-passage-text">{esc(passage["text"])}</div></div>'
            )
        html_block("".join(blocks))


def render_report(turn: dict[str, Any]) -> None:
    if not (turn.get("turn_id") and turn["evidence"] and turn["status"] in ("answered", "partial")):
        return
    reports: dict[str, dict] = state.setdefault("reports", {})
    report = reports.get(turn["turn_id"])
    if report is None:
        if st.button(
            "Turn this into a research report",
            key=f"report_{turn['turn_id']}",
            icon=":material/description:",
        ):
            with st.spinner("Writing a structured report from the evidence (about 10-20 s)…"):
                try:
                    reports[turn["turn_id"]] = client.post("/reports", {"turn_id": turn["turn_id"]})
                except ApiError as exc:
                    st.error(exc.message)
                else:
                    st.rerun()
        return
    content = report["content"]
    findings = content["key_findings"]
    html_block(
        '<div class="rf-report"><div class="rf-row">'
        + chip("Research report", "brand", dot=True)
        + chip(plural(len(findings), "finding"))
        + chip(plural(len(content["sources"]), "source"))
        + f'</div><div class="rf-report-title">{esc(report["title"])}</div>'
        f'<div class="rf-report-sum">{esc(content["executive_summary"])}</div></div>'
    )
    read, download = st.columns([3, 1], vertical_alignment="top")
    with read.expander("Read the full report", icon=":material/article:"):
        st.markdown(report["markdown"])
    download.download_button(
        "Download .md",
        data=report["markdown"],
        file_name=f"{report['title'][:60].replace(' ', '_') or 'report'}.md",
        mime="text/markdown",
        key=f"dl_{report['id']}",
        icon=":material/download:",
        use_container_width=True,
    )


def render_question(text: str, number: int) -> None:
    html_block(
        f'<div class="rf-q"><div class="rf-q-avatar">Q{number}</div><div>'
        f'<div class="rf-q-label">Your question</div><div class="rf-q-text">{esc(text)}</div>'
        "</div></div>"
    )


def render_turn(turn: dict[str, Any], number: int) -> None:
    render_question(turn["question"], number)
    anchor = f"rf-{number}"
    tone, label, explanation = STATUS_INFO.get(turn["status"], ("", turn["status"], ""))
    execution = turn["execution"]
    with st.container(key=f"answer_{number}"):
        meta = [
            chip(plural(len(turn["citations"]), "source")),
            chip(plural(execution["retrieval_attempts"], "search", "searches")),
        ]
        if execution["retries_used"]:
            meta.append(chip(plural(execution["retries_used"], "retry", "retries"), "warn"))
        if (seconds := state.get("durations", {}).get(turn.get("turn_id"))) is not None:
            meta.append(chip(f"{seconds:.0f} s"))
        html_block(
            '<div class="rf-answer-head">'
            + chip(label, tone, dot=True, large=True)
            + f'<div class="rf-row">{"".join(meta)}</div></div>'
            + (f'<div class="rf-answer-note">{esc(explanation)}</div>' if explanation else "")
            + answer_body(turn, anchor)
            + (pipeline_track(execution) if execution["steps"] else "")
        )
        if turn["citations"]:
            section("Sources", "Click a citation in the answer to jump to its passage.")
            html_block(sources_grid(turn, anchor))
        if turn["limitations"]:
            callout = "warn" if turn["status"] == "partial" else "info"
            items = "".join(f"<li>{esc(item)}</li>" for item in turn["limitations"])
            html_block(
                f'<div class="rf-callout {callout}"><b>What\'s missing or uncertain</b>'
                f"<ul>{items}</ul></div>"
            )
        st.write("")
        render_trace(turn)
        render_passages(turn)
        render_report(turn)


def ask(question: str, number: int) -> None:
    render_question(question, number)
    with st.container(key="thinking"):
        html_block(
            '<div class="rf-think-head"><span class="rf-spin"></span>Researching your documents…'
            '</div><div class="rf-think-steps"><span>Understand the question</span>'
            "<span>Search passages</span><span>Draft an answer</span>"
            "<span>Fact-check &amp; cite</span></div>"
            '<div class="rf-muted">Usually 15 to 60 seconds. If the AI service is busy the agent '
            "waits and retries automatically.</div>"
        )
        started = time.monotonic()
        try:
            result = client.post(
                "/research",
                {
                    "question": question,
                    "collection_id": collection_id,
                    "conversation_id": state.get("conversation_id"),
                },
            )
        except ApiError as exc:
            st.error(exc.message)
            return
    state.setdefault("durations", {})[result["turn_id"]] = time.monotonic() - started
    state["conversation_id"] = result["conversation_id"]
    state["collection_id"] = collection_id
    st.rerun()


def render_welcome() -> None:
    html_block(
        '<div class="rf-steps">'
        '<div class="rf-step"><div class="rf-step-n">01 · ASK</div><div class="rf-step-t">'
        'Ask in plain language</div><div class="rf-step-d">Any question about the documents '
        "in the selected collection.</div></div>"
        '<div class="rf-step"><div class="rf-step-n">02 · SEARCH</div><div class="rf-step-t">'
        'Search by meaning</div><div class="rf-step-d">Relevant passages are found with '
        "semantic vector search.</div></div>"
        '<div class="rf-step"><div class="rf-step-n">03 · VERIFY</div><div class="rf-step-t">'
        'Fact-check the draft</div><div class="rf-step-d">Unsupported claims are removed; '
        "weak evidence triggers a new search.</div></div>"
        '<div class="rf-step"><div class="rf-step-n">04 · CITE</div><div class="rf-step-t">'
        'Show the receipts</div><div class="rf-step-d">Every claim links to its document, '
        "page and quote.</div></div></div>"
    )
    try:
        documents = client.get("/documents", collection_id=collection_id)
    except ApiError:
        documents = []
    examples = examples_for([d["filename"] for d in documents])
    section("Try an example", "Picked for the documents in this collection.")
    with st.container(key="examples"):
        columns = st.columns(2)
        for number, example in enumerate(examples):
            honesty_test = number == len(examples) - 1 and len(examples) == 4
            if columns[number % 2].button(
                example + ("  (not in the documents: tests honesty)" if honesty_test else ""),
                key=f"example_{number}",
                icon=":material/shield:" if honesty_test else ":material/arrow_outward:",
                disabled=not llm_ready or credits_left("question") == 0,
            ):
                state["pending_question"] = example
                st.rerun()
    if documents:
        section("In this collection", plural(len(documents), "document"))
        html_block(
            '<div class="rf-row">'
            + "".join(
                f'<span class="rf-chip">{file_badge(d["file_type"], small=True)}'
                f"{esc(d['filename'])}</span>"
                for d in documents[:12]
            )
            + (chip(f"+{len(documents) - 12} more") if len(documents) > 12 else "")
            + "</div>"
        )


# -- page body ------------------------------------------------------------------------------
turn_count = len(conversation["turns"]) if conversation else 0
if conversation is not None:
    for number, turn in enumerate(conversation["turns"], start=1):
        render_turn(turn, number)
elif collection is None:
    st.error("No collections available — is the backend running?")
elif doc_count == 0:
    with st.container(key="emptystate"):
        html_block(
            '<div class="rf-empty"><div class="rf-empty-icon">📂</div>'
            "<h3>This collection has no documents yet</h3><p>Add PDF or TXT files first. Not "
            "sure what to use? Load the sample documents — three short reports about a "
            "fictional town — and try the example questions.</p></div>"
        )
        first, second = st.columns(2)
        if can_manage_documents() and first.button(
            "Load sample documents",
            type="primary",
            icon=":material/auto_stories:",
            use_container_width=True,
        ):
            load_sample_documents(collection_id)
            st.rerun()
        second.page_link(
            "views/documents.py",
            label="Upload my own documents",
            icon=":material/upload:",
            use_container_width=True,
        )
else:
    render_welcome()

if info is not None and not llm_ready:
    st.warning("The AI model isn't configured, so questions can't be answered yet.")

empty = doc_count == 0 and conversation is None
if empty:
    placeholder = "Add documents first…"
elif conversation is not None:
    placeholder = "Ask a follow-up question…"
else:
    placeholder = f"Ask anything about {collection['name'] if collection else 'your documents'}…"
remaining = credits_left("question")
if remaining == 0:
    out_of_credits_notice("question")
elif remaining is not None:
    st.caption(f"{plural(remaining, 'free question')} left.")
typed = st.chat_input(
    placeholder, disabled=not llm_ready or empty or remaining == 0, max_chars=2000
)
question = typed or state.pop("pending_question", None)
if question:
    ask(question, turn_count + 1)
