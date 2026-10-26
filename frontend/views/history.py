from __future__ import annotations

import streamlit as st
from ui_common import (
    STATUS_INFO,
    ApiError,
    chip,
    esc,
    friendly_time,
    get_client,
    hero,
    html_block,
    list_collections,
    plural,
    setup_page,
)

client = get_client()
setup_page()

hero(
    "Saved automatically",
    'Past <span class="rf-grad">research</span>',
    "Every conversation is saved, including after a restart. Open one to read it again or "
    "continue with follow-up questions.",
)

try:
    conversations = client.get("/conversations")
except ApiError as exc:
    st.error(exc.message)
    conversations = []

if not conversations:
    with st.container(key="emptystate"):
        html_block(
            '<div class="rf-empty"><div class="rf-empty-icon">🕘</div>'
            "<h3>Nothing here yet</h3><p>Your questions and answers will appear here "
            "automatically.</p></div>"
        )
        st.page_link("views/research.py", label="Ask your first question", icon=":material/search:")
    st.stop()

collection_names = {c["id"]: c["name"] for c in list_collections()}
search = st.text_input(
    "Search", placeholder="Filter by title…", label_visibility="collapsed", icon=":material/search:"
)
shown = [c for c in conversations if search.lower() in c["title"].lower()]
st.caption(plural(len(shown), "conversation"))

for conversation in shown:
    try:
        turns = client.get(f"/conversations/{conversation['id']}")["turns"]
    except ApiError:
        turns = []
    with st.container(key=f"histrow_{conversation['id']}"):
        text_col, button_col = st.columns([5, 1], vertical_alignment="center")
        listed = "".join(
            f'<li><i class="{STATUS_INFO.get(t["status"], ("",))[0]}"></i>{esc(t["question"])}</li>'
            for t in turns[:3]
        )
        more = f'<li class="rf-muted">+{len(turns) - 3} more</li>' if len(turns) > 3 else ""
        text_col.markdown(
            '<div class="rf-doc-name" style="white-space:normal">'
            f"{esc(conversation['title'])}</div>"
            '<div class="rf-row" style="margin-top:8px">'
            + chip(
                collection_names.get(conversation["collection_id"], "Unknown collection"), "brand"
            )
            + chip(plural(conversation["turn_count"], "question"))
            + (
                chip(
                    STATUS_INFO[turns[-1]["status"]][1],
                    STATUS_INFO[turns[-1]["status"]][0],
                    dot=True,
                )
                if turns and turns[-1]["status"] in STATUS_INFO
                else ""
            )
            + f'<span class="rf-muted">{esc(friendly_time(conversation["updated_at"]))}</span>'
            + "</div>"
            + (f'<ul class="rf-turns">{listed}{more}</ul>' if len(turns) > 1 else ""),
            unsafe_allow_html=True,
        )
        if button_col.button(
            "Open",
            key=f"open_{conversation['id']}",
            type="primary",
            icon=":material/open_in_new:",
            use_container_width=True,
        ):
            st.session_state["conversation_id"] = conversation["id"]
            st.session_state["collection_id"] = conversation["collection_id"]
            st.session_state.pop("reports", None)
            st.switch_page("views/research.py")
