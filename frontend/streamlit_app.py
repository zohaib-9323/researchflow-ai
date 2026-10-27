"""ResearchFlow AI — Streamlit entry point.

Run with:  streamlit run frontend/streamlit_app.py

On single-process hosts (Streamlit Community Cloud) the API runs inside this process,
see embedded.py. Set APP_ACCESS_CODE to require a shared code before the app opens (used
for public deployments so visitors can't spend the Gemini quota).
"""

from __future__ import annotations

import hmac
import os

import embedded
import streamlit as st
from ui_common import CSS, hero, html_block

st.set_page_config(page_title="ResearchFlow AI", page_icon="🔎", layout="wide")

embedded.export_secrets()
if embedded.enabled():
    embedded.ensure_backend()

ACCESS_CODE = os.environ.get("APP_ACCESS_CODE", "")


def access_gate() -> None:
    html_block(CSS + "<style>[data-testid='stSidebar']{display:none}</style>")
    _, middle, _ = st.columns([1, 2, 1])
    with middle:
        st.write("")
        hero(
            "Private demo",
            'Research<span class="rf-grad">Flow</span> AI',
            "Cited, fact-checked answers from your own documents. Enter the access code to "
            "continue.",
        )
        with st.form("access"):
            code = st.text_input("Access code", type="password")
            if st.form_submit_button("Enter", type="primary", use_container_width=True):
                if hmac.compare_digest(code.encode(), ACCESS_CODE.encode()):
                    st.session_state["access_granted"] = True
                    st.rerun()
                st.error("That code isn't right.")


if ACCESS_CODE and not st.session_state.get("access_granted"):
    access_gate()
    st.stop()

# The sidebar renders its own branded navigation (see ui_common.setup_page).
pages = [
    st.Page("views/research.py", title="Ask your documents", icon="🔎", default=True),
    st.Page("views/agent.py", title="Deep research", icon="🧭"),
    st.Page("views/documents.py", title="Document library", icon="📚"),
    st.Page("views/history.py", title="Past research", icon="🕘"),
]
st.navigation(pages, position="hidden").run()
