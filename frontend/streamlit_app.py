"""ResearchFlow AI — Streamlit entry point.

Run with:  streamlit run frontend/streamlit_app.py

On single-process hosts (Streamlit Community Cloud) the API runs inside this process,
see embedded.py.

Access control, in order of preference:
* Google sign-in, when an [auth] section is present in Streamlit secrets. Every user gets
  free credits (FREE_QUESTIONS / FREE_AGENT_RUNS); emails in ADMIN_EMAILS are unlimited.
  Credits are enforced by the API, which only trusts requests carrying the internal token.
* Otherwise APP_ACCESS_CODE, a shared code, if set.
* Otherwise the app is open (local development).
"""

from __future__ import annotations

import hmac
import os

import embedded
import streamlit as st
from ui_common import CSS, auth_configured, hero, html_block

st.set_page_config(page_title="ResearchFlow AI", page_icon="🔎", layout="wide")

embedded.export_secrets()
if embedded.enabled():
    embedded.configure_auth(auth_configured())
    embedded.ensure_backend()

ACCESS_CODE = os.environ.get("APP_ACCESS_CODE", "")


def _centered_hero(eyebrow: str, subtitle: str) -> None:
    html_block(CSS + "<style>[data-testid='stSidebar']{display:none}</style>")
    st.write("")
    hero(eyebrow, 'Research<span class="rf-grad">Flow</span> AI', subtitle)


def sign_in_page() -> None:
    _, middle, _ = st.columns([1, 2, 1])
    with middle:
        _centered_hero(
            "Free to try",
            "Cited, fact-checked answers from your documents, and an autonomous agent that "
            "researches and writes reports. Sign in with Google to get free credits: "
            f"{os.environ.get('FREE_QUESTIONS', '2')} questions and "
            f"{os.environ.get('FREE_AGENT_RUNS', '2')} deep research runs.",
        )
        if st.button(
            "Continue with Google",
            type="primary",
            icon=":material/login:",
            use_container_width=True,
        ):
            st.login("google")
        st.caption("We only use your Google account's name and email to keep your credits.")


def unverified_page() -> None:
    _, middle, _ = st.columns([1, 2, 1])
    with middle:
        _centered_hero("Sign-in problem", "Your Google email address isn't verified.")
        if st.button("Sign out", use_container_width=True):
            st.logout()


def access_gate() -> None:
    _, middle, _ = st.columns([1, 2, 1])
    with middle:
        _centered_hero(
            "Private demo",
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


if auth_configured():
    if not st.user.is_logged_in:
        sign_in_page()
        st.stop()
    if st.user.get("email_verified") is False:
        unverified_page()
        st.stop()
elif ACCESS_CODE and not st.session_state.get("access_granted"):
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
