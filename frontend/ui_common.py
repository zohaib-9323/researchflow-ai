"""Shared helpers for the Streamlit pages: a thin API client, the design system and
small HTML components.

The UI talks to the FastAPI backend only; it contains no business logic. All dynamic
text that ends up in raw HTML goes through ``esc``.
"""

from __future__ import annotations

import html
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")
API_TIMEOUT = float(os.environ.get("API_TIMEOUT_SECONDS", "180"))
SAMPLE_DOCS_DIR = Path(__file__).resolve().parent.parent / "evaluation" / "docs"

# status -> (tone, short label, explanation)
STATUS_INFO = {
    "answered": ("ok", "Verified answer", "Every claim was checked against your documents."),
    "partial": (
        "warn",
        "Partial answer",
        "Only part of the question could be backed by your documents.",
    ),
    "insufficient_evidence": (
        "info",
        "Not found in your documents",
        "Your documents don't contain enough information to answer this.",
    ),
    "not_applicable": (
        "info",
        "Not a document question",
        "Ask something about the content of your documents.",
    ),
    "rejected": ("bad", "Question not accepted", ""),
    "error": ("bad", "Something went wrong", ""),
}
DOC_STATUS = {
    "indexed": ("ok", "Ready"),
    "processing": ("warn", "Processing"),
    "failed": ("bad", "Failed"),
}
# node -> (short label for the pipeline track, sentence for the trace)
STAGES = {
    "classify_question": ("Understand", "Understood the question"),
    "retrieve_documents": ("Search", "Searched your documents"),
    "generate_answer": ("Draft", "Drafted an answer from the passages"),
    "evaluate_evidence": ("Fact-check", "Fact-checked the answer against the passages"),
    "revise_search": ("Re-search", "Evidence was thin, so it tried a different search"),
    "revise_answer": ("Revise", "Removed or fixed unsupported claims"),
    "finalize_response": ("Cite", "Verified citations and finished"),
}
RETRY_NODES = {"revise_search", "revise_answer"}

# Example questions are picked by matching a document filename in the collection.
EXAMPLE_SETS = [
    (
        "larkspur_",
        [
            "Why do the pilot report and the council memo give different savings figures?",
            "How much did the solar pilot cost and who paid for it?",
            "How many wells exceeded the nitrate limit, and where were they?",
            "What was the hourly wage of the solar installers?",
        ],
    ),
    (
        "attention_is_all_you_need",
        [
            "How does Self-RAG differ from the original RAG approach?",
            "Why do language models struggle with information in the middle of a long context?",
            "How does Sentence-BERT make semantic similarity search faster than BERT?",
            "Which of these papers won a Turing Award?",
        ],
    ),
]
GENERIC_EXAMPLES = [
    "What are the main findings across these documents?",
    "Where do these documents agree, and where do they disagree?",
    "What limitations or open questions do the authors mention?",
]

LOGO_SVG = (
    '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" '
    'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">'
    '<circle cx="10.5" cy="10.5" r="6.5"/><path d="M15.5 15.5 21 21"/>'
    '<path d="M8 11.5l1.8 1.8L13.5 9"/></svg>'
)

CSS = "<style>" + (Path(__file__).with_name("styles.css")).read_text(encoding="utf-8") + "</style>"


class ApiError(Exception):
    def __init__(self, message: str, code: str = "error", status: int = 0) -> None:
        super().__init__(message)
        self.message, self.code, self.status = message, code, status


class ApiClient:
    def __init__(self, base_url: str = API_BASE_URL, timeout: float = API_TIMEOUT) -> None:
        self.base_url = base_url
        self.timeout = timeout

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = httpx.request(
                method,
                f"{self.base_url}{path}",
                timeout=self.timeout,
                headers=identity_headers(),
                **kwargs,
            )
        except httpx.TimeoutException as exc:
            raise ApiError(
                "The request took too long. The AI provider may be busy — please try again.",
                "timeout",
            ) from exc
        except httpx.TransportError as exc:
            raise ApiError(
                f"Can't reach the ResearchFlow backend at {self.base_url}. Is it running?",
                "unreachable",
            ) from exc
        if response.status_code >= 400:
            try:
                error = response.json().get("error", {})
            except ValueError:
                error = {}
            raise ApiError(
                error.get("message", f"Request failed ({response.status_code})."),
                error.get("code", "error"),
                response.status_code,
            )
        return response

    def get(self, path: str, **params: Any) -> Any:
        return self._request("GET", path, params={k: v for k, v in params.items() if v}).json()

    def post(self, path: str, payload: dict[str, Any]) -> Any:
        return self._request("POST", path, json=payload).json()

    def delete(self, path: str) -> None:
        self._request("DELETE", path)

    def upload(self, files: list[tuple[str, bytes, str]], collection_id: str | None) -> Any:
        return self._request(
            "POST",
            "/documents/upload",
            files=[("files", f) for f in files],
            data={"collection_id": collection_id} if collection_id else None,
        ).json()


@st.cache_resource
def get_client() -> ApiClient:
    return ApiClient()


# -- signed-in user ----------------------------------------------------------------------------
def auth_configured() -> bool:
    """Google sign-in is configured: an [auth] section with a real Google client in secrets."""
    try:
        auth = st.secrets.get("auth", {})
        google = auth.get("google", {})
        values = [auth.get("redirect_uri"), auth.get("cookie_secret")]
        values += [google.get("client_id"), google.get("client_secret")]
    except Exception:  # no secrets file
        return False
    return all(isinstance(v, str) and v.strip() and "PASTE" not in v for v in values)


def signed_in_user() -> dict[str, str] | None:
    if not auth_configured() or not st.user.is_logged_in:
        return None
    email = str(st.user.get("email") or "").strip().lower()
    return {"email": email, "name": str(st.user.get("name") or "")} if email else None


def identity_headers() -> dict[str, str]:
    """Who is asking, for the API: the shared internal token plus the verified email.
    Computed per request because the API client is shared by all sessions."""
    headers: dict[str, str] = {}
    token = os.environ.get("INTERNAL_API_TOKEN")
    if token:
        headers["X-Internal-Token"] = token
    user = signed_in_user()
    if user:
        headers["X-User-Email"] = user["email"]
        # HTTP headers are latin-1; drop characters that can't be sent.
        headers["X-User-Name"] = user["name"].encode("latin-1", "ignore").decode()[:200]
    return headers


def account() -> dict[str, Any]:
    """The /me response for this rerun (credits, admin flag); unlimited when sign-in is
    off or the backend is unreachable."""
    return st.session_state.get("_account") or {"auth_enabled": False, "is_admin": True}


def credits_left(kind: str) -> int | None:
    """Remaining free credits ('question' or 'agent_run'); None means unlimited."""
    credits = account().get("credits")
    return None if not credits else credits[kind]["remaining"]


def can_manage_documents() -> bool:
    return bool(account().get("is_admin", True))


def out_of_credits_notice(kind: str) -> None:
    label = "questions" if kind == "question" else "deep research runs"
    st.info(
        f"You've used all your free {label}. Thanks for trying ResearchFlow AI! "
        "Contact the administrator for more access.",
        icon=":material/hourglass_empty:",
    )


# -- tiny HTML components ----------------------------------------------------------------------
def esc(value: Any) -> str:
    return html.escape(str(value))


def html_block(markup: str) -> None:
    """Render raw HTML. Callers build markup without blank lines so Markdown leaves it alone."""
    st.markdown(markup, unsafe_allow_html=True)


def chip(text: str, tone: str = "", dot: bool = False, large: bool = False) -> str:
    classes = " ".join(c for c in ("rf-chip", tone, "lg" if large else "") if c)
    marker = '<span class="rf-dot"></span>' if dot else ""
    return f'<span class="{classes}">{marker}{esc(text)}</span>'


def file_badge(file_type: str, small: bool = False) -> str:
    kind = file_type.lower()
    size = " sm" if small else ""
    return f'<span class="rf-file {esc(kind)}{size}">{esc(kind.upper()[:4])}</span>'


# Line icons (stroke SVG, inherit the text colour), so the UI uses one drawn icon set
# instead of emoji.
ICONS = {
    "search": '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>',
    "doc": '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/>'
    '<path d="M14 3v5h5M9 13h6M9 17h4"/>',
    "check": '<path d="m5 12.5 4.2 4.2L19 7"/>',
    "shield": '<path d="M12 3 5 6v5c0 4.4 3 8.3 7 10 4-1.7 7-5.6 7-10V6z"/>'
    '<path d="m9 12 2 2 4-4"/>',
    "quote": '<path d="M7 17c-2 0-3-1.5-3-3.5C4 10 6 7.5 9 7M17 17c-2 0-3-1.5-3-3.5'
    'C14 10 16 7.5 19 7"/>',
    "spark": '<path d="M12 3v4M12 17v4M3 12h4M17 12h4M6.3 6.3l2.5 2.5M15.2 15.2l2.5 2.5'
    'M6.3 17.7l2.5-2.5M15.2 8.8l2.5-2.5"/>',
    "map": '<path d="M9 4 3 6.5v13L9 17l6 2.5 6-2.5V4l-6 2.5z"/><path d="M9 4v13M15 6.5v13"/>',
    "compare": '<path d="M8 3v18M16 3v18"/><path d="m4 8 4-4 4 4M12 16l4 4 4-4"/>',
    "calc": '<rect x="5" y="3" width="14" height="18" rx="2"/><path d="M8 7h8M8 12h.01M12 12h.01'
    'M16 12h.01M8 16h.01M12 16h.01M16 16h.01"/>',
    "globe": '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c2.5 2.7 3.8 5.7 3.8 9'
    's-1.3 6.3-3.8 9c-2.5-2.7-3.8-5.7-3.8-9S9.5 5.7 12 3z"/>',
    "eye": '<path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z"/><circle cx="12" '
    'cy="12" r="3"/>',
    "retry": '<path d="M3 12a9 9 0 0 1 15.5-6.2L21 8M21 3v5h-5M21 12a9 9 0 0 1-15.5 6.2L3 16'
    'M3 21v-5h5"/>',
    "pen": '<path d="M4 20h4L19 9a2.8 2.8 0 0 0-4-4L4 16z"/><path d="m13.5 6.5 4 4"/>',
    "flag": '<path d="M5 21V4M5 4h11l-2 4 2 4H5"/>',
    "play": '<path d="m7 4 13 8-13 8z"/>',
    "lock": '<rect x="5" y="11" width="14" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/>',
    "list": '<path d="M9 6h11M9 12h11M9 18h11M4 6h.01M4 12h.01M4 18h.01"/>',
}


def icon(name: str) -> str:
    return f'<svg class="rf-ico" viewBox="0 0 24 24" aria-hidden="true">{ICONS.get(name, "")}</svg>'


def orb(size: int = 38, live: bool = False) -> str:
    """The assistant's animated presence (spins and breathes while it is working)."""
    return f'<span class="rf-orb{" live" if live else ""}" style="--s:{size}px"></span>'


def initials(name: str) -> str:
    parts = [p for p in re.split(r"[\s._@-]+", name) if p]
    return "".join(p[0] for p in parts[:2]).upper() or "?"


def hero(eyebrow: str, title_html: str, subtitle: str) -> None:
    html_block(
        f'<div class="rf-hero"><span class="rf-eyebrow"><i></i>{esc(eyebrow)}</span>'
        f'<div class="rf-title">{title_html}</div><div class="rf-sub">{esc(subtitle)}</div></div>'
    )


def section(title: str, note: str = "") -> None:
    html_block(f'<div class="rf-section"><h4>{esc(title)}</h4><span>{esc(note)}</span></div>')


def stat_cards(items: list[tuple[str, Any]]) -> None:
    cells = "".join(
        f'<div class="rf-stat"><small>{esc(label)}</small><strong>{esc(value)}</strong></div>'
        for label, value in items
    )
    html_block(f'<div class="rf-stats">{cells}</div>')


def cite_links(text: str, anchor_prefix: str) -> str:
    """Escape ``text`` and turn [S1] markers into links to the matching source card."""
    return re.sub(
        r"\[(S\d+)\]",
        lambda m: f'<a class="rf-cite" href="#{anchor_prefix}-{m[1]}">{m[1]}</a>',
        esc(text),
    )


# -- page chrome -------------------------------------------------------------------------------
def setup_page() -> dict[str, Any] | None:
    """Inject styles and render the sidebar; returns backend health (None if down)."""
    html_block(CSS + '<div class="rf-aurora"><i></i><i></i><i></i></div>')
    try:
        info = get_client().get("/health")
    except ApiError:
        info = None
    try:
        st.session_state["_account"] = get_client().get("/me") if info else None
    except ApiError:
        st.session_state["_account"] = None
    with st.sidebar:
        html_block(
            f'<div class="rf-brand"><div class="rf-logo">{LOGO_SVG}</div><div>'
            '<div class="rf-brand-name">Research<b>Flow</b></div>'
            '<div class="rf-brand-tag">Your documents, thoughtfully researched</div></div></div>'
        )
        html_block('<div class="rf-side-label">Explore</div>')
        st.page_link("views/research.py", label="Ask your documents", icon=":material/search:")
        st.page_link("views/agent.py", label="Deep research", icon=":material/travel_explore:")
        st.page_link("views/documents.py", label="Document library", icon=":material/folder_open:")
        st.page_link("views/history.py", label="Past research", icon=":material/history:")
        user = signed_in_user()
        if user:
            st.write("")
            html_block('<div class="rf-side-label">Account</div>')
            _account_box(user)
        st.write("")
        html_block('<div class="rf-side-label">Status</div>')
        if info is None:
            _status("bad", "Backend offline", f"Start the API at {API_BASE_URL}")
        elif info["llm_configured"]:
            keys = info["llm_keys"]
            if keys["available_entries"]:
                _status("ok", "Ready to research", "Every answer is checked against sources")
            else:
                _status("warn", "Taking a short break", "The AI's daily limit was reached")
            if can_manage_documents():  # technical details are for the administrator
                _system_details(info, keys)
        else:
            _status("warn", "Limited mode", "No Gemini API key: add GOOGLE_API_KEY_1")
        html_block(
            '<div class="rf-side-foot">Answers come only from your documents, with a '
            "citation for every claim.</div>"
        )
    return info


def _system_details(info: dict[str, Any], keys: dict[str, Any]) -> None:
    with st.sidebar, st.expander("System details"):
        st.markdown(
            f"- Model in use: `{keys['active_model']}` (key {keys['active_slot']} "
            f"of {keys['total_keys']})\n"
            f"- Fallback models: {', '.join(info['llm_fallback_models']) or 'none'}\n"
            f"- Usable key/model combinations: {keys['available_entries']} of "
            f"{keys['total_entries']}\n"
            f"- Embeddings: `{info['embedding_model']}`\n"
            f"- Automatic retries per question: up to {info['max_retries']}"
        )


def _meter(label: str, credit: dict[str, Any]) -> str:
    left, limit = credit["remaining"], credit["limit"] or 1
    width = max(0, min(100, round(left / limit * 100)))
    return (
        f'<div><div class="rf-meter-top">{esc(label)}<span>{left} of {limit} left</span></div>'
        f'<div class="rf-meter"><b style="width:{width}%"></b></div></div>'
    )


def _account_box(user: dict[str, str]) -> None:
    me = account()
    if me.get("is_admin"):
        detail = f'<span class="rf-admin">{icon("shield")}Administrator · unlimited</span>'
    elif me.get("credits"):
        detail = (
            '<div class="rf-meters">'
            + _meter("Questions", me["credits"]["question"])
            + _meter("Deep research", me["credits"]["agent_run"])
            + "</div>"
        )
    else:
        detail = ""
    name = user["name"] or user["email"]
    html_block(
        '<div class="rf-account" style="display:block"><div style="display:flex;gap:11px;'
        f'align-items:center"><span class="rf-avatar">{esc(initials(name))}</span>'
        f'<div style="min-width:0"><div class="rf-account-name">{esc(name)}</div>'
        f'<div class="rf-account-mail">{esc(user["email"])}</div></div></div>{detail}</div>'
    )
    if st.button("Sign out", icon=":material/logout:", key="sign_out", type="tertiary"):
        st.logout()


def _status(tone: str, title: str, subtitle: str) -> None:
    html_block(
        f'<div class="rf-status {tone}"><span class="rf-pulse"></span><div>'
        f'<div class="rf-status-top">{esc(title)}</div>'
        f'<div class="rf-status-sub">{esc(subtitle)}</div></div></div>'
    )


def list_collections() -> list[dict[str, Any]]:
    try:
        return get_client().get("/collections")
    except ApiError as exc:
        st.error(exc.message)
        return []


def collection_selector(
    key: str, disabled: bool = False, preselect: str | None = None
) -> dict[str, Any] | None:
    collections = list_collections()
    if not collections:
        return None
    ids = [c["id"] for c in collections]
    index = ids.index(preselect) if preselect in ids else 0
    return st.selectbox(
        "Collection",
        collections,
        index=index,
        format_func=lambda c: f"{c['name']}  ·  {plural(c['indexed_documents'], 'document')}",
        key=key,
        disabled=disabled,
        help="Collections keep sets of documents separate. Questions only search the "
        "selected collection.",
    )


def examples_for(filenames: list[str]) -> list[str]:
    for marker, questions in EXAMPLE_SETS:
        if any(marker in name for name in filenames):
            return questions
    return GENERIC_EXAMPLES


def load_sample_documents(collection_id: str | None) -> None:
    files = [
        (path.name, path.read_bytes(), "text/plain")
        for path in sorted(SAMPLE_DOCS_DIR.glob("*.txt"))
    ]
    if not files:
        st.error("Sample documents are not available in this installation.")
        return
    with st.spinner("Indexing sample documents…"):
        try:
            result = get_client().upload(files, collection_id)
        except ApiError as exc:
            st.error(exc.message)
            return
    added = result["succeeded"]
    already = sum(1 for r in result["results"] if r["error_code"] == "duplicate_document")
    st.toast(f"Sample documents ready ({added} added, {already} already there).", icon="📚")


# -- formatting --------------------------------------------------------------------------------
def plural(count: int, word: str, plural_word: str | None = None) -> str:
    return f"{count} {word if count == 1 else plural_word or word + 's'}"


def friendly_time(iso: str | None) -> str:
    """Relative time for the last week ("5 min ago"), a date after that."""
    if not iso:
        return ""
    try:
        moment = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return iso
    if moment.tzinfo is None:  # the API stores UTC
        moment = moment.replace(tzinfo=UTC)
    seconds = (datetime.now(UTC) - moment).total_seconds()
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)} min ago"
    if seconds < 86400:
        return plural(int(seconds // 3600), "hour") + " ago"
    if seconds < 7 * 86400:
        return plural(int(seconds // 86400), "day") + " ago"
    return moment.strftime("%d %b %Y")


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB"):
        if value < 1024 or unit == "MB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"
