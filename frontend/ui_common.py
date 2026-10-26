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
                method, f"{self.base_url}{path}", timeout=self.timeout, **kwargs
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
    html_block(CSS)
    try:
        info = get_client().get("/health")
    except ApiError:
        info = None
    with st.sidebar:
        html_block(
            f'<div class="rf-brand"><div class="rf-logo">{LOGO_SVG}</div><div>'
            '<div class="rf-brand-name">Research<b>Flow</b> AI</div>'
            '<div class="rf-brand-tag">Grounded answers from your documents</div></div></div>'
        )
        html_block('<div class="rf-side-label">Workspace</div>')
        st.page_link("views/research.py", label="Ask your documents", icon=":material/search:")
        st.page_link("views/documents.py", label="Document library", icon=":material/folder_open:")
        st.page_link("views/history.py", label="Past research", icon=":material/history:")
        st.write("")
        html_block('<div class="rf-side-label">Status</div>')
        if info is None:
            _status("bad", "Backend offline", f"Start the API at {API_BASE_URL}")
        elif info["llm_configured"]:
            keys = info["llm_keys"]
            if keys["available_entries"]:
                _status(
                    "ok",
                    "AI assistant ready",
                    f"{keys['active_model']} · {plural(keys['total_keys'], 'API key')}",
                )
            else:
                _status("warn", "AI quota used up", "Answers will fail until the quota resets")
            with st.expander("System details"):
                st.markdown(
                    f"- Model in use: `{keys['active_model']}` (key {keys['active_slot']} "
                    f"of {keys['total_keys']})\n"
                    f"- Fallback models: {', '.join(info['llm_fallback_models']) or 'none'}\n"
                    f"- Usable key/model combinations: {keys['available_entries']} of "
                    f"{keys['total_entries']}\n"
                    f"- Embeddings: `{info['embedding_model']}`\n"
                    f"- Automatic retries per question: up to {info['max_retries']}"
                )
        else:
            _status("warn", "Limited mode", "No Gemini API key — add GOOGLE_API_KEY_1")
        html_block(
            '<div class="rf-side-foot">LangGraph agent · Gemini · FAISS vector search<br>'
            "Answers cite only your documents.</div>"
        )
    return info


def _status(tone: str, title: str, subtitle: str) -> None:
    html_block(
        f'<div class="rf-status {tone}"><div class="rf-status-top"><span class="rf-pulse"></span>'
        f'{esc(title)}</div><div class="rf-status-sub">{esc(subtitle)}</div></div>'
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
