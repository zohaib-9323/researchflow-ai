"""Run the FastAPI backend inside the Streamlit process.

Single-app hosts such as Streamlit Community Cloud run one process, so the UI starts the
API on a private localhost port and keeps talking to it over HTTP exactly as it does when
the two run separately. Enabled with EMBEDDED_API=true (automatic on Streamlit Cloud).
"""

from __future__ import annotations

import logging
import os
import secrets
import sys
import threading
import time
from pathlib import Path

import httpx
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
HOST, PORT = "127.0.0.1", 8000
logger = logging.getLogger("researchflow.embedded")


def export_secrets() -> None:
    """Expose root-level Streamlit secrets as environment variables.

    The backend's settings and the access gate read the environment; Streamlit only copies
    secrets there lazily, so do it explicitly before anything reads them. Real
    environment variables win.
    """
    try:
        items = dict(st.secrets)
    except Exception:  # no secrets file: local runs configure via .env files instead
        return
    for key, value in items.items():
        if isinstance(value, str | int | float | bool):
            os.environ.setdefault(key, str(value))


def configure_auth(sign_in: bool) -> None:
    """With Google sign-in, turn on per-user credits in the embedded API.

    The API trusts the signed-in email only from requests carrying INTERNAL_API_TOKEN.
    The API is private to this process, so a random token is generated when none is set.
    Must run before ensure_backend() starts the API.
    """
    if not sign_in:
        return
    os.environ.setdefault("AUTH_ENABLED", "true")
    os.environ.setdefault("INTERNAL_API_TOKEN", secrets.token_urlsafe(32))


def enabled() -> bool:
    flag = os.environ.get("EMBEDDED_API", "").strip().lower()
    if flag:
        return flag in {"1", "true", "yes"}
    return Path("/mount/src").is_dir()  # Streamlit Community Cloud checkout location


@st.cache_resource(show_spinner="Starting the research engine…")
def ensure_backend() -> str:
    """Start the API once per process; returns its base URL."""
    url = f"http://{HOST}:{PORT}"
    if _healthy(url):
        return url
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    os.environ.setdefault("EMBEDDING_PROVIDER", "fastembed")

    import uvicorn

    from app.main import create_app
    from app.scripts import seed_demo

    server = uvicorn.Server(uvicorn.Config(create_app(), host=HOST, port=PORT, log_level="warning"))
    threading.Thread(target=server.run, name="researchflow-api", daemon=True).start()
    deadline = time.monotonic() + 120
    while not _healthy(url):
        if time.monotonic() > deadline:
            raise RuntimeError("The embedded API did not start.")
        time.sleep(0.5)
    if os.environ.get("SEED_DEMO", "true").lower() in {"1", "true", "yes"}:
        threading.Thread(
            target=_seed, args=(seed_demo, url), name="researchflow-seed", daemon=True
        ).start()
    return url


def _seed(seed_demo, url: str) -> None:  # type: ignore[no-untyped-def]
    try:
        seed_demo.main(["--api", url, "--papers"])
    except Exception:  # a failed seed must not take the app down
        logger.exception("loading the demo documents failed")


def _healthy(url: str) -> bool:
    try:
        return httpx.get(f"{url}/health", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False
