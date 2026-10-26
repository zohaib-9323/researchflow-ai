from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.container import Container, build_container
from app.llm.embeddings import HashingEmbeddings
from app.main import create_app
from app.persistence.repositories import CollectionRepository
from tests.fakes import ScriptedLLM

SAMPLE_DOCS = Path(__file__).resolve().parent.parent / "evaluation" / "docs"


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The offline suite must never see real API keys or tracing settings."""
    for name in list(os.environ):
        if name.startswith(("GOOGLE_API_KEY", "GEMINI_API_KEY", "LANGSMITH", "LANGCHAIN")):
            monkeypatch.delenv(name, raising=False)


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "data_dir": tmp_path,
        "embedding_provider": "hashing",
        "chunk_size": 400,
        "chunk_overlap": 50,
        "retrieval_min_score": 0.0,
        "max_retries": 2,
        "google_api_key_1": "test-key-1",
        "log_level": "WARNING",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
def llm() -> ScriptedLLM:
    return ScriptedLLM()


@pytest.fixture
def container(settings: Settings, llm: ScriptedLLM) -> Container:
    return build_container(settings, llm=llm, embeddings=HashingEmbeddings())


@pytest.fixture
def collection_id(container: Container) -> str:
    with container.database.session() as session:
        return CollectionRepository(session).get_or_create_default().id


@pytest.fixture
def client(container: Container) -> Iterator[TestClient]:
    with TestClient(create_app(container)) as test_client:
        yield test_client


def sample_bytes(name: str) -> bytes:
    return (SAMPLE_DOCS / name).read_bytes()


def make_pdf(pages: list[str], password: str | None = None) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontsize=11)
    if password:
        data = doc.tobytes(
            encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw=password, owner_pw=password
        )
    else:
        data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def indexed_samples(container: Container, collection_id: str) -> dict[str, str]:
    """Ingest the three sample documents; returns filename -> document id."""
    ids = {}
    for name in (
        "larkspur_solar_pilot_report.txt",
        "larkspur_council_memo.txt",
        "larkspur_water_quality.txt",
    ):
        document = container.documents.upload(collection_id, name, sample_bytes(name))
        assert document.status == "indexed", document.error_message
        ids[name] = document.id
    return ids
