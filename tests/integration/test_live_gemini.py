"""Optional end-to-end smoke test against the real Gemini API.

Excluded from the default suite. Run with:

    pytest -m live

Requires at least one key in .env.local / the environment. Uses the hashing embedder
unless LIVE_EMBEDDINGS=huggingface is set, so it also runs where PyTorch is unavailable.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.config import PROJECT_ROOT, Settings
from app.container import build_container
from app.persistence.repositories import CollectionRepository
from app.schemas.research import ResearchRequest
from tests.conftest import SAMPLE_DOCS

pytestmark = pytest.mark.live


@pytest.fixture
def live_container(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.undo()  # undo the autouse env isolation for this test
    settings = Settings(
        _env_file=(PROJECT_ROOT / ".env", PROJECT_ROOT / ".env.local"),  # type: ignore[call-arg]
        data_dir=tmp_path,
        embedding_provider=os.environ.get("LIVE_EMBEDDINGS", "hashing"),
    )
    if not settings.llm_configured:
        pytest.skip("No Gemini API key configured")
    return build_container(settings)


def test_grounded_answer_and_honest_refusal(live_container) -> None:
    with live_container.database.session() as session:
        collection_id = CollectionRepository(session).get_or_create_default().id
    for path in sorted(SAMPLE_DOCS.glob("*.txt")):
        document = live_container.documents.upload(collection_id, path.name, path.read_bytes())
        assert document.status == "indexed"

    answer = live_container.research.ask(
        ResearchRequest(question="How many wells exceeded the nitrate limit, and where were they?")
    )
    assert answer.status in {"answered", "partial"}, answer.answer
    assert any(c.document_name == "larkspur_water_quality.txt" for c in answer.citations)
    retrieved = {e.chunk_id for e in answer.evidence}
    assert all(c.chunk_id in retrieved for c in answer.citations)

    unknown = live_container.research.ask(
        ResearchRequest(
            question="What was the average salary of the solar installers?",
            conversation_id=answer.conversation_id,
        )
    )
    assert unknown.status in {"insufficient_evidence", "partial", "not_applicable"}
    assert "salary" not in " ".join(c.excerpt.lower() for c in unknown.citations)
