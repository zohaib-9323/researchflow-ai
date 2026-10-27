"""Sign-in, free credits, admin access and per-user isolation (AUTH_ENABLED=true)."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.container import Container, build_container
from app.llm.embeddings import HashingEmbeddings
from app.main import create_app
from app.persistence.repositories import CollectionRepository
from app.services.user_service import SYSTEM_EMAIL
from tests.agent.fakes import ScriptedAgentLLM
from tests.conftest import make_settings, sample_bytes
from tests.fakes import ScriptedLLM

TOKEN = "internal-test-token"  # noqa: S105 (test value)
ADMIN = "admin@example.com"


def headers(email: str | None, token: str | None = TOKEN) -> dict[str, str]:
    values = {}
    if token:
        values["X-Internal-Token"] = token
    if email:
        values["X-User-Email"] = email
    return values


@pytest.fixture
def auth_container(tmp_path: Path) -> Container:
    container = build_container(
        make_settings(
            tmp_path,
            auth_enabled=True,
            internal_api_token=TOKEN,
            admin_emails=f"{ADMIN.upper()}, other-admin@example.com",
            free_questions=2,
            free_agent_runs=1,
        ),
        llm=ScriptedLLM(),
        agent_llm=ScriptedAgentLLM(),
        embeddings=HashingEmbeddings(),
        agent_background=False,
    )
    with container.database.session() as session:
        collection_id = CollectionRepository(session).get_or_create_default().id
    for name in ("larkspur_solar_pilot_report.txt", "larkspur_council_memo.txt"):
        container.documents.upload(collection_id, name, sample_bytes(name))
    return container


@pytest.fixture
def api(auth_container: Container) -> Iterator[TestClient]:
    with TestClient(create_app(auth_container)) as client:
        yield client


QUESTION = {"question": "How many rooftop solar systems did the pilot install?"}
OBJECTIVE = {"objective": "How many rooftop solar systems did the pilot install?"}


def test_auth_enabled_requires_internal_token(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="INTERNAL_API_TOKEN"):
        make_settings(tmp_path, auth_enabled=True)


@pytest.mark.parametrize(
    "sent",
    [
        {},
        headers("user@example.com", token=None),
        headers("user@example.com", "wrong"),
        headers(None),
        headers("not-an-email"),
    ],
)
def test_requests_without_valid_identity_are_rejected(api: TestClient, sent: dict) -> None:
    assert api.get("/documents", headers=sent).status_code == 401
    assert api.post("/research", json=QUESTION, headers=sent).status_code == 401
    assert api.get("/health").status_code == 200  # health stays public


def test_free_question_credits(api: TestClient) -> None:
    user = headers("Alice@Example.com")
    me = api.get("/me", headers=user).json()
    assert me["email"] == "alice@example.com"
    assert not me["is_admin"]
    assert me["credits"]["question"] == {"limit": 2, "used": 0, "remaining": 2}

    for _ in range(2):
        assert api.post("/research", json=QUESTION, headers=user).status_code == 200
    blocked = api.post("/research", json=QUESTION, headers=user)
    assert blocked.status_code == 429
    assert blocked.json()["error"]["code"] == "credits_exhausted"
    assert api.get("/me", headers=user).json()["credits"]["question"]["remaining"] == 0
    # another user has their own credits
    assert (
        api.post("/research", json=QUESTION, headers=headers("bob@example.com")).status_code == 200
    )


def test_failed_question_is_refunded(auth_container: Container, api: TestClient) -> None:
    from app.core.exceptions import ProviderUnavailableError

    llm = auth_container.llm
    assert isinstance(llm, ScriptedLLM)
    llm.script["classify"] = deque([ProviderUnavailableError()])
    user = headers("carol@example.com")
    api.post("/research", json=QUESTION, headers=user)
    assert api.get("/me", headers=user).json()["credits"]["question"]["used"] == 0


def test_free_agent_run_credits_and_free_resume(api: TestClient) -> None:
    user = headers("dave@example.com")
    first = api.post("/agent/runs", json=OBJECTIVE, headers=user)
    assert first.status_code == 202
    assert api.post("/agent/runs", json=OBJECTIVE, headers=user).status_code == 429
    me = api.get("/me", headers=user).json()
    assert me["credits"]["agent_run"] == {"limit": 1, "used": 1, "remaining": 0}


def test_invalid_agent_request_does_not_use_a_credit(api: TestClient) -> None:
    user = headers("erin@example.com")
    bad = api.post("/agent/runs", json={**OBJECTIVE, "collection_id": "missing"}, headers=user)
    assert bad.status_code == 404
    assert api.get("/me", headers=user).json()["credits"]["agent_run"]["used"] == 0


def test_admin_is_unlimited(api: TestClient) -> None:
    admin = headers(ADMIN)
    me = api.get("/me", headers=admin).json()
    assert me["is_admin"]
    assert me["credits"]["question"]["remaining"] is None
    for _ in range(4):
        assert api.post("/research", json=QUESTION, headers=admin).status_code == 200
    for _ in range(3):
        assert api.post("/agent/runs", json=OBJECTIVE, headers=admin).status_code == 202


def test_only_admins_change_the_library(api: TestClient) -> None:
    files = {"files": ("notes.txt", b"Some research notes about solar panels.", "text/plain")}
    user = headers("frank@example.com")
    assert api.post("/documents/upload", files=files, headers=user).status_code == 403
    assert api.post("/collections", json={"name": "Mine"}, headers=user).status_code == 403
    doc_id = api.get("/documents", headers=user).json()[0]["id"]
    assert api.delete(f"/documents/{doc_id}", headers=user).status_code == 403
    assert api.post("/documents/upload", files=files, headers=headers(ADMIN)).status_code == 200
    # the demo seeder's identity is an administrator too
    assert (
        api.post("/collections", json={"name": "Seeded"}, headers=headers(SYSTEM_EMAIL)).status_code
        == 201
    )


def test_history_and_runs_are_private(api: TestClient) -> None:
    alice, bob = headers("alice@example.com"), headers("bob@example.com")
    answer = api.post("/research", json=QUESTION, headers=alice).json()
    run = api.post("/agent/runs", json=OBJECTIVE, headers=alice).json()

    assert [c["id"] for c in api.get("/conversations", headers=alice).json()] == [
        answer["conversation_id"]
    ]
    assert api.get("/conversations", headers=bob).json() == []
    assert api.get(f"/conversations/{answer['conversation_id']}", headers=bob).status_code == 404
    follow_up = {**QUESTION, "conversation_id": answer["conversation_id"]}
    assert api.post("/research", json=follow_up, headers=bob).status_code == 404
    assert api.post("/reports", json={"turn_id": answer["turn_id"]}, headers=bob).status_code == 404

    assert [r["id"] for r in api.get("/agent/runs", headers=alice).json()] == [run["id"]]
    assert api.get("/agent/runs", headers=bob).json() == []
    for path in ("", "/report", "/report/download"):
        assert api.get(f"/agent/runs/{run['id']}{path}", headers=bob).status_code == 404
    assert api.post(f"/agent/runs/{run['id']}/resume", headers=bob).status_code == 404

    # the administrator sees everyone's work
    assert len(api.get("/conversations", headers=headers(ADMIN)).json()) == 1
    assert api.get(f"/agent/runs/{run['id']}", headers=headers(ADMIN)).status_code == 200


def test_sign_in_disabled_means_no_limits(client: TestClient) -> None:
    me = client.get("/me").json()
    assert me == {"auth_enabled": False, "email": None, "is_admin": True, "credits": None}
