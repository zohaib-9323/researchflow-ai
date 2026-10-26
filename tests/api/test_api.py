from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.container import build_container
from app.core.exceptions import ProviderUnavailableError
from app.llm.embeddings import HashingEmbeddings
from app.main import create_app
from tests.conftest import make_pdf, make_settings, sample_bytes
from tests.fakes import ScriptedLLM


def upload(client: TestClient, *files: tuple[str, bytes, str]) -> dict:
    response = client.post(
        "/documents/upload", files=[("files", (name, data, mime)) for name, data, mime in files]
    )
    assert response.status_code == 200, response.text
    return response.json()


def upload_samples(client: TestClient) -> None:
    body = upload(
        client,
        ("solar.txt", sample_bytes("larkspur_solar_pilot_report.txt"), "text/plain"),
        ("water.txt", sample_bytes("larkspur_water_quality.txt"), "text/plain"),
    )
    assert body["succeeded"] == 2


def test_health_reports_mode_without_secrets(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["llm_keys"]["total_keys"] == 1
    assert "test-key-1" not in response.text
    assert response.headers["X-Request-ID"]


def test_upload_list_get_and_delete(client: TestClient) -> None:
    pdf = make_pdf(["Page one text about turbines.", "Page two about 42 megawatts."])
    body = upload(
        client,
        ("turbines.pdf", pdf, "application/pdf"),
        ("notes.txt", b"Plain notes about wind farms.", "text/plain"),
    )
    assert body["succeeded"] == 2
    pdf_doc = body["results"][0]["document"]
    assert (pdf_doc["status"], pdf_doc["page_count"], pdf_doc["file_type"]) == ("indexed", 2, "pdf")
    # No internal fields (storage paths, hashes) are exposed.
    assert "storage_name" not in pdf_doc
    assert "content_hash" not in pdf_doc

    listed = client.get("/documents").json()
    assert {d["filename"] for d in listed} == {"turbines.pdf", "notes.txt"}
    assert client.get(f"/documents/{pdf_doc['id']}").json()["chunk_count"] >= 1

    assert client.delete(f"/documents/{pdf_doc['id']}").status_code == 204
    assert client.get(f"/documents/{pdf_doc['id']}").status_code == 404
    assert client.delete(f"/documents/{pdf_doc['id']}").status_code == 404
    assert {d["filename"] for d in client.get("/documents").json()} == {"notes.txt"}


def test_invalid_files_are_rejected_individually(client: TestClient) -> None:
    body = upload(
        client,
        ("evil.exe", b"MZ\x90\x00", "application/octet-stream"),
        ("fake.pdf", b"not really a pdf", "application/pdf"),
        ("scan.pdf", make_pdf(["", ""]), "application/pdf"),
        ("ok.txt", b"Valid text document.", "text/plain"),
    )
    by_name = {r["filename"]: r for r in body["results"]}
    assert by_name["evil.exe"]["error_code"] == "unsupported_file"
    assert by_name["fake.pdf"]["error_code"] == "unsupported_file"
    assert by_name["scan.pdf"]["error_code"] == "ocr_required"
    assert by_name["scan.pdf"]["document"]["status"] == "failed"
    assert by_name["ok.txt"]["ok"] is True
    assert (body["succeeded"], body["failed"]) == (1, 3)


def test_duplicate_upload_reported(client: TestClient) -> None:
    upload(client, ("a.txt", b"Same content here.", "text/plain"))
    body = upload(client, ("b.txt", b"Same content here.", "text/plain"))
    assert body["results"][0]["error_code"] == "duplicate_document"


def test_oversized_upload_rejected(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, max_upload_size_mb=0.001)
    container = build_container(settings, llm=ScriptedLLM(), embeddings=HashingEmbeddings())
    with TestClient(create_app(container)) as client:
        body = upload(client, ("big.txt", b"a" * 5000, "text/plain"))
    assert body["results"][0]["error_code"] == "file_too_large"


def test_research_validation(client: TestClient) -> None:
    response = client.post("/research", json={"question": "   "})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert client.post("/research", json={}).status_code == 422
    missing = client.post("/research", json={"question": "q", "conversation_id": "nope"})
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "not_found"


def test_research_conversation_and_report_flow(client: TestClient) -> None:
    upload_samples(client)
    response = client.post(
        "/research", json={"question": "How many wells exceeded the nitrate limit?"}
    )
    assert response.status_code == 200, response.text
    first = response.json()
    assert first["status"] == "answered"
    assert first["citations"][0]["document_name"] == "water.txt"
    assert first["execution"]["passed_evaluation"] is True
    conversation_id = first["conversation_id"]

    follow_up = client.post(
        "/research",
        json={"question": "What did the solar pilot cost?", "conversation_id": conversation_id},
    ).json()
    assert follow_up["conversation_id"] == conversation_id

    conversations = client.get("/conversations").json()
    assert conversations[0]["id"] == conversation_id
    assert conversations[0]["turn_count"] == 2
    detail = client.get(f"/conversations/{conversation_id}").json()
    assert [t["question"] for t in detail["turns"]] == [
        "How many wells exceeded the nitrate limit?",
        "What did the solar pilot cost?",
    ]

    report = client.post("/reports", json={"turn_id": first["turn_id"]})
    assert report.status_code == 201, report.text
    report_id = report.json()["id"]
    fetched = client.get(f"/reports/{report_id}").json()
    assert fetched["content"]["research_question"] == first["question"]
    assert fetched["content"]["sources"][0]["document_name"] == "water.txt"

    download = client.get(f"/reports/{report_id}/download")
    assert download.status_code == 200
    assert download.headers["content-type"].startswith("text/markdown")
    assert "attachment" in download.headers["content-disposition"]
    assert "## Evidence and citations" in download.text
    assert client.get("/reports/missing").status_code == 404


def test_conversation_cannot_switch_collection(client: TestClient) -> None:
    upload_samples(client)
    first = client.post("/research", json={"question": "How many wells were sampled?"}).json()
    other = client.post("/collections", json={"name": "Other"}).json()
    response = client.post(
        "/research",
        json={
            "question": "q",
            "conversation_id": first["conversation_id"],
            "collection_id": other["id"],
        },
    )
    assert response.status_code == 422


def test_report_requires_evidence(client: TestClient) -> None:
    turn = client.post("/research", json={"question": "Anything?"}).json()  # no documents
    assert turn["status"] == "insufficient_evidence"
    response = client.post("/reports", json={"turn_id": turn["turn_id"]})
    assert response.status_code == 422


def test_limited_mode_without_api_key(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, google_api_key_1=None)
    with TestClient(
        create_app(build_container(settings, embeddings=HashingEmbeddings()))
    ) as client:
        health = client.get("/health").json()
        assert health["llm_configured"] is False
        assert health["mode"].startswith("limited")
        # Non-LLM features still work.
        assert upload(client, ("a.txt", b"Some text to index.", "text/plain"))["succeeded"] == 1
        response = client.post("/research", json={"question": "What is in the text?"})
    assert response.status_code == 503
    error = response.json()["error"]
    assert error["code"] == "llm_not_configured"
    assert "GOOGLE_API_KEY_1" in error["message"]


def test_provider_unavailable_is_reported_not_faked(tmp_path: Path) -> None:
    llm = ScriptedLLM(classify=[ProviderUnavailableError()])
    container = build_container(make_settings(tmp_path), llm=llm, embeddings=HashingEmbeddings())
    with TestClient(create_app(container)) as client:
        upload_samples(client)
        body = client.post("/research", json={"question": "How many wells?"}).json()
    assert body["status"] == "error"
    assert body["error"]["code"] == "provider_unavailable"
    assert body["citations"] == []


def test_errors_do_not_leak_internals(client: TestClient) -> None:
    response = client.get("/documents/../../etc/passwd")
    assert response.status_code == 404
    assert "Traceback" not in response.text
    assert "/Users" not in response.text
    assert "/private" not in response.text


def test_openapi_docs_available(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    for path in (
        "/health",
        "/documents/upload",
        "/research",
        "/conversations",
        "/reports/{report_id}/download",
    ):
        assert path in schema["paths"]
