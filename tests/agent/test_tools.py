from __future__ import annotations

import json
import time

import httpx
import pytest
from langchain_core.documents import Document

from app.agent import tools as tools_module
from app.agent.schemas import ComparisonDraft, ComparisonPoint
from app.agent.tools import (
    TavilySearch,
    ToolContext,
    ToolExecutor,
    ToolFailedError,
    WebResult,
    build_tools,
    safe_calculate,
    tools_for,
)
from app.core.exceptions import ProviderUnavailableError
from tests.agent.fakes import FakeWebSearch


def _doc(chunk: str, text: str, page: int = 1) -> Document:
    return Document(
        page_content=text,
        metadata={
            "chunk_id": chunk,
            "document_id": "doc1",
            "document_name": "report.txt",
            "page": page,
            "score": 0.8,
        },
    )


def _ctx(docs: list[Document] | None = None, **kwargs: object) -> ToolContext:
    defaults: dict[str, object] = {
        "collection_id": "c1",
        "task_id": "T1",
        "sources": {},
        "max_sources": 10,
        "retrieve": lambda _cid, _q, _k: list(docs or []),
        "compare": lambda aspect, sources: ComparisonDraft(points=[], summary=""),
    }
    defaults.update(kwargs)
    return ToolContext(**defaults)  # type: ignore[arg-type]


executor = ToolExecutor(timeout_seconds=2)


# -- calculator ------------------------------------------------------------------------------
def test_safe_calculate_arithmetic() -> None:
    assert safe_calculate("(18 - 12) / 18 * 100") == pytest.approx(33.333, rel=1e-3)
    assert safe_calculate("round(sqrt(16) + max(1, 2), 1)") == 6.0


@pytest.mark.parametrize(
    "expression",
    ["__import__('os').system('ls')", "x + 1", "'a' * 3", "(1).real", "9 ** 9 ** 9", "1 / 0", ""],
)
def test_safe_calculate_rejects_code_and_bad_input(expression: str) -> None:
    with pytest.raises(ToolFailedError):
        safe_calculate(expression)


# -- argument validation and execution -------------------------------------------------------
def test_search_documents_registers_sources_with_metadata() -> None:
    ctx = _ctx([_doc("a", "Savings were 18 percent."), _doc("b", "Costs were high.", 2)])
    record, output = executor.run(build_tools(ctx), "search_documents", {"query": "savings"}, "T1")
    assert record.ok
    assert output.result_count == 2
    assert output.source_ids == ["D1", "D2"]
    source = ctx.sources["D2"]
    assert (source.page, source.chunk_id, source.document_id) == (2, "b", "doc1")
    assert source.provenance == "search_documents('savings')"
    assert source.task_id == "T1"


def test_search_dedupes_and_caps_sources() -> None:
    ctx = _ctx([_doc("a", "one"), _doc("a", "one"), _doc("b", "two")], max_sources=1)
    _, output = executor.run(build_tools(ctx), "search_documents", {"query": "x y"}, "T1")
    assert output.source_ids == ["D1"]
    assert list(ctx.sources) == ["D1"]


def test_empty_search_is_ok_with_zero_results() -> None:
    record, output = executor.run(build_tools(_ctx([])), "search_documents", {"query": "x y"}, "T1")
    assert record.ok
    assert output.result_count == 0
    assert "No passages" in output.summary


@pytest.mark.parametrize(
    "args",
    [{}, {"query": "a"}, {"query": "valid", "k": 50}, {"query": 123}, {"query": "x" * 400}],
)
def test_invalid_arguments_are_rejected_before_running(args: dict) -> None:
    calls: list[str] = []
    ctx = _ctx(retrieve=lambda _c, q, _k: calls.append(q) or [])
    record, output = executor.run(build_tools(ctx), "search_documents", args, "T1")
    assert not record.ok
    assert "Invalid arguments" in (output.error or "")
    assert calls == []


def test_unknown_or_unexposed_tool_is_refused() -> None:
    tools = build_tools(_ctx())
    record, _ = executor.run(
        {"calculate": tools["calculate"]}, "search_web", {"query": "x y"}, "T1"
    )
    assert not record.ok
    assert "not available" in record.summary


def test_approval_required_tools_are_not_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tools_module, "REQUIRES_APPROVAL", frozenset({"calculate"}))
    record, _ = executor.run(build_tools(_ctx()), "calculate", {"expression": "1+1"}, "T1")
    assert not record.ok
    assert "approval" in record.summary


def test_tool_timeout_becomes_failed_result() -> None:
    def slow(_c: str, _q: str, _k: int) -> list[Document]:
        time.sleep(1.0)
        return []

    record, _ = ToolExecutor(timeout_seconds=0.1).run(
        build_tools(_ctx(retrieve=slow)), "search_documents", {"query": "slow one"}, "T1"
    )
    assert not record.ok
    assert "timed out" in record.summary


def test_unexpected_tool_exception_is_contained() -> None:
    def broken(_c: str, _q: str, _k: int) -> list[Document]:
        raise RuntimeError("secret internal detail")

    record, _ = executor.run(
        build_tools(_ctx(retrieve=broken)), "search_documents", {"query": "x y"}, "T1"
    )
    assert not record.ok
    assert "secret" not in record.summary


def test_llm_provider_failure_propagates() -> None:
    def failing(_aspect: str, _sources: list) -> ComparisonDraft:
        raise ProviderUnavailableError()

    ctx = _ctx([_doc("a", "one two"), _doc("b", "three four")], compare=failing)
    tools = build_tools(ctx)
    executor.run(tools, "search_documents", {"query": "x y"}, "T1")
    with pytest.raises(ProviderUnavailableError):
        executor.run(tools, "compare_sources", {"source_ids": ["D1", "D2"], "aspect": "cost"}, "T1")


def test_inspect_and_compare_validate_source_ids() -> None:
    def compare(aspect: str, sources: list) -> ComparisonDraft:
        return ComparisonDraft(
            points=[
                ComparisonPoint(point="ok", source_ids=["D1", "D2"], relation="disagree"),
                ComparisonPoint(point="invented", source_ids=["D9"], relation="agree"),
            ],
            summary="s",
        )

    ctx = _ctx([_doc("a", "one two"), _doc("b", "three four")], compare=compare)
    tools = build_tools(ctx)
    executor.run(tools, "search_documents", {"query": "x y"}, "T1")
    bad, _ = executor.run(tools, "inspect_source", {"source_id": "D7"}, "T1")
    assert not bad.ok
    pattern, _ = executor.run(tools, "inspect_source", {"source_id": "../etc"}, "T1")
    assert not pattern.ok
    assert "Invalid arguments" in pattern.summary
    good, output = executor.run(tools, "inspect_source", {"source_id": "D1"}, "T1")
    assert good.ok
    assert output.data["text"] == "one two"
    _, compared = executor.run(
        tools, "compare_sources", {"source_ids": ["D1", "D2"], "aspect": "cost"}, "T1"
    )
    assert [p["point"] for p in compared.data["points"]] == ["ok"]  # D9 point dropped


def test_calculate_tool() -> None:
    record, output = executor.run(build_tools(_ctx()), "calculate", {"expression": "18-12"}, "T1")
    assert record.ok
    assert output.data == {"value": 6.0}
    bad, _ = executor.run(build_tools(_ctx()), "calculate", {"expression": "open('x')"}, "T1")
    assert not bad.ok


# -- web search ------------------------------------------------------------------------------
def test_tools_for_hides_web_search_when_not_configured() -> None:
    assert "search_web" not in tools_for("web_search", web_available=False)
    assert "search_documents" in tools_for("web_search", web_available=False)
    assert "search_web" in tools_for("web_search", web_available=True)
    assert "create_research_report" not in tools_for("mixed", web_available=True)


def test_search_web_without_provider_fails_honestly() -> None:
    record, _ = executor.run(build_tools(_ctx()), "search_web", {"query": "solar news"}, "T1")
    assert not record.ok
    assert "not configured" in record.summary


def test_search_web_with_provider_creates_w_sources() -> None:
    web = FakeWebSearch([WebResult(title="Page", url="https://example.org/a", content="Text")])
    ctx = _ctx(web_search=web)
    _, output = executor.run(build_tools(ctx), "search_web", {"query": "solar news"}, "T1")
    assert output.source_ids == ["W1"]
    assert ctx.sources["W1"].url == "https://example.org/a"
    assert "fake-web" in ctx.sources["W1"].provenance


def _tavily(handler: httpx.MockTransport) -> TavilySearch:
    return TavilySearch("tvly-test", client=httpx.Client(transport=handler))


def test_tavily_parses_results_and_drops_non_http_urls() -> None:
    seen: dict = {}

    def handle(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "results": [
                    {"title": "A", "url": "https://a.example", "content": "alpha", "score": 0.9},
                    {"title": "B", "url": "javascript:alert(1)", "content": "bad"},
                ]
            },
        )

    results = _tavily(httpx.MockTransport(handle)).search("solar", 5)
    assert [r.url for r in results] == ["https://a.example"]
    assert seen["auth"] == "Bearer tvly-test"
    assert seen["body"]["query"] == "solar"


@pytest.mark.parametrize("status", [401, 429, 500])
def test_tavily_http_errors_become_tool_failures(status: int) -> None:
    provider = _tavily(httpx.MockTransport(lambda _r: httpx.Response(status)))
    with pytest.raises(ToolFailedError, match=str(status)):
        provider.search("solar", 5)


def test_tavily_timeout_becomes_tool_failure() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(ToolFailedError, match="timed out"):
        _tavily(httpx.MockTransport(handle)).search("solar", 5)
