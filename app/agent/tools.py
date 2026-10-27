"""Research tools: real LangChain tools backed by application functions.

Every tool has a Pydantic argument schema (validated before execution), runs a real
application function (vector search, a web-search API, a database lookup, an LLM
comparison or a restricted calculator) and returns a `ToolOutput`. Nothing here returns
canned results.

Safety model:
* tools are read-only; none can write files, run shell commands or contact anything but
  the configured search provider and the Gemini API;
* each tool declares `requires_approval`; the executor refuses tools that need human
  approval (none of the current tools do) — the hook for future side-effecting tools;
* tool output is untrusted data and is only ever shown to the model inside
  <tool_result> tags with explicit instructions not to follow it;
* arguments are validated, execution is time-boxed, and failures become `ok=False`
  results instead of crashing the run.
"""

from __future__ import annotations

import ast
import math
import operator
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx
from langchain_core.documents import Document
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, ValidationError

from app.agent.schemas import (
    SOURCE_ID_PATTERN,
    Approach,
    ComparisonDraft,
    SourceRecord,
    ToolCallRecord,
)
from app.core.exceptions import ProviderError, ResearchFlowError
from app.core.logging import get_logger

logger = get_logger(__name__)

MAX_SOURCE_CHARS = 2000  # stored per source
MAX_RESULT_TEXT = 700  # per result shown back to the model


class ToolFailedError(Exception):
    """A tool could not produce a result (bad input, provider failure, ...)."""


# -- web search provider ---------------------------------------------------------------------
class WebResult(BaseModel):
    title: str
    url: str
    content: str
    score: float | None = None


class WebSearchProvider(Protocol):
    name: str

    def search(self, query: str, max_results: int) -> list[WebResult]: ...


class TavilySearch:
    """Web search through the Tavily API (https://tavily.com)."""

    name = "tavily"
    endpoint = "https://api.tavily.com/search"

    def __init__(
        self, api_key: str, timeout: float = 20.0, client: httpx.Client | None = None
    ) -> None:
        self._api_key = api_key
        self._timeout = timeout
        self._client = client

    def search(self, query: str, max_results: int) -> list[WebResult]:
        payload = {"query": query, "max_results": max_results, "search_depth": "basic"}
        headers = {"Authorization": f"Bearer {self._api_key}"}
        try:
            if self._client is not None:
                response = self._client.post(
                    self.endpoint, json=payload, headers=headers, timeout=self._timeout
                )
            else:
                response = httpx.post(
                    self.endpoint, json=payload, headers=headers, timeout=self._timeout
                )
            response.raise_for_status()
            body = response.json()
        except httpx.TimeoutException as exc:
            raise ToolFailedError("The web search provider timed out.") from exc
        except httpx.HTTPStatusError as exc:
            raise ToolFailedError(
                f"The web search provider returned HTTP {exc.response.status_code}."
            ) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise ToolFailedError("The web search provider could not be reached.") from exc
        results = []
        for item in body.get("results", [])[:max_results]:
            url = str(item.get("url") or "")
            if not url.startswith(("http://", "https://")):
                continue  # never keep a result without a real URL
            results.append(
                WebResult(
                    title=str(item.get("title") or url)[:300],
                    url=url,
                    content=str(item.get("content") or "")[:MAX_SOURCE_CHARS],
                    score=item.get("score"),
                )
            )
        return results


# -- safe calculator -------------------------------------------------------------------------
_BINARY: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY: dict[type, Callable[[Any], Any]] = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
}


def safe_calculate(expression: str) -> float:
    """Evaluate arithmetic only: numbers, + - * / // % **, parentheses and a few
    functions. Names, attributes, strings, comprehensions and calls to anything else
    are rejected, so this cannot execute code."""

    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            if isinstance(node.value, bool):
                raise ToolFailedError("Booleans are not numbers here.")
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Pow) and (abs(right) > 10 or abs(left) > 1e6):
                raise ToolFailedError("Exponent too large.")
            return _BINARY[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](evaluate(node.operand))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCTIONS
            and not node.keywords
        ):
            args = [evaluate(arg) for arg in node.args]
            if node.func.id == "round" and len(args) == 2:
                args[1] = int(args[1])  # ndigits must be an integer
            return _FUNCTIONS[node.func.id](*args)
        raise ToolFailedError(f"Unsupported expression element: {type(node).__name__}.")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolFailedError("The expression is not valid arithmetic.") from exc
    try:
        result = evaluate(tree)
    except ZeroDivisionError as exc:
        raise ToolFailedError("Division by zero.") from exc
    except (OverflowError, ValueError, TypeError) as exc:
        raise ToolFailedError(f"Calculation failed: {exc}.") from exc
    if isinstance(result, float) and not math.isfinite(result):
        raise ToolFailedError("The result is not a finite number.")
    return float(result)


# -- argument schemas ------------------------------------------------------------------------
class SearchDocumentsArgs(BaseModel):
    query: str = Field(min_length=2, max_length=300, description="Semantic search query.")
    k: int = Field(default=5, ge=1, le=10, description="Number of passages to return.")


class SearchWebArgs(BaseModel):
    query: str = Field(min_length=2, max_length=300, description="Web search query.")
    max_results: int = Field(default=5, ge=1, le=10)


class InspectSourceArgs(BaseModel):
    source_id: str = Field(pattern=SOURCE_ID_PATTERN, description="A source id such as D3.")


class CompareSourcesArgs(BaseModel):
    source_ids: list[str] = Field(min_length=2, max_length=8, description="Sources to compare.")
    aspect: str = Field(min_length=3, max_length=200, description="What to compare.")


class CalculateArgs(BaseModel):
    expression: str = Field(
        min_length=1, max_length=200, description="Arithmetic expression, e.g. (18-12)/18*100."
    )


class CreateReportArgs(BaseModel):
    objective: str = Field(min_length=3, max_length=4000)


class ToolOutput(BaseModel):
    ok: bool
    summary: str
    result_count: int = 0
    source_ids: list[str] = Field(default_factory=list)
    data: Any = None
    error: str | None = None


# -- execution context -----------------------------------------------------------------------
@dataclass
class ToolContext:
    """Everything a tool may touch during one task of one run."""

    collection_id: str
    task_id: str
    sources: dict[str, SourceRecord]
    max_sources: int
    retrieve: Callable[[str, str, int], list[Document]]
    compare: Callable[[str, list[SourceRecord]], ComparisonDraft]
    write_report: Callable[[str], Any] | None = None
    web_search: WebSearchProvider | None = None
    document_info: Callable[[str], dict[str, Any] | None] = lambda _doc_id: None
    added: list[str] = field(default_factory=list)

    def _next_id(self, prefix: str) -> str:
        numbers = [int(sid[1:]) for sid in self.sources if sid.startswith(prefix)]
        return f"{prefix}{max(numbers, default=0) + 1}"

    def add_source(self, record: dict[str, Any]) -> SourceRecord | None:
        """Register a result with an application-assigned id; dedupes and caps."""
        for existing in self.sources.values():
            same_chunk = record.get("chunk_id") and existing.chunk_id == record.get("chunk_id")
            same_url = record.get("url") and existing.url == record.get("url")
            if same_chunk or same_url:
                return existing
        if len(self.sources) >= self.max_sources:
            return None
        prefix = "D" if record["kind"] == "document" else "W"
        source = SourceRecord(id=self._next_id(prefix), task_id=self.task_id, **record)
        self.sources[source.id] = source
        self.added.append(source.id)
        return source


def _preview(source: SourceRecord) -> dict[str, Any]:
    return {
        "source_id": source.id,
        "title": source.title,
        "page": source.page,
        "url": source.url,
        "text": source.text[:MAX_RESULT_TEXT],
    }


def build_tools(ctx: ToolContext) -> dict[str, BaseTool]:
    """Create the LangChain tools bound to this context."""

    def search_documents(query: str, k: int = 5) -> ToolOutput:
        documents = ctx.retrieve(ctx.collection_id, query, k)
        found = []
        for doc in documents:
            meta = doc.metadata
            source = ctx.add_source(
                {
                    "kind": "document",
                    "title": meta["document_name"],
                    "document_id": meta["document_id"],
                    "chunk_id": meta["chunk_id"],
                    "page": meta.get("page"),
                    "text": doc.page_content[:MAX_SOURCE_CHARS],
                    "score": meta.get("score"),
                    "provenance": f"search_documents({query!r})",
                }
            )
            if source is not None and source not in found:
                found.append(source)
        if not found:
            return ToolOutput(ok=True, summary=f"No passages matched '{query}'.", data=[])
        return ToolOutput(
            ok=True,
            summary=f"Found {len(found)} passage(s) for '{query}'.",
            result_count=len(found),
            source_ids=[s.id for s in found],
            data=[_preview(s) for s in found],
        )

    def search_web(query: str, max_results: int = 5) -> ToolOutput:
        if ctx.web_search is None:
            raise ToolFailedError("Web search is not configured for this deployment.")
        results = ctx.web_search.search(query, max_results)
        found = []
        for result in results:
            source = ctx.add_source(
                {
                    "kind": "web",
                    "title": result.title,
                    "url": result.url,
                    "text": result.content,
                    "score": result.score,
                    "provenance": f"search_web({query!r}) via {ctx.web_search.name}",
                }
            )
            if source is not None and source not in found:
                found.append(source)
        if not found:
            return ToolOutput(ok=True, summary=f"No web results for '{query}'.", data=[])
        return ToolOutput(
            ok=True,
            summary=f"Found {len(found)} web result(s) for '{query}'.",
            result_count=len(found),
            source_ids=[s.id for s in found],
            data=[_preview(s) for s in found],
        )

    def inspect_source(source_id: str) -> ToolOutput:
        source = ctx.sources.get(source_id)
        if source is None:
            raise ToolFailedError(f"Unknown source id '{source_id}'. Use ids returned by a search.")
        details: dict[str, Any] = source.model_dump(exclude={"text"})
        details["text"] = source.text
        if source.document_id and (info := ctx.document_info(source.document_id)):
            details["document"] = info
        return ToolOutput(
            ok=True,
            summary=f"Inspected {source.id} ({source.title}).",
            result_count=1,
            source_ids=[source.id],
            data=details,
        )

    def compare_sources(source_ids: list[str], aspect: str) -> ToolOutput:
        unknown = [sid for sid in source_ids if sid not in ctx.sources]
        if unknown:
            raise ToolFailedError(f"Unknown source id(s): {', '.join(unknown)}.")
        selected = [ctx.sources[sid] for sid in dict.fromkeys(source_ids)]
        draft = ctx.compare(aspect, selected)
        allowed = {s.id for s in selected}
        points = [
            p.model_dump()
            for p in draft.points
            if p.source_ids and set(p.source_ids) <= allowed  # drop points citing others
        ]
        return ToolOutput(
            ok=True,
            summary=f"Compared {len(selected)} sources on '{aspect}': {len(points)} point(s).",
            result_count=len(points),
            source_ids=sorted(allowed),
            data={"points": points, "summary": draft.summary},
        )

    def calculate(expression: str) -> ToolOutput:
        value = safe_calculate(expression)
        return ToolOutput(
            ok=True, summary=f"{expression} = {value:g}", result_count=1, data={"value": value}
        )

    def create_research_report(objective: str) -> ToolOutput:
        if ctx.write_report is None:
            raise ToolFailedError("Report generation is only available after verification.")
        report = ctx.write_report(objective)
        return ToolOutput(ok=True, summary="Report generated.", result_count=1, data=report)

    specs: list[tuple[Callable[..., ToolOutput], type[BaseModel], str]] = [
        (
            search_documents,
            SearchDocumentsArgs,
            "Search the user's selected document collection by meaning. Returns passages "
            "with source ids (D1, D2, ...).",
        ),
        (
            search_web,
            SearchWebArgs,
            "Search the public web. Returns pages with source ids (W1, ...) and real URLs.",
        ),
        (
            inspect_source,
            InspectSourceArgs,
            "Get the full text and metadata of one source returned by a search.",
        ),
        (
            compare_sources,
            CompareSourcesArgs,
            "Compare what 2-8 retrieved sources say about one aspect.",
        ),
        (calculate, CalculateArgs, "Evaluate an arithmetic expression (no code)."),
        (
            create_research_report,
            CreateReportArgs,
            "Write the final structured report from verified findings.",
        ),
    ]
    return {
        fn.__name__: StructuredTool.from_function(
            func=fn, name=fn.__name__, description=desc, args_schema=schema
        )
        for fn, schema, desc in specs
    }


# Tools that need human approval before running. All current tools are read-only; a
# future tool that sends email or publishes content must be listed here.
REQUIRES_APPROVAL: frozenset[str] = frozenset()

_APPROACH_TOOLS: dict[str, list[str]] = {
    "document_search": ["search_documents", "inspect_source"],
    "web_search": ["search_web", "inspect_source"],
    "comparison": ["search_documents", "search_web", "compare_sources", "inspect_source"],
    "calculation": ["calculate", "search_documents"],
    "mixed": ["search_documents", "search_web", "inspect_source", "compare_sources", "calculate"],
}


def tools_for(approach: Approach, web_available: bool) -> list[str]:
    """Only the tools a task needs; web search only when it is really configured."""
    names = list(_APPROACH_TOOLS[approach])
    if not web_available:
        names = [n for n in names if n != "search_web"]
        if "search_documents" not in names:
            names.insert(0, "search_documents")
    return names


class ToolExecutor:
    """Validates, time-boxes and records every tool call."""

    def __init__(self, timeout_seconds: float) -> None:
        self.timeout_seconds = timeout_seconds

    def run(
        self, tools: dict[str, BaseTool], name: str, arguments: dict[str, Any], task_id: str
    ) -> tuple[ToolCallRecord, ToolOutput]:
        started = time.monotonic()
        output = self._execute(tools, name, arguments)
        record = ToolCallRecord(
            task_id=task_id,
            tool=name,
            arguments=_safe_arguments(arguments),
            ok=output.ok,
            summary=output.summary,
            result_count=output.result_count,
            source_ids=output.source_ids,
            error=output.error,
            duration_ms=int((time.monotonic() - started) * 1000),
        )
        logger.info(
            "agent_tool_call",
            extra={
                "tool": name,
                "ok": output.ok,
                "result_count": output.result_count,
                "duration_ms": record.duration_ms,
                "task_id": task_id,
            },
        )
        return record, output

    def _execute(
        self, tools: dict[str, BaseTool], name: str, arguments: dict[str, Any]
    ) -> ToolOutput:
        tool = tools.get(name)
        if tool is None:
            return _failed(f"Tool '{name}' is not available for this task.")
        if name in REQUIRES_APPROVAL:
            return _failed(f"Tool '{name}' requires human approval and was not run.")
        schema = tool.args_schema
        try:
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                schema.model_validate(arguments)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
            )
            return _failed(f"Invalid arguments for {name}: {problems}")
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"tool-{name}")
        future = pool.submit(tool.invoke, arguments)
        try:
            result = future.result(timeout=self.timeout_seconds)
        except FutureTimeout:
            return _failed(f"{name} timed out after {self.timeout_seconds:g} s.")
        except ToolFailedError as exc:
            return _failed(str(exc))
        except ValidationError as exc:
            return _failed(f"Invalid arguments for {name}: {exc.error_count()} problem(s).")
        except ProviderError:
            raise  # a Gemini outage is a run-level error, handled by the graph
        except ResearchFlowError as exc:  # e.g. index unavailable: report it to the agent
            return _failed(f"{name} failed: {exc.message}")
        except Exception as exc:  # unexpected errors must not crash the run
            logger.warning(
                "agent_tool_error", extra={"tool": name, "error_type": type(exc).__name__}
            )
            return _failed(f"{name} failed ({type(exc).__name__}).")
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        return result if isinstance(result, ToolOutput) else ToolOutput.model_validate(result)


def _failed(message: str) -> ToolOutput:
    return ToolOutput(ok=False, summary=message, error=message)


def _safe_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Arguments as recorded in state/logs: short strings only."""
    return {key: (str(value)[:200]) for key, value in arguments.items()}
