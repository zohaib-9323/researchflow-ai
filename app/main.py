"""FastAPI application factory."""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from app.api.routes import agent, conversations, documents, reports, research
from app.config import Settings, get_settings
from app.container import Container, LazyEmbeddings, build_container
from app.core.exceptions import ResearchFlowError
from app.core.logging import configure_logging, get_logger, request_id_var

logger = get_logger(__name__)


def configure_tracing(settings: Settings) -> None:
    """LangSmith tracing is opt-in: enabled only when explicitly configured."""
    if settings.langsmith_tracing and settings.langsmith_api_key is not None:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key.get_secret_value()
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    else:
        os.environ["LANGSMITH_TRACING"] = "false"


def _error(status: int, code: str, message: str, details: dict | None = None) -> JSONResponse:
    body: dict = {"error": {"code": code, "message": message, "request_id": request_id_var.get()}}
    if details:
        body["error"]["details"] = details
    return JSONResponse(status_code=status, content=body)


class HealthResponse(BaseModel):
    status: str
    mode: str
    llm_configured: bool
    llm_model: str
    llm_fallback_models: list[str]
    llm_keys: dict
    embedding_provider: str
    embedding_model: str
    max_retries: int
    web_search_configured: bool


def create_app(container: Container | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or (container.settings if container else get_settings())
    configure_logging(settings.log_level)
    configure_tracing(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if getattr(app.state, "container", None) is None:
            app.state.container = build_container(settings)
        built: Container = app.state.container
        if not built.settings.llm_configured:
            logger.warning("llm_not_configured_limited_mode")
        logger.info(
            "api_started",
            extra={"llm_keys": len(built.key_pool), "model": built.settings.chat_model},
        )
        yield

    app = FastAPI(
        title="ResearchFlow AI API",
        version="0.1.0",
        description=(
            "Document research assistant: upload PDF/TXT files, ask grounded questions "
            "with verified citations, and generate research reports."
        ),
        lifespan=lifespan,
    )
    app.state.container = container
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        incoming = request.headers.get("x-request-id", "")
        request_id = incoming if incoming.isalnum() and len(incoming) <= 64 else uuid.uuid4().hex
        token = request_id_var.set(request_id)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(ResearchFlowError)
    async def domain_error(_request: Request, exc: ResearchFlowError) -> JSONResponse:
        return _error(exc.status_code, exc.code, exc.message, exc.details or None)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        problems = [
            {"field": ".".join(str(p) for p in err["loc"] if p != "body"), "message": err["msg"]}
            for err in exc.errors()
        ]
        return _error(422, "validation_error", "The request is invalid.", {"problems": problems})

    @app.exception_handler(Exception)
    async def unexpected_error(_request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_error", extra={"error_type": type(exc).__name__})
        return _error(500, "internal_error", "An unexpected error occurred.")

    @app.get("/health", response_model=HealthResponse, tags=["health"])
    def health(request: Request) -> HealthResponse:
        built: Container = request.app.state.container
        cfg = built.settings
        return HealthResponse(
            status="ok",
            mode="full" if cfg.llm_configured else "limited (no Gemini API key configured)",
            llm_configured=cfg.llm_configured,
            llm_model=cfg.chat_model,
            llm_fallback_models=cfg.chat_models[1:],
            llm_keys=built.key_pool.status(),
            embedding_provider=cfg.embedding_provider,
            embedding_model=cfg.embedding_model
            if cfg.embedding_provider == "huggingface"
            else "hashing (development only)",
            max_retries=cfg.max_retries,
            web_search_configured=built.agent.web_available,
        )

    @app.get("/health/embeddings", tags=["health"])
    def embeddings_health(request: Request) -> dict:
        built: Container = request.app.state.container
        embedder = built.embeddings
        ok = embedder.available() if isinstance(embedder, LazyEmbeddings) else True
        return {"available": ok, "provider": built.settings.embedding_provider}

    app.include_router(documents.router)
    app.include_router(research.router)
    app.include_router(conversations.router)
    app.include_router(reports.router)
    app.include_router(agent.router)
    return app
