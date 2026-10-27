"""Explicit construction of all services (simple dependency injection).

Tests build a Container with test settings and substitute fakes for the LLM and
embeddings; the API reads the container from `app.state`.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from langchain_core.embeddings import Embeddings
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from app.agent.llm import AgentLLM, GeminiAgentLLM
from app.agent.nodes import AgentDependencies
from app.agent.service import AgentService
from app.agent.tools import TavilySearch, ToolExecutor, WebSearchProvider
from app.agent.workflow import build_agent_graph
from app.config import Settings
from app.core.exceptions import EmbeddingUnavailableError
from app.graph.nodes import GraphDependencies
from app.graph.workflow import build_research_graph
from app.ingestion.indexing import FaissIndexManager
from app.llm.embeddings import build_embeddings, embedding_model_id
from app.llm.key_pool import KeyPool
from app.llm.models import LLMGateway
from app.llm.research_llm import GeminiResearchLLM, ResearchLLM
from app.persistence.checkpointer import create_sqlite_checkpointer
from app.persistence.database import Database
from app.persistence.repositories import CollectionRepository, DocumentRepository
from app.services.document_service import DocumentService
from app.services.report_service import ReportService
from app.services.research_service import ResearchService
from app.services.retrieval_service import RetrievalService
from app.services.user_service import UserService


class LazyEmbeddings(Embeddings):
    """Loads the embedding model on first use so the API can start (limited mode)
    even where the model is unavailable, reporting a clear error instead."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Embeddings | None = None
        self._lock = threading.Lock()

    def _get(self) -> Embeddings:
        with self._lock:
            if self._model is None:
                self._model = build_embeddings(self._settings)
            return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._get().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._get().embed_query(text)

    def available(self) -> bool:
        try:
            self._get()
        except EmbeddingUnavailableError:
            return False
        return True


@dataclass
class Container:
    settings: Settings
    database: Database
    key_pool: KeyPool
    llm: ResearchLLM
    index: FaissIndexManager
    documents: DocumentService
    retrieval: RetrievalService
    graph: CompiledStateGraph
    research: ResearchService
    reports: ReportService
    embeddings: Embeddings
    agent: AgentService
    users: UserService


def build_container(
    settings: Settings,
    *,
    llm: ResearchLLM | None = None,
    embeddings: Embeddings | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    agent_llm: AgentLLM | None = None,
    web_search: WebSearchProvider | None = None,
    agent_background: bool = True,
) -> Container:
    settings.ensure_directories()
    database = Database(settings.database_url_value)
    database.create_all()
    with database.session() as session:
        CollectionRepository(session).get_or_create_default()

    key_pool = KeyPool(
        settings.api_keys,
        settings.chat_models,
        state_path=settings.key_pool_state_path,
        max_wait_seconds=settings.llm_max_wait_seconds,
    )
    gateway = LLMGateway(settings, key_pool)
    research_llm = llm or GeminiResearchLLM(gateway)
    research_agent_llm = agent_llm or GeminiAgentLLM(gateway, settings.agent_max_context_chars)
    if web_search is None and settings.web_search_configured and settings.tavily_api_key:
        web_search = TavilySearch(
            settings.tavily_api_key.get_secret_value(),
            timeout=settings.agent_tool_timeout_seconds,
        )

    embedder = embeddings or LazyEmbeddings(settings)
    index = FaissIndexManager(settings.vector_store_path, embedder, embedding_model_id(settings))
    documents = DocumentService(settings, database, index)
    retrieval = RetrievalService(
        index, database, settings.retrieval_top_k, settings.retrieval_min_score
    )

    def indexed_count(collection_id: str) -> int:
        with database.session() as session:
            return CollectionRepository(session).indexed_document_count(collection_id)

    def document_info(document_id: str) -> dict | None:
        with database.session() as session:
            doc = DocumentRepository(session).get(document_id)
            if doc is None:
                return None
            return {"filename": doc.filename, "pages": doc.page_count, "chunks": doc.chunk_count}

    saver = checkpointer or create_sqlite_checkpointer(settings.checkpoint_path)
    agent_graph = build_agent_graph(
        AgentDependencies(
            llm=research_agent_llm,
            retrieve=lambda cid, query, k: retrieval.retrieve(cid, query, k),
            indexed_document_count=indexed_count,
            executor=ToolExecutor(settings.agent_tool_timeout_seconds),
            web_search=web_search,
            document_info=document_info,
            max_context_chars=settings.agent_max_context_chars,
            search_k=settings.retrieval_top_k,
        ),
        checkpointer=saver,
    )
    graph = build_research_graph(
        GraphDependencies(
            llm=research_llm,
            retrieve=lambda cid, query, k: retrieval.retrieve(cid, query, k),
            indexed_document_count=indexed_count,
            top_k=settings.retrieval_top_k,
            max_evidence=settings.max_evidence_chunks,
            max_question_chars=settings.max_question_chars,
            max_history_turns=settings.max_history_turns,
        ),
        checkpointer=saver,
    )
    return Container(
        settings=settings,
        database=database,
        key_pool=key_pool,
        llm=research_llm,
        index=index,
        documents=documents,
        retrieval=retrieval,
        graph=graph,
        research=ResearchService(settings, database, graph, research_llm),
        reports=ReportService(database, research_llm),
        embeddings=embedder,
        users=UserService(settings, database),
        agent=AgentService(
            settings,
            database,
            agent_graph,
            llm_configured=agent_llm is not None or settings.llm_configured,
            web_available=web_search is not None,
            background=agent_background,
        ),
    )
