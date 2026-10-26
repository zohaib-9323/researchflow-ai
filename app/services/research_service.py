"""Runs research questions through the LangGraph workflow and persists the results."""

from __future__ import annotations

from typing import Any

from langgraph.graph.state import CompiledStateGraph

from app.config import Settings
from app.core.exceptions import LLMNotConfiguredError, NotFoundError, ValidationFailedError
from app.core.logging import conversation_id_var, get_logger
from app.graph.state import initial_turn_state
from app.graph.workflow import thread_config
from app.llm.models import models_used_var
from app.llm.research_llm import ResearchLLM
from app.persistence.database import Database
from app.persistence.models import Conversation, ResearchTurn
from app.persistence.repositories import CollectionRepository, ConversationRepository
from app.schemas.research import (
    ConversationDetail,
    ConversationSummary,
    EvaluationResult,
    ExecutionStep,
    ExecutionSummary,
    ResearchRequest,
    ResearchResponse,
)

logger = get_logger(__name__)


def execution_summary(state: dict[str, Any]) -> ExecutionSummary:
    evaluation = state.get("evaluation") or {}
    return ExecutionSummary(
        steps=[ExecutionStep(**step) for step in state.get("steps", [])],
        retrieval_attempts=state.get("retrieval_attempts", 0),
        answer_revisions=state.get("answer_revisions", 0),
        retries_used=state.get("retry_count", 0),
        max_retries=state.get("max_retries", 0),
        passed_evaluation=evaluation.get("outcome") == "sufficient",
        final_reason=state.get("final_reason", ""),
        search_queries=state.get("search_queries", []),
    )


def turn_to_response(turn: ResearchTurn) -> ResearchResponse:
    return ResearchResponse(
        conversation_id=turn.conversation_id,
        turn_id=turn.id,
        question=turn.question,
        status=turn.status,  # type: ignore[arg-type]
        answer=turn.answer,
        claims=turn.claims or [],  # type: ignore[arg-type]
        citations=turn.citations or [],  # type: ignore[arg-type]
        evidence=turn.evidence or [],  # type: ignore[arg-type]
        evaluation=EvaluationResult.model_validate(turn.evaluation) if turn.evaluation else None,
        limitations=turn.limitations or [],
        execution=ExecutionSummary.model_validate(turn.execution or {}),
        error=(turn.execution or {}).get("error"),
        created_at=turn.created_at,
    )


class ResearchService:
    def __init__(
        self,
        settings: Settings,
        database: Database,
        graph: CompiledStateGraph,
        llm: ResearchLLM,
    ) -> None:
        self.settings = settings
        self.database = database
        self.graph = graph
        self.llm = llm

    def _resolve_conversation(self, request: ResearchRequest) -> Conversation:
        with self.database.session() as session:
            collections = CollectionRepository(session)
            conversations = ConversationRepository(session)
            if request.conversation_id:
                conversation = conversations.get(request.conversation_id)
                if conversation is None:
                    raise NotFoundError("Conversation not found.")
                if request.collection_id and request.collection_id != conversation.collection_id:
                    raise ValidationFailedError(
                        "This conversation belongs to a different collection. Start a new "
                        "conversation to research another collection."
                    )
                return conversation
            if request.collection_id:
                collection = collections.get(request.collection_id)
                if collection is None:
                    raise NotFoundError("Collection not found.")
            else:
                collection = collections.get_or_create_default()
            return conversations.create(collection.id, request.question[:120])

    def ask(self, request: ResearchRequest) -> ResearchResponse:
        if not self.llm.configured:
            raise LLMNotConfiguredError()
        if len(request.question) > self.settings.max_question_chars:
            raise ValidationFailedError(
                f"Questions are limited to {self.settings.max_question_chars} characters."
            )
        conversation = self._resolve_conversation(request)
        token = conversation_id_var.set(conversation.id)
        models_used: list[str] = []
        models_token = models_used_var.set(models_used)
        try:
            state = self.graph.invoke(
                initial_turn_state(
                    request.question, conversation.collection_id, self.settings.max_retries
                ),
                thread_config(conversation.id),
            )
            return self._persist(conversation.id, request.question, state, models_used)
        finally:
            models_used_var.reset(models_token)
            conversation_id_var.reset(token)

    def _persist(
        self,
        conversation_id: str,
        question: str,
        state: dict[str, Any],
        models_used: list[str] | None = None,
    ) -> ResearchResponse:
        execution = execution_summary(state).model_dump()
        execution["models_used"] = list(dict.fromkeys(m for m in models_used or [] if m))
        if state.get("error"):
            execution["error"] = state["error"]
        with self.database.session() as session:
            repo = ConversationRepository(session)
            conversation = repo.get(conversation_id)
            if conversation is None:
                raise NotFoundError("Conversation not found.")
            turn = repo.add_turn(
                conversation,
                question=question,
                answer=state.get("answer", ""),
                status=state.get("status", "error"),
                claims=state.get("claims", []),
                citations=state.get("citations", []),
                evidence=state.get("retrieved_documents", []),
                evaluation=state.get("evaluation"),
                limitations=state.get("limitations", []),
                execution=execution,
            )
            response = turn_to_response(turn)
        logger.info(
            "research_completed",
            extra={
                "status": response.status,
                "retry_count": response.execution.retries_used,
                "error_category": (state.get("error") or {}).get("code"),
            },
        )
        return response

    # -- conversations ---------------------------------------------------------------------
    def list_conversations(self) -> list[ConversationSummary]:
        with self.database.session() as session:
            repo = ConversationRepository(session)
            return [
                ConversationSummary(
                    id=c.id,
                    title=c.title,
                    collection_id=c.collection_id,
                    turn_count=repo.turn_count(c.id),
                    created_at=c.created_at,
                    updated_at=c.updated_at,
                )
                for c in repo.list()
            ]

    def get_conversation(self, conversation_id: str) -> ConversationDetail:
        with self.database.session() as session:
            conversation = ConversationRepository(session).get(conversation_id)
            if conversation is None:
                raise NotFoundError("Conversation not found.")
            turns = [turn_to_response(turn) for turn in conversation.turns]
            return ConversationDetail(
                id=conversation.id,
                title=conversation.title,
                collection_id=conversation.collection_id,
                turn_count=len(turns),
                created_at=conversation.created_at,
                updated_at=conversation.updated_at,
                turns=turns,
            )
