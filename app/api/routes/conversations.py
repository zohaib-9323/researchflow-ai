from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_container
from app.container import Container
from app.schemas.reports import ReportResponse
from app.schemas.research import ConversationDetail, ConversationSummary

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("", response_model=list[ConversationSummary])
def list_conversations(container: Container = Depends(get_container)) -> list[ConversationSummary]:
    return container.research.list_conversations()


@router.get("/{conversation_id}", response_model=ConversationDetail)
def get_conversation(
    conversation_id: str, container: Container = Depends(get_container)
) -> ConversationDetail:
    return container.research.get_conversation(conversation_id)


@router.get("/{conversation_id}/reports", response_model=list[ReportResponse])
def list_reports(
    conversation_id: str, container: Container = Depends(get_container)
) -> list[ReportResponse]:
    container.research.get_conversation(conversation_id)  # 404 if missing
    return container.reports.list_for_conversation(conversation_id)
