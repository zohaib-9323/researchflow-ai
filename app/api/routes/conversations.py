from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import current_user, get_container
from app.container import Container
from app.schemas.reports import ReportResponse
from app.schemas.research import ConversationDetail, ConversationSummary
from app.services.user_service import CurrentUser, UserService

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("", response_model=list[ConversationSummary])
def list_conversations(
    container: Container = Depends(get_container),
    user: CurrentUser | None = Depends(current_user),
) -> list[ConversationSummary]:
    return container.research.list_conversations(UserService.owner_filter(user))


@router.get("/{conversation_id}", response_model=ConversationDetail)
def get_conversation(
    conversation_id: str,
    container: Container = Depends(get_container),
    user: CurrentUser | None = Depends(current_user),
) -> ConversationDetail:
    UserService.check_owner(user, container.research.owner_of(conversation_id))
    return container.research.get_conversation(conversation_id)


@router.get("/{conversation_id}/reports", response_model=list[ReportResponse])
def list_reports(
    conversation_id: str,
    container: Container = Depends(get_container),
    user: CurrentUser | None = Depends(current_user),
) -> list[ReportResponse]:
    UserService.check_owner(user, container.research.owner_of(conversation_id))  # 404s too
    return container.reports.list_for_conversation(conversation_id)
