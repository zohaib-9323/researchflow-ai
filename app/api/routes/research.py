from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import current_user, get_container
from app.container import Container
from app.schemas.research import ResearchRequest, ResearchResponse
from app.services.user_service import CurrentUser

router = APIRouter(tags=["research"])


@router.post("/research", response_model=ResearchResponse)
def research(
    request: ResearchRequest,
    container: Container = Depends(get_container),
    user: CurrentUser | None = Depends(current_user),
) -> ResearchResponse:
    """Run the research workflow. Omit `conversation_id` to start a new conversation.

    Each question uses one free credit for signed-in users; failed runs are refunded."""
    if request.conversation_id:
        container.users.check_owner(user, container.research.owner_of(request.conversation_id))
    container.users.reserve(user, "question")
    try:
        response = container.research.ask(request, owner_email=user.email if user else None)
    except Exception:
        container.users.refund(user, "question")
        raise
    if response.status == "error":
        container.users.refund(user, "question")
    return response
