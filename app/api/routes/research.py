from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_container
from app.container import Container
from app.schemas.research import ResearchRequest, ResearchResponse

router = APIRouter(tags=["research"])


@router.post("/research", response_model=ResearchResponse)
def research(
    request: ResearchRequest, container: Container = Depends(get_container)
) -> ResearchResponse:
    """Run the research workflow. Omit `conversation_id` to start a new conversation."""
    return container.research.ask(request)
