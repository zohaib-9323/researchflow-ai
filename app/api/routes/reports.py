from __future__ import annotations

import re

from fastapi import APIRouter, Depends, Response, status

from app.api.dependencies import current_user, get_container
from app.container import Container
from app.schemas.reports import ReportRequest, ReportResponse
from app.services.user_service import CurrentUser, UserService

router = APIRouter(prefix="/reports", tags=["reports"])


def _owned_report(report_id: str, container: Container, user: CurrentUser | None) -> ReportResponse:
    report = container.reports.get(report_id)
    UserService.check_owner(user, container.research.owner_of(report.conversation_id))
    return report


@router.post("", response_model=ReportResponse, status_code=status.HTTP_201_CREATED)
def create_report(
    request: ReportRequest,
    container: Container = Depends(get_container),
    user: CurrentUser | None = Depends(current_user),
) -> ReportResponse:
    """Generate a structured report from a research turn's answer and evidence."""
    UserService.check_owner(user, container.research.turn_owner(request.turn_id))
    return container.reports.create(request.turn_id)


@router.get("/{report_id}", response_model=ReportResponse)
def get_report(
    report_id: str,
    container: Container = Depends(get_container),
    user: CurrentUser | None = Depends(current_user),
) -> ReportResponse:
    return _owned_report(report_id, container, user)


@router.get("/{report_id}/download")
def download_report(
    report_id: str,
    container: Container = Depends(get_container),
    user: CurrentUser | None = Depends(current_user),
) -> Response:
    report = _owned_report(report_id, container, user)
    slug = re.sub(r"[^a-z0-9]+", "-", report.title.lower()).strip("-")[:60] or "report"
    return Response(
        content=report.markdown,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{slug}.md"'},
    )
