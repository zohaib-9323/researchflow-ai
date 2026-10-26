from __future__ import annotations

import re

from fastapi import APIRouter, Depends, Response, status

from app.api.dependencies import get_container
from app.container import Container
from app.schemas.reports import ReportRequest, ReportResponse

router = APIRouter(prefix="/reports", tags=["reports"])


@router.post("", response_model=ReportResponse, status_code=status.HTTP_201_CREATED)
def create_report(
    request: ReportRequest, container: Container = Depends(get_container)
) -> ReportResponse:
    """Generate a structured report from a research turn's answer and evidence."""
    return container.reports.create(request.turn_id)


@router.get("/{report_id}", response_model=ReportResponse)
def get_report(report_id: str, container: Container = Depends(get_container)) -> ReportResponse:
    return container.reports.get(report_id)


@router.get("/{report_id}/download")
def download_report(report_id: str, container: Container = Depends(get_container)) -> Response:
    report = container.reports.get(report_id)
    slug = re.sub(r"[^a-z0-9]+", "-", report.title.lower()).strip("-")[:60] or "report"
    return Response(
        content=report.markdown,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{slug}.md"'},
    )
