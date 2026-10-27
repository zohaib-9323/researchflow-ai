from __future__ import annotations

import re

from fastapi import APIRouter, Depends, Query, Response, status

from app.api.dependencies import get_container
from app.container import Container
from app.schemas.agent import AgentReportOut, RunCreate, RunDetail, RunSummary

router = APIRouter(prefix="/agent", tags=["research agent"])


@router.post("/runs", response_model=RunDetail, status_code=status.HTTP_202_ACCEPTED)
def start_run(request: RunCreate, container: Container = Depends(get_container)) -> RunDetail:
    """Start a planned, tool-using research run. Poll GET /agent/runs/{id} for progress."""
    return container.agent.start(request.objective, request.collection_id)


@router.get("/runs", response_model=list[RunSummary])
def list_runs(
    limit: int = Query(default=50, ge=1, le=200), container: Container = Depends(get_container)
) -> list[RunSummary]:
    return container.agent.list(limit)


@router.get("/runs/{run_id}", response_model=RunDetail)
def get_run(run_id: str, container: Container = Depends(get_container)) -> RunDetail:
    """Run status: tasks, tool calls, evaluation outcomes, retries and report status."""
    return container.agent.get(run_id)


@router.post("/runs/{run_id}/resume", response_model=RunDetail, status_code=202)
def resume_run(run_id: str, container: Container = Depends(get_container)) -> RunDetail:
    """Continue an interrupted or failed run from its last checkpoint."""
    return container.agent.resume(run_id)


@router.get("/runs/{run_id}/report", response_model=AgentReportOut)
def get_report(run_id: str, container: Container = Depends(get_container)) -> AgentReportOut:
    return container.agent.report(run_id)


@router.get("/runs/{run_id}/report/download")
def download_report(run_id: str, container: Container = Depends(get_container)) -> Response:
    report = container.agent.report(run_id)
    slug = re.sub(r"[^a-z0-9]+", "-", report.title.lower()).strip("-")[:60] or "report"
    return Response(
        content=report.markdown,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{slug}.md"'},
    )
