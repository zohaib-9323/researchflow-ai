from __future__ import annotations

import re

from fastapi import APIRouter, Depends, Query, Response, status

from app.api.dependencies import current_user, get_container
from app.container import Container
from app.schemas.agent import AgentReportOut, RunCreate, RunDetail, RunSummary
from app.services.user_service import CurrentUser, UserService

router = APIRouter(prefix="/agent", tags=["research agent"])


def _owned_run(run_id: str, container: Container, user: CurrentUser | None) -> RunDetail:
    run = container.agent.get(run_id)
    UserService.check_owner(user, run.owner_email)
    return run


@router.post("/runs", response_model=RunDetail, status_code=status.HTTP_202_ACCEPTED)
def start_run(
    request: RunCreate,
    container: Container = Depends(get_container),
    user: CurrentUser | None = Depends(current_user),
) -> RunDetail:
    """Start a planned, tool-using research run. Poll GET /agent/runs/{id} for progress.

    Uses one free credit for signed-in users (resuming a run is free)."""
    container.users.reserve(user, "agent_run")
    try:
        return container.agent.start(
            request.objective, request.collection_id, owner_email=user.email if user else None
        )
    except Exception:
        container.users.refund(user, "agent_run")
        raise


@router.get("/runs", response_model=list[RunSummary])
def list_runs(
    limit: int = Query(default=50, ge=1, le=200),
    container: Container = Depends(get_container),
    user: CurrentUser | None = Depends(current_user),
) -> list[RunSummary]:
    return container.agent.list(limit, owner_email=UserService.owner_filter(user))


@router.get("/runs/{run_id}", response_model=RunDetail)
def get_run(
    run_id: str,
    container: Container = Depends(get_container),
    user: CurrentUser | None = Depends(current_user),
) -> RunDetail:
    """Run status: tasks, tool calls, evaluation outcomes, retries and report status."""
    return _owned_run(run_id, container, user)


@router.post("/runs/{run_id}/resume", response_model=RunDetail, status_code=202)
def resume_run(
    run_id: str,
    container: Container = Depends(get_container),
    user: CurrentUser | None = Depends(current_user),
) -> RunDetail:
    """Continue an interrupted or failed run from its last checkpoint."""
    _owned_run(run_id, container, user)
    return container.agent.resume(run_id)


@router.get("/runs/{run_id}/report", response_model=AgentReportOut)
def get_report(
    run_id: str,
    container: Container = Depends(get_container),
    user: CurrentUser | None = Depends(current_user),
) -> AgentReportOut:
    _owned_run(run_id, container, user)
    return container.agent.report(run_id)


@router.get("/runs/{run_id}/report/download")
def download_report(
    run_id: str,
    container: Container = Depends(get_container),
    user: CurrentUser | None = Depends(current_user),
) -> Response:
    _owned_run(run_id, container, user)
    report = container.agent.report(run_id)
    slug = re.sub(r"[^a-z0-9]+", "-", report.title.lower()).strip("-")[:60] or "report"
    return Response(
        content=report.markdown,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{slug}.md"'},
    )
