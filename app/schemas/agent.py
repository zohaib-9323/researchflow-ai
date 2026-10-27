from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class RunCreate(BaseModel):
    objective: str = Field(min_length=3, max_length=4000)
    collection_id: str | None = Field(default=None, max_length=64)


class RunSummary(BaseModel):
    id: str
    collection_id: str
    objective: str
    status: str
    resumable: bool
    has_report: bool
    created_at: datetime
    updated_at: datetime


class RunDetail(RunSummary):
    mode: str | None = None
    current_task_id: str | None = None
    stop_reason: str | None = None
    tasks: list[dict[str, Any]] = Field(default_factory=list)
    events: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    counters: dict[str, Any] = Field(default_factory=dict)
    limits: dict[str, Any] = Field(default_factory=dict)
    verification: dict[str, Any] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    web_available: bool = False
    error: dict[str, Any] | None = None


class AgentReportOut(BaseModel):
    id: str
    run_id: str
    title: str
    content: dict[str, Any]
    markdown: str
    created_at: datetime
