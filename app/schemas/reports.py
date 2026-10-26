from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.research import Citation


class ReportRequest(BaseModel):
    turn_id: str = Field(min_length=1, max_length=64)


class ReportFindingOut(BaseModel):
    statement: str
    kind: Literal["supported_finding", "interpretation"]
    citations: list[str]


class ReportContent(BaseModel):
    title: str
    research_question: str
    executive_summary: str
    key_findings: list[ReportFindingOut]
    agreements: list[str]
    disagreements: list[str]
    limitations: list[str]
    open_questions: list[str]
    conclusion: str
    sources: list[Citation]


class ReportResponse(BaseModel):
    id: str
    turn_id: str
    conversation_id: str
    title: str
    content: ReportContent
    markdown: str
    created_at: datetime
