"""Data models for the research agent.

Two kinds of models live here:

* state records (`ResearchTask`, `SourceRecord`, `ToolCallRecord`, `Finding`, ...) that are
  stored in the graph state as plain dicts and exposed through the API;
* LLM output schemas (`PlanDraft`, `TaskAssessment`, `ComparisonDraft`,
  `AgentReportDraft`) that the model fills in through validated structured output.

Source ids are assigned by the application, never by the model: `D<n>` for document
passages and `W<n>` for web results. Model output may only *reference* ids that exist.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

TaskStatus = Literal["pending", "in_progress", "completed", "insufficient", "failed", "skipped"]
Approach = Literal["document_search", "web_search", "comparison", "calculation", "mixed"]
RunStatus = Literal[
    "queued",
    "running",
    "completed",  # report generated from verified findings
    "partial",  # report generated, but some tasks lacked evidence
    "insufficient_evidence",  # nothing verifiable was found
    "limit_reached",  # stopped by a tool-call / time budget before finishing
    "interrupted",  # process stopped mid-run; can be resumed
    "error",
]
SOURCE_ID_PATTERN = r"^[DW][1-9][0-9]{0,2}$"


# -- state records ---------------------------------------------------------------------------
class ResearchTask(BaseModel):
    id: str  # T1, T2, ...
    description: str
    approach: Approach
    source_requirements: str = ""
    status: TaskStatus = "pending"
    attempts: int = 0
    notes: str = ""  # what is still missing (drives the retry)
    source_ids: list[str] = Field(default_factory=list)
    summary: str = ""


class SourceRecord(BaseModel):
    id: str  # D1 / W1
    kind: Literal["document", "web"]
    title: str  # document file name or page title
    url: str | None = None
    document_id: str | None = None
    chunk_id: str | None = None
    page: int | None = None
    text: str  # the retrieved passage / snippet (untrusted data)
    score: float | None = None
    provenance: str  # e.g. "search_documents('solar savings')" — how it was obtained
    task_id: str


class ToolCallRecord(BaseModel):
    task_id: str
    tool: str
    arguments: dict
    ok: bool
    summary: str  # concise, user-facing result description
    result_count: int = 0
    source_ids: list[str] = Field(default_factory=list)
    error: str | None = None
    duration_ms: int = 0


class Finding(BaseModel):
    statement: str
    source_ids: list[str]
    task_id: str
    verified: bool = False
    support: float | None = None  # lexical support score from verification


class AgentEvent(BaseModel):
    """One entry of the user-visible activity log (no hidden reasoning)."""

    kind: Literal["plan", "task", "tool", "evaluation", "retry", "verify", "report", "stop"]
    message: str
    task_id: str | None = None


# -- LLM output schemas ----------------------------------------------------------------------
class PlannedTask(BaseModel):
    description: str = Field(description="One concrete research step, phrased as a task.")
    approach: Approach = Field(description="How the step should be researched.")
    source_requirements: str = Field(
        default="", description="What kind of source would answer it (e.g. 'council memo')."
    )


class PlanDraft(BaseModel):
    tasks: list[PlannedTask] = Field(
        description="Ordered research tasks. Use one task for a simple factual question."
    )
    rationale: str = Field(description="One sentence on why this plan covers the objective.")


class AssessedFinding(BaseModel):
    statement: str = Field(description="One factual finding stated in the cited sources.")
    source_ids: list[str] = Field(description="Ids of the sources (e.g. 'D2', 'W1') backing it.")


class TaskAssessment(BaseModel):
    outcome: Literal["complete", "needs_more_evidence", "not_answerable"] = Field(
        description=(
            "complete: the sources answer the task; needs_more_evidence: relevant but "
            "incomplete; not_answerable: the available sources cannot answer it."
        )
    )
    findings: list[AssessedFinding] = Field(
        description="Findings for this task, each backed by source ids. May be empty."
    )
    missing_information: str = Field(default="", description="What is still missing.")
    suggested_query: str = Field(default="", description="A better search query, if useful.")
    summary: str = Field(description="One-sentence summary of what was established.")


class ComparisonPoint(BaseModel):
    point: str = Field(description="What the sources say about one aspect.")
    source_ids: list[str]
    relation: Literal["agree", "disagree", "complement"]


class ComparisonDraft(BaseModel):
    points: list[ComparisonPoint]
    summary: str


class ReportItem(BaseModel):
    text: str
    source_ids: list[str] = Field(default_factory=list)


class AgentReportDraft(BaseModel):
    title: str
    executive_summary: str
    key_findings: list[ReportItem]
    comparisons: list[ReportItem] = Field(
        description="Source-backed comparisons between sources; empty if none."
    )
    contradictions_or_gaps: list[str] = Field(
        description="Contradictory or missing evidence, stated plainly."
    )
    limitations: list[str]
    conclusion: str
