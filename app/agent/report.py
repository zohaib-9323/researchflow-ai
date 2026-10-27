"""Turning a model-written report draft into a validated report and Markdown.

The model only references source ids; this module checks every reference against the
sources gathered by the run, drops items whose citations are all invalid, and renders
the citation list from application metadata (never from model text).
"""

from __future__ import annotations

import re
from typing import Any

from app.agent.schemas import AgentReportDraft, ReportItem, SourceRecord

_ID_IN_TEXT = re.compile(r"\[([DW][0-9]{1,3})\]")


def _valid_ids(ids: list[str], sources: dict[str, SourceRecord]) -> list[str]:
    return [sid for sid in dict.fromkeys(ids) if sid in sources]


def _clean_item(item: ReportItem, sources: dict[str, SourceRecord]) -> dict[str, Any] | None:
    ids = _valid_ids(item.source_ids + _ID_IN_TEXT.findall(item.text), sources)
    if not ids:
        return None
    text = _ID_IN_TEXT.sub("", item.text).strip()
    return {"text": text, "source_ids": ids}


def build_report(
    draft: AgentReportDraft,
    *,
    objective: str,
    sources: dict[str, SourceRecord],
    gaps: list[str],
    limitations: list[str],
) -> tuple[dict[str, Any], list[str]]:
    """Validated report content plus a list of problems found while validating."""
    problems: list[str] = []
    findings = []
    for item in draft.key_findings:
        cleaned = _clean_item(item, sources)
        if cleaned is None:
            problems.append(f"Removed a finding without valid citations: {item.text[:120]}")
        else:
            findings.append(cleaned)
    comparisons = []
    for item in draft.comparisons:
        cleaned = _clean_item(item, sources)
        if cleaned is None:
            problems.append(f"Removed a comparison without valid citations: {item.text[:120]}")
        else:
            comparisons.append(cleaned)

    cited = {sid for item in findings + comparisons for sid in item["source_ids"]}
    citations = citations_for(sources, cited)
    content = {
        "title": draft.title.strip()[:300] or "Research report",
        "objective": objective,
        "executive_summary": draft.executive_summary.strip(),
        "key_findings": findings,
        "comparisons": comparisons,
        "contradictions_or_gaps": list(dict.fromkeys([*draft.contradictions_or_gaps, *gaps])),
        "limitations": list(dict.fromkeys([*draft.limitations, *limitations])),
        "conclusion": draft.conclusion.strip(),
        "citations": citations,
    }
    return content, problems


def empty_report(objective: str, gaps: list[str], limitations: list[str]) -> dict[str, Any]:
    """Report for a run that found no verifiable evidence (written without the model)."""
    return {
        "title": "No verifiable evidence found",
        "objective": objective,
        "executive_summary": (
            "The research did not find evidence in the available sources that answers "
            "this objective, so no findings are reported."
        ),
        "key_findings": [],
        "comparisons": [],
        "contradictions_or_gaps": gaps,
        "limitations": limitations,
        "conclusion": "The available sources do not contain sufficient evidence.",
        "citations": [],
    }


def citations_for(sources: dict[str, SourceRecord], ids: set[str]) -> list[dict[str, Any]]:
    """Citation entries built from application metadata, in source-id order."""
    return [_citation(sources[sid]) for sid in sorted(ids, key=_sort_key) if sid in sources]


def _sort_key(source_id: str) -> tuple[str, int]:
    return source_id[0], int(source_id[1:])


def _citation(source: SourceRecord) -> dict[str, Any]:
    return {
        "source_id": source.id,
        "kind": source.kind,
        "title": source.title,
        "url": source.url,
        "document_id": source.document_id,
        "page": source.page,
        "excerpt": source.text[:300],
        "provenance": source.provenance,
    }


def _cite(ids: list[str]) -> str:
    return " " + "".join(f"[{sid}]" for sid in ids) if ids else ""


def render_markdown(content: dict[str, Any], tasks: list[dict[str, Any]] | None = None) -> str:
    lines = [f"# {content['title']}", "", "## Objective", "", content["objective"], ""]
    lines += ["## Executive summary", "", content["executive_summary"] or "—", ""]
    lines += ["## Key findings", ""]
    lines += [f"- {f['text']}{_cite(f['source_ids'])}" for f in content["key_findings"]] or [
        "- No verified findings."
    ]
    lines.append("")
    if content["comparisons"]:
        lines += ["## Source-backed comparisons", ""]
        lines += [f"- {c['text']}{_cite(c['source_ids'])}" for c in content["comparisons"]]
        lines.append("")
    lines += ["## Contradictory or missing evidence", ""]
    lines += [f"- {gap}" for gap in content["contradictions_or_gaps"]] or ["- None identified."]
    lines += ["", "## Limitations", ""]
    lines += [f"- {item}" for item in content["limitations"]] or ["- None noted."]
    lines += ["", "## Conclusion", "", content["conclusion"] or "—", ""]
    if tasks:
        lines += ["## Research process", ""]
        lines += [f"- **{t['id']}** ({t['status']}): {t['description']}" for t in tasks]
        lines.append("")
    lines += ["## Citations", ""]
    if not content["citations"]:
        lines.append("- No sources cited.")
    for c in content["citations"]:
        where = c["url"] or (f"page {c['page']}" if c["page"] is not None else "")
        excerpt = c["excerpt"].replace("\n", " ").strip()
        lines.append(f"- **[{c['source_id']}]** {c['title']}{f' — {where}' if where else ''}")
        lines.append(f"  > {excerpt[:240]}")
    return "\n".join(lines).rstrip() + "\n"
