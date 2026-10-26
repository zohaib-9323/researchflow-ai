"""Citation validation and deterministic grounding checks.

LLM output is never trusted for citation details: the model only returns source
labels (S1, S2, ...). Labels are checked against the passages actually retrieved for
this answer, and document names, pages and excerpts always come from the index.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from app.llm.structured_outputs import DraftClaim
from app.schemas.research import AnswerClaim, Citation, EvidencePassage

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "may",
        "might",
        "more",
        "most",
        "no",
        "not",
        "of",
        "on",
        "or",
        "our",
        "over",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "under",
        "up",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "also",
        "about",
        "between",
        "both",
        "each",
        "other",
        "some",
        "any",
        "all",
        "only",
        "very",
    ]
)
# Fraction of a claim's content words that must appear in its cited passages.
MIN_LEXICAL_SUPPORT = 0.2
_LABEL = re.compile(r"^S\d+$")


def content_tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in _WORD.findall(text)
        if token.lower() not in _STOPWORDS and len(token) > 2
    }


def lexical_support(claim: str, passages: Iterable[str]) -> float:
    """Share of the claim's content words found in the passages (0..1)."""
    claim_tokens = content_tokens(claim)
    if not claim_tokens:
        return 1.0
    passage_tokens: set[str] = set()
    for passage in passages:
        passage_tokens |= content_tokens(passage)
    # Allow simple morphological variation (e.g. "increase" vs "increased").
    matched = sum(
        1
        for token in claim_tokens
        if token in passage_tokens or any(p.startswith(token[:5]) for p in passage_tokens)
    )
    return matched / len(claim_tokens)


def normalize_label(raw: str) -> str:
    label = raw.strip().strip("[]()").upper().replace(" ", "")
    return label if label.startswith("S") else f"S{label}"


def validate_claims(
    draft_claims: list[DraftClaim], evidence: list[EvidencePassage]
) -> tuple[list[AnswerClaim], list[str]]:
    """Keep only citations to retrieved passages; flag claims with no valid support.

    Returns the validated claims and a list of human-readable issues found.
    """
    by_label = {passage.source_id: passage for passage in evidence}
    claims: list[AnswerClaim] = []
    issues: list[str] = []
    for number, draft in enumerate(draft_claims, start=1):
        statement = draft.statement.strip()
        if not statement:
            continue
        labels: list[str] = []
        for raw in draft.source_ids:
            label = normalize_label(raw)
            if not _LABEL.match(label) or label not in by_label:
                issues.append(f"Claim {number} cited unknown source '{raw}' (removed).")
                continue
            if label not in labels:
                labels.append(label)
        supported = bool(labels)
        if not labels:
            issues.append(f"Claim {number} has no valid citation.")
        else:
            support = lexical_support(statement, (by_label[label].text for label in labels))
            if support < MIN_LEXICAL_SUPPORT:
                supported = False
                issues.append(
                    f"Claim {number} shares little wording with its cited passages "
                    f"({support:.0%} overlap)."
                )
        claims.append(AnswerClaim(text=statement, citations=labels, supported=supported))
    return claims, issues


def best_excerpt(passage: str, claim: str, max_chars: int = 300) -> str:
    """The passage sentence(s) sharing the most words with the claim."""
    sentences = [s.strip() for s in _SENTENCE.split(passage) if s.strip()]
    if not sentences:
        return passage[:max_chars]
    claim_tokens = content_tokens(claim)
    best = max(sentences, key=lambda s: len(content_tokens(s) & claim_tokens))
    if len(best) > max_chars:
        return best[: max_chars - 1].rstrip() + "…"
    return best


def build_citations(claims: list[AnswerClaim], evidence: list[EvidencePassage]) -> list[Citation]:
    """One citation per distinct source used by a supported claim, in first-use order."""
    by_label = {passage.source_id: passage for passage in evidence}
    citations: dict[str, Citation] = {}
    for claim in claims:
        if not claim.supported:
            continue
        for label in claim.citations:
            passage = by_label.get(label)
            if passage is None or label in citations:
                continue
            citations[label] = Citation(
                source_id=label,
                chunk_id=passage.chunk_id,
                document_id=passage.document_id,
                document_name=passage.document_name,
                page=passage.page,
                excerpt=best_excerpt(passage.text, claim.text),
            )
    return list(citations.values())


def format_source(passage: EvidencePassage | Citation) -> str:
    page = f", p. {passage.page}" if passage.page is not None else ""
    return f"{passage.document_name}{page}"


def format_evidence(evidence: list[EvidencePassage]) -> str:
    if not evidence:
        return "(no passages retrieved)"
    return "\n\n".join(
        f"[{passage.source_id}] ({format_source(passage)})\n{passage.text}" for passage in evidence
    )


def format_claims(claims: Sequence[AnswerClaim | DraftClaim]) -> str:
    lines = []
    for number, claim in enumerate(claims, start=1):
        text = claim.text if isinstance(claim, AnswerClaim) else claim.statement
        labels = claim.citations if isinstance(claim, AnswerClaim) else claim.source_ids
        lines.append(f"{number}. {text} [cites: {', '.join(labels) or 'none'}]")
    return "\n".join(lines) or "(no claims)"


def render_answer(claims: list[AnswerClaim]) -> str:
    """Markdown answer built from validated claims with inline source labels."""
    lines = []
    for claim in claims:
        if not claim.supported:
            continue
        markers = "".join(f" [{label}]" for label in claim.citations)
        lines.append(f"- {claim.text}{markers}")
    return "\n".join(lines)
