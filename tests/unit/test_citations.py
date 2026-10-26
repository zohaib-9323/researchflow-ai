from __future__ import annotations

from app.llm.structured_outputs import DraftClaim
from app.schemas.research import AnswerClaim, EvidencePassage
from app.utils.citations import (
    build_citations,
    lexical_support,
    render_answer,
    validate_claims,
)

EVIDENCE = [
    EvidencePassage(
        source_id="S1",
        chunk_id="c1",
        document_id="d1",
        document_name="report.pdf",
        page=3,
        text="Households reduced grid consumption by 18 percent. Costs were 1.2 million dollars.",
        score=0.8,
    ),
    EvidencePassage(
        source_id="S2",
        chunk_id="c2",
        document_id="d2",
        document_name="memo.txt",
        page=None,
        text="The committee found a 12 percent reduction after excluding heat pump homes.",
        score=0.7,
    ),
]


def test_unknown_citation_labels_are_removed() -> None:
    claims, issues = validate_claims(
        [
            DraftClaim(
                statement="Households reduced grid consumption by 18 percent.",
                source_ids=["S1", "S9", "[s2]"],
            )
        ],
        EVIDENCE,
    )
    assert claims[0].citations == ["S1", "S2"]
    assert claims[0].supported
    assert any("S9" in issue for issue in issues)


def test_claim_without_valid_citation_is_unsupported() -> None:
    claims, issues = validate_claims(
        [DraftClaim(statement="Something about solar.", source_ids=["S7"])], EVIDENCE
    )
    assert claims[0].supported is False
    assert claims[0].citations == []
    assert any("no valid citation" in issue for issue in issues)


def test_claim_unrelated_to_cited_passage_is_flagged() -> None:
    claims, issues = validate_claims(
        [
            DraftClaim(
                statement="Penguins migrate across Antarctic glaciers annually.", source_ids=["S1"]
            )
        ],
        EVIDENCE,
    )
    assert claims[0].supported is False
    assert any("overlap" in issue for issue in issues)


def test_lexical_support_handles_inflections() -> None:
    assert lexical_support("consumption reduced 18 percent", [EVIDENCE[0].text]) == 1.0
    assert lexical_support("glaciers penguins", [EVIDENCE[0].text]) == 0.0


def test_citations_come_from_index_metadata_not_llm() -> None:
    claims = [
        AnswerClaim(text="Costs were 1.2 million dollars.", citations=["S1"]),
        AnswerClaim(text="Unsupported", citations=["S2"], supported=False),
    ]
    citations = build_citations(claims, EVIDENCE)
    assert len(citations) == 1  # unsupported claim contributes no citation
    citation = citations[0]
    assert (citation.document_name, citation.page, citation.chunk_id) == ("report.pdf", 3, "c1")
    assert citation.excerpt == "Costs were 1.2 million dollars."


def test_render_answer_only_includes_supported_claims() -> None:
    answer = render_answer(
        [
            AnswerClaim(text="A supported claim.", citations=["S1", "S2"]),
            AnswerClaim(text="Dropped claim.", citations=[], supported=False),
        ]
    )
    assert answer == "- A supported claim. [S1] [S2]"
