"""Evaluation harness.

Ground-truth metrics are computed from the labelled dataset (expected documents and
answer terms). The workflow's own `passed_evaluation` flag comes from an LLM judge and
is reported separately, clearly labelled as an estimate.

    python -m app.evaluation.run                 # retrieval metrics only (no LLM calls)
    python -m app.evaluation.run --with-llm      # full workflow (uses your Gemini quota)
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import PROJECT_ROOT, Settings
from app.container import Container, build_container
from app.persistence.repositories import CollectionRepository
from app.schemas.research import ResearchRequest, ResearchResponse

DATASET = PROJECT_ROOT / "evaluation" / "dataset.jsonl"
DOCS = PROJECT_ROOT / "evaluation" / "docs"
RESULTS = PROJECT_ROOT / "evaluation" / "results"
REFUSAL_STATUSES = {"insufficient_evidence", "not_applicable"}


@dataclass
class Example:
    id: str
    question: str
    answerable: bool
    expected_documents: list[str]
    expected_terms: list[str] = field(default_factory=list)


def load_dataset(path: Path = DATASET) -> list[Example]:
    return [Example(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


# -- metric functions (pure, unit-tested) -------------------------------------------------
def retrieval_hit(retrieved_documents: list[str], expected: list[str]) -> bool:
    return any(doc in retrieved_documents for doc in expected)


def reciprocal_rank(retrieved_documents: list[str], expected: list[str]) -> float:
    for rank, doc in enumerate(retrieved_documents, start=1):
        if doc in expected:
            return 1.0 / rank
    return 0.0


def citation_precision(cited_documents: list[str], expected: list[str]) -> float | None:
    if not cited_documents:
        return None
    return sum(doc in expected for doc in cited_documents) / len(cited_documents)


def term_recall(answer: str, terms: list[str]) -> float | None:
    if not terms:
        return None
    lowered = answer.lower()
    return sum(term.lower() in lowered for term in terms) / len(terms)


def is_correct_refusal(response: ResearchResponse) -> bool:
    """For unanswerable questions: refused, or answered with no citations at all."""
    return response.status in REFUSAL_STATUSES or not response.citations


def mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


# -- runner ----------------------------------------------------------------------------------
def ingest(container: Container) -> str:
    with container.database.session() as session:
        collection_id = CollectionRepository(session).get_or_create_default().id
    for path in sorted(DOCS.glob("*")):
        if path.suffix.lower() in {".txt", ".pdf"}:
            document = container.documents.upload(collection_id, path.name, path.read_bytes())
            if document.status != "indexed":
                raise RuntimeError(f"{path.name} failed to index: {document.error_message}")
    return collection_id


def evaluate_retrieval(container: Container, collection_id: str, data: list[Example]) -> dict:
    hits, ranks = [], []
    for example in data:
        if not example.answerable:
            continue
        docs = container.retrieval.retrieve(collection_id, example.question)
        names = [d.metadata["document_name"] for d in docs]
        hits.append(float(retrieval_hit(names, example.expected_documents)))
        ranks.append(reciprocal_rank(names, example.expected_documents))
    return {
        "hit_rate_at_k": mean(hits),
        "mrr": mean(ranks),
        "k": container.settings.retrieval_top_k,
    }


def evaluate_workflow(container: Container, data: list[Example]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for example in data:
        response = container.research.ask(ResearchRequest(question=example.question))
        cited = [c.document_name for c in response.citations]
        rows.append(
            {
                "id": example.id,
                "answerable": example.answerable,
                "status": response.status,
                "retries": response.execution.retries_used,
                "completed": response.status != "error",
                "citation_precision": citation_precision(cited, example.expected_documents)
                if example.answerable
                else None,
                "term_recall": term_recall(response.answer, example.expected_terms),
                "correct_refusal": is_correct_refusal(response) if not example.answerable else None,
                "llm_judge_passed": response.execution.passed_evaluation,
                "error": (response.error or {}).get("code"),
            }
        )
        print(f"  {example.id}: {response.status} (retries={response.execution.retries_used})")

    answerable = [r for r in rows if r["answerable"] and r["completed"]]
    unanswerable = [r for r in rows if not r["answerable"] and r["completed"]]
    return {
        "ground_truth": {
            "completion_rate": mean([float(r["completed"]) for r in rows]),
            "answer_term_recall": mean(
                [r["term_recall"] for r in answerable if r["term_recall"] is not None]
            ),
            "citation_precision": mean(
                [r["citation_precision"] for r in answerable if r["citation_precision"] is not None]
            ),
            "answered_with_citations_rate": mean(
                [
                    float(
                        r["status"] in {"answered", "partial"}
                        and r["citation_precision"] is not None
                    )
                    for r in answerable
                ]
            ),
            "refusal_accuracy": mean([float(r["correct_refusal"]) for r in unanswerable]),
        },
        "workflow": {
            "mean_retries": mean([float(r["retries"]) for r in rows]),
            "questions_with_retry": sum(r["retries"] > 0 for r in rows),
        },
        "llm_judged_estimates": {
            "evaluation_pass_rate": mean(
                [float(r["llm_judge_passed"]) for r in rows if r["completed"]]
            ),
            "note": "Self-assessed by the LLM evaluator; an estimate, not ground truth.",
        },
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-llm", action="store_true", help="run the full workflow")
    args = parser.parse_args()

    data = load_dataset()
    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(data_dir=Path(tmp))
        container = build_container(settings)
        collection_id = ingest(container)
        report: dict[str, Any] = {
            "embedding_provider": settings.embedding_provider,
            "retrieval": evaluate_retrieval(container, collection_id, data),
        }
        if args.with_llm:
            if not settings.llm_configured:
                print("No Gemini API key configured; skipping workflow evaluation.")
            else:
                print(f"Running {len(data)} questions through the workflow…")
                report["workflow"] = evaluate_workflow(container, data)

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "latest.json").write_text(json.dumps(report, indent=2))
    summary = {k: v for k, v in report.items() if k != "workflow"}
    if "workflow" in report:
        summary["workflow"] = {k: v for k, v in report["workflow"].items() if k != "rows"}
    print(json.dumps(summary, indent=2))
    print(f"Full results: {RESULTS / 'latest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
