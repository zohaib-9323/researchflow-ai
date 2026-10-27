"""Load the demo documents through the running API. Safe to run repeatedly.

Used by the Hugging Face Space on start-up, where storage may be wiped on restart.

Run:  python -m app.scripts.seed_demo [--api http://127.0.0.1:8000] [--papers]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import httpx

from app.services.user_service import SYSTEM_EMAIL

SAMPLE_DOCS_DIR = Path(__file__).resolve().parents[2] / "evaluation" / "docs"
PAPERS_COLLECTION = "AI research papers"
# (arXiv id, file name shown in the app)
DEMO_PAPERS = [
    ("1706.03762", "attention_is_all_you_need.pdf"),
    ("2005.11401", "retrieval_augmented_generation_rag.pdf"),
    ("2004.04906", "dense_passage_retrieval.pdf"),
    ("1908.10084", "sentence_bert.pdf"),
    ("1702.08734", "faiss_billion_scale_similarity_search.pdf"),
    ("2307.03172", "lost_in_the_middle.pdf"),
    ("2310.11511", "self_rag.pdf"),
    ("2210.03629", "react_reasoning_and_acting.pdf"),
    ("2201.11903", "chain_of_thought_prompting.pdf"),
    ("2309.15217", "ragas_rag_evaluation.pdf"),
]

logger = logging.getLogger("seed_demo")


def wait_for_api(client: httpx.Client, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            if client.get("/health").status_code == 200:
                return
        except httpx.TransportError:
            pass
        if time.monotonic() > deadline:
            raise SystemExit("API did not become healthy in time")
        time.sleep(2)


def upload(client: httpx.Client, files: list[tuple[str, bytes, str]], collection_id: str) -> None:
    response = client.post(
        "/documents/upload",
        files=[("files", f) for f in files],
        data={"collection_id": collection_id},
    )
    response.raise_for_status()
    result = response.json()
    logger.info("uploaded %s, %s not added", result["succeeded"], result["failed"])


def seed_samples(client: httpx.Client, collections: list[dict]) -> None:
    default = collections[0]
    if default["indexed_documents"]:
        logger.info("collection %r already has documents", default["name"])
        return
    files = [(p.name, p.read_bytes(), "text/plain") for p in sorted(SAMPLE_DOCS_DIR.glob("*.txt"))]
    upload(client, files, default["id"])


def seed_papers(client: httpx.Client, collections: list[dict]) -> None:
    existing = next((c for c in collections if c["name"] == PAPERS_COLLECTION), None)
    if existing and existing["indexed_documents"]:
        logger.info("collection %r already has documents", PAPERS_COLLECTION)
        return
    if existing is None:
        response = client.post("/collections", json={"name": PAPERS_COLLECTION})
        response.raise_for_status()
        existing = response.json()
    files = []
    with httpx.Client(timeout=60, follow_redirects=True) as arxiv:
        for arxiv_id, filename in DEMO_PAPERS:
            try:
                response = arxiv.get(f"https://arxiv.org/pdf/{arxiv_id}")
                response.raise_for_status()
            except httpx.HTTPError as exc:
                logger.warning("could not download %s: %s", arxiv_id, exc)
                continue
            files.append((filename, response.content, "application/pdf"))
    if files:
        upload(client, files, existing["id"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--papers", action="store_true", help="also load the arXiv papers")
    parser.add_argument("--wait", type=float, default=300, help="seconds to wait for the API")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    # With sign-in enabled, uploads need the internal token and an admin identity.
    headers = {"X-User-Email": SYSTEM_EMAIL}
    if token := os.environ.get("INTERNAL_API_TOKEN"):
        headers["X-Internal-Token"] = token
    with httpx.Client(base_url=args.api, timeout=600, headers=headers) as client:
        wait_for_api(client, args.wait)
        collections = client.get("/collections").json()
        seed_samples(client, collections)
        if args.papers:
            seed_papers(client, collections)
    return 0


if __name__ == "__main__":
    sys.exit(main())
