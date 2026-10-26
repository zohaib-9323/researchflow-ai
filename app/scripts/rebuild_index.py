"""Rebuild FAISS indexes from the chunks stored in the application database.

Use after changing EMBEDDING_MODEL, after an index failure, or if index files are
lost. Also completes deletions that were interrupted by an index error.

    python -m app.scripts.rebuild_index            # all collections
    python -m app.scripts.rebuild_index <id> ...   # specific collections
"""

from __future__ import annotations

import sys

from app.config import get_settings
from app.container import build_container
from app.persistence.repositories import CollectionRepository


def main(argv: list[str]) -> int:
    container = build_container(get_settings())
    with container.database.session() as session:
        collections = {c.id: c.name for c in CollectionRepository(session).list()}
    targets = argv or list(collections)
    for collection_id in targets:
        if collection_id not in collections:
            print(f"Unknown collection: {collection_id}", file=sys.stderr)
            return 1
        chunks = container.documents.rebuild_index(collection_id)
        print(f"Rebuilt '{collections[collection_id]}' ({collection_id}): {chunks} chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
