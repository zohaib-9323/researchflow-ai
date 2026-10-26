"""LangGraph checkpoint persistence (SQLite for local development)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver


def create_sqlite_checkpointer(path: Path) -> SqliteSaver:
    """Create a file-backed checkpointer so graph threads survive process restarts.

    `check_same_thread=False` is required because FastAPI runs sync endpoints in a
    thread pool; SqliteSaver serializes access with its own lock.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), check_same_thread=False)
    saver = SqliteSaver(connection)
    saver.setup()
    return saver
