"""SQLAlchemy engine/session setup.

SQLite is used for the MVP; everything goes through SQLAlchemy so the database URL can
be switched to PostgreSQL without touching the repositories.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


def create_db_engine(database_url: str) -> Engine:
    connect_args: dict[str, Any] = {}
    if database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    engine = create_engine(database_url, connect_args=connect_args)
    if database_url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_connection: Any, _record: Any) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

    return engine


class Database:
    def __init__(self, database_url: str) -> None:
        self.engine = create_db_engine(database_url)
        self._session_factory = sessionmaker(self.engine, expire_on_commit=False)

    def create_all(self) -> None:
        from app.persistence import models  # noqa: F401  (register tables)

        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Transactional scope: commits on success, rolls back on any exception."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except BaseException:
            session.rollback()
            raise
        finally:
            session.close()
