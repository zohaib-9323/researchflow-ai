"""SQLAlchemy engine/session setup.

SQLite is used for the MVP; everything goes through SQLAlchemy so the database URL can
be switched to PostgreSQL without touching the repositories.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import Engine, create_engine, event, inspect, text
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
        self._add_missing_columns()

    def _add_missing_columns(self) -> None:
        """Add nullable columns introduced after a table was first created.

        `create_all` never alters existing tables; this keeps older local databases
        working without a migration tool (only simple nullable columns are handled).
        """
        inspector = inspect(self.engine)
        with self.engine.begin() as connection:
            for table in Base.metadata.sorted_tables:
                if not inspector.has_table(table.name):
                    continue
                existing = {column["name"] for column in inspector.get_columns(table.name)}
                for column in table.columns:
                    if column.name not in existing and column.nullable:
                        kind = column.type.compile(dialect=self.engine.dialect)
                        connection.execute(
                            text(f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {kind}')
                        )

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
