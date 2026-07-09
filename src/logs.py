"""M14 - Logging, database access, and audit trail.

This module provides structured JSON logging and shared SQLite database plumbing. It owns the
logger factory, database connection factory, and schema migrator used by both the audit trail
and the protected data store.

The audit trail API will be added later. This module currently exposes the database and
migration support that API will build on.

Design notes:

* Database opens a new SQLite connection for each connect call and returns it as a context manager.
  Connections are not shared across threads, which keeps Streamlit interactions isolated.
* SchemaMigrator reads numbered SQL migration files from src.schema and applies only migrations
  newer than the highest recorded schema_version. Applied migrations are not run again.
* Migrations run lazily on the first successful connect call for each Database instance, so
  callers do not need a separate setup step.
* The database file comes from Settings.sqlite_url. The URL keeps the sqlite:/// prefix so a future
  SQLAlchemy layer can use the same value without translation.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any, Iterable

from src.config import Settings, get_settings

# --------------------------------------------------------------------------- #
# Structured JSON application logging
# --------------------------------------------------------------------------- #


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(level.upper())
    return logger


_LOG = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Database errors
# --------------------------------------------------------------------------- #


class DatabaseError(RuntimeError):
    """Base class for M14 database errors that are not sqlite3's own."""


class MigrationError(DatabaseError):
    """Raised when a schema migration cannot be parsed or applied."""


# --------------------------------------------------------------------------- #
# Path resolution
# --------------------------------------------------------------------------- #


_SQLITE_URL_RE = re.compile(r"^sqlite:///(?P<path>.+)$")


def _path_from_sqlite_url(url: str) -> Path:
    """Return the filesystem path encoded in a sqlite:/// URL.

    The M11 configuration surface uses the SQLAlchemy URL dialect so a
    later ORM layer can consume the same value without translation. M14
    only needs the file path, which the sqlite:/// scheme places verbatim
    after the third slash.
    """

    match = _SQLITE_URL_RE.match(url.strip())
    if match is None:
        raise DatabaseError(f"Expected a 'sqlite:///<path>' URL; got: {url!r}")
    return Path(match.group("path"))


# --------------------------------------------------------------------------- #
# Schema migrator
# --------------------------------------------------------------------------- #


_SCHEMA_PACKAGE_DIR: Path = Path(__file__).resolve().parent / "schema"
_MIGRATION_FILENAME_RE = re.compile(r"^(?P<version>\d{3})_[a-z0-9_]+\.sql$")


@dataclass(frozen=True, slots=True)
class Migration:
    """A single ordered, idempotent schema change."""

    version: int
    name: str
    sql: str


class SchemaMigrator:
    """Find and apply ordered SQL migrations from src.schema.

    A migration is a file named NNN_name.sql, where NNN is a zero-padded,
    three-digit version number. Applied versions are recorded in the
    schema_version table, which is created on first use.
    """

    _VERSION_TABLE_SQL = (
        "CREATE TABLE IF NOT EXISTS schema_version ("
        " version INTEGER PRIMARY KEY,"
        " name TEXT NOT NULL,"
        " applied_at TEXT NOT NULL"
        ")"
    )

    def __init__(self, migrations_dir: Path = _SCHEMA_PACKAGE_DIR) -> None:
        self._migrations_dir = migrations_dir

    def discover(self) -> tuple[Migration, ...]:
        """Return all migration files sorted by version.

        Files that do not match the NNN_name.sql pattern are ignored, so __init__.py
        and editor scratch files do not break discovery.
        """

        migrations: list[Migration] = []
        if not self._migrations_dir.is_dir():
            return ()
        for entry in sorted(self._migrations_dir.iterdir()):
            if not entry.is_file():
                continue
            match = _MIGRATION_FILENAME_RE.match(entry.name)
            if match is None:
                continue
            version = int(match.group("version"))
            sql = entry.read_text(encoding="utf-8")
            migrations.append(Migration(version=version, name=entry.name, sql=sql))
        self._require_unique_versions(migrations)
        return tuple(migrations)

    @staticmethod
    def _require_unique_versions(migrations: Iterable[Migration]) -> None:
        seen: set[int] = set()
        for migration in migrations:
            if migration.version in seen:
                raise MigrationError(
                    f"Duplicate migration version detected: {migration.version:03d}"
                )
            seen.add(migration.version)

    def apply(self, connection: sqlite3.Connection) -> tuple[int, ...]:
        """Apply all migrations that have not already run.

        Returns the versions newly applied. Existing rows in schema_version are treated
        as authoritative, so later calls against the same database apply nothing.
        """

        connection.execute(self._VERSION_TABLE_SQL)
        applied = {int(row[0]) for row in connection.execute("SELECT version FROM schema_version")}
        newly_applied: list[int] = []
        for migration in self.discover():
            if migration.version in applied:
                continue
            try:
                with connection:  # implicit transaction
                    connection.executescript(migration.sql)
                    connection.execute(
                        "INSERT INTO schema_version (version, name, applied_at)"
                        " VALUES (?, ?, ?)",
                        (
                            migration.version,
                            migration.name,
                            datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        ),
                    )
            except sqlite3.DatabaseError as exc:
                raise MigrationError(f"Failed to apply migration {migration.name}: {exc}") from exc
            newly_applied.append(migration.version)
            _LOG.info(
                "applied schema migration version=%d name=%s",
                migration.version,
                migration.name,
            )
        return tuple(newly_applied)


# --------------------------------------------------------------------------- #
# Database connection factory
# --------------------------------------------------------------------------- #


class _ConnectionContext:
    """Context manager that yields a live sqlite3 connection.

    sqlite3.Connection supports the context-manager protocol, but its exit
    method only commits or rolls back the transaction. It does not close the
    connection, so this adapter closes it at the end of the with block.
    """

    __slots__ = ("_connection",)

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def __enter__(self) -> sqlite3.Connection:
        return self._connection

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        try:
            if exc is None:
                self._connection.commit()
            else:
                self._connection.rollback()
        finally:
            self._connection.close()


class Database:
    """SQLite connection factory for the shared database file.

    The first successful connect call for each instance applies pending migrations.
    Later calls skip the migration check. An instance-level lock prevents
    concurrent Streamlit worker threads from racing during setup.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        migrator: SchemaMigrator | None = None,
    ) -> None:
        self._settings = settings if settings is not None else get_settings()
        self._migrator = migrator if migrator is not None else SchemaMigrator()
        self._path = _path_from_sqlite_url(self._settings.sqlite_url)
        self._migrated = False
        self._migration_lock = threading.Lock()

    @property
    def path(self) -> Path:
        """The SQLite file this database resolves to."""

        return self._path

    def connect(self) -> _ConnectionContext:
        """Open a new connection and apply pending migrations on first use.

        SQLite creates the database file on first connect, but the parent directory
        must already exist. Each returned connection enables foreign keys and WAL mode
        so concurrent readers do not block writers.
        """

        self._path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            self._path,
            detect_types=sqlite3.PARSE_DECLTYPES,
            timeout=30.0,
            isolation_level="DEFERRED",
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        self._ensure_migrated(connection)
        return _ConnectionContext(connection)

    def _ensure_migrated(self, connection: sqlite3.Connection) -> None:
        if self._migrated:
            return
        with self._migration_lock:
            if self._migrated:
                return
            self._migrator.apply(connection)
            self._migrated = True


# --------------------------------------------------------------------------- #
# Public surface
# --------------------------------------------------------------------------- #


__all__ = [
    "Database",
    "DatabaseError",
    "Migration",
    "MigrationError",
    "SchemaMigrator",
    "get_logger",
]
