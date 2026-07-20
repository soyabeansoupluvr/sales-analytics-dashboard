"""M14 - Logging, database access, and audit trail.

This module provides structured JSON logging, shared SQLite database plumbing, and the
append-only audit trail. It owns the logger factory, the database connection factory, the
schema migrator, and the audit-log API used by security-sensitive modules.

Design notes:

* Database opens a new SQLite connection for each connect call and returns it as a context manager.
  Connections are not shared across threads, which keeps Streamlit interactions isolated.
* SchemaMigrator reads numbered SQL migration files from src.schema and applies only migrations
  newer than the highest recorded schema_version. Applied migrations are not run again.
* Migrations run lazily on the first successful connect call for each Database instance, so
  callers do not need a separate setup step.
* The database file comes from Settings.sqlite_url. The URL keeps the sqlite:/// prefix so a future
  SQLAlchemy layer can use the same value without translation.
* AuditLog rows form a hash chain. Each row's event_hash includes the
  previous row's event_hash, so edits or deletions break verification.
* The audit trail is tamper-evident, not tamper-proof. A database writer could
  rebuild the chain, so file permissions still matter.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import sys
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType, TracebackType
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
# Audit trail
# --------------------------------------------------------------------------- #


class AuditAction(StrEnum):
    """Security-relevant actions recorded in the audit log.

    Values are stored in audit_event.action as lowercase snake_case strings.
    AuditLog rejects events whose action is not defined here.
    """

    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    ACCESS_DENIED = "access_denied"
    PSEUDONYM_KEY_ROTATED = "pseudonym_key_rotated"
    PROTECTED_STORE_READ = "protected_store_read"
    PROTECTED_STORE_WRITE = "protected_store_write"
    AUDIT_LOG_VERIFIED = "audit_log_verified"


class AuditOutcome(StrEnum):
    """Enumerated outcomes for an audited action."""

    SUCCESS = "success"
    FAILURE = "failure"
    DENIED = "denied"


class AuditIntegrityError(DatabaseError):
    """Raised when the hash chain in the audit table is broken or inconsistent."""


@dataclass(frozen=True, slots=True)
class AuditEvent:
    """A single audited action before it is saved.

    The timestamp is normalized to UTC, and details is copied into an immutable
    mapping so callers cannot change the event after creation.
    """

    timestamp: datetime
    actor: str
    action: AuditAction
    outcome: AuditOutcome
    resource: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.actor, str) or not self.actor:
            raise ValueError("AuditEvent.actor must be a non-empty string")
        if not isinstance(self.action, AuditAction):
            raise TypeError("AuditEvent.action must be an AuditAction member")
        if not isinstance(self.outcome, AuditOutcome):
            raise TypeError("AuditEvent.outcome must be an AuditOutcome member")
        if self.timestamp.tzinfo is None:
            raise ValueError("AuditEvent.timestamp must be timezone-aware")

        # Normalize to UTC and freeze the details mapping.
        object.__setattr__(self, "timestamp", self.timestamp.astimezone(timezone.utc))
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))


def _canonical_details(details: Mapping[str, Any]) -> str:
    """Return a stable JSON encoding of an event's details payload.

    The encoding sorts keys and disables whitespace so identical logical
    payloads produce identical hashes across Python versions and platforms.
    """

    return json.dumps(dict(details), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash_event(
    timestamp: str,
    actor: str,
    action: str,
    outcome: str,
    resource: str | None,
    details_json: str,
    previous_hash: str | None,
) -> str:
    """Return the hex SHA-256 hash for an audit row.

    Field values are separated with ASCII NUL characters so changes in field
    boundaries cannot produce the same input. previous_hash links the row to the
    previous audit event.
    """

    hasher = hashlib.sha256()
    parts = (
        previous_hash or "",
        timestamp,
        actor,
        action,
        outcome,
        resource or "",
        details_json,
    )
    hasher.update("\x00".join(parts).encode("utf-8"))
    return hasher.hexdigest()


@dataclass(frozen=True, slots=True)
class _StoredEvent:
    """An audit row after persistence, including its hash-chain columns."""

    id: int
    event: AuditEvent
    previous_hash: str | None
    event_hash: str


class AuditLog:
    """Append-only audit log backed by SQLite.

    The log uses short-lived connections from Database. Keep one AuditLog instance
    for the application lifetime so its lock can serialize appends and protect the
    hash chain during concurrent Streamlit interactions.
    """

    _INSERT_SQL = (
        "INSERT INTO audit_event"
        " (timestamp, actor, action, outcome, resource, details_json,"
        "  previous_hash, event_hash)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
    )

    def __init__(self, database: Database) -> None:
        self._database = database
        self._append_lock = threading.Lock()

    def append(self, event: AuditEvent) -> int:
        """Save an event and return its database row id.

        Appends are serialized on this instance so previous_hash points to the latest
        event_hash already stored.
        """

        with self._append_lock:
            with self._database.connect() as connection:
                previous_hash = self._latest_hash(connection)
                details_json = _canonical_details(event.details)
                timestamp_iso = event.timestamp.isoformat(timespec="seconds")
                event_hash = _hash_event(
                    timestamp=timestamp_iso,
                    actor=event.actor,
                    action=event.action.value,
                    outcome=event.outcome.value,
                    resource=event.resource,
                    details_json=details_json,
                    previous_hash=previous_hash,
                )
                cursor = connection.execute(
                    self._INSERT_SQL,
                    (
                        timestamp_iso,
                        event.actor,
                        event.action.value,
                        event.outcome.value,
                        event.resource,
                        details_json,
                        previous_hash,
                        event_hash,
                    ),
                )
                new_id = int(cursor.lastrowid or 0)
                _LOG.info(
                    "audit append id=%d actor=%s action=%s outcome=%s",
                    new_id,
                    event.actor,
                    event.action.value,
                    event.outcome.value,
                )
                return new_id

    def read(
        self,
        limit: int = 100,
        since: datetime | None = None,
    ) -> Sequence[AuditEvent]:
        """Return events in chronological order.

        limit caps the number of rows returned. since returns only events at or after
        the supplied time.
        """

        if limit <= 0:
            raise ValueError("AuditLog.read requires a positive limit")

        query = (
            "SELECT timestamp, actor, action, outcome, resource, details_json" " FROM audit_event"
        )
        parameters: tuple[Any, ...] = ()
        if since is not None:
            if since.tzinfo is None:
                raise ValueError("AuditLog.read requires a timezone-aware 'since'")
            query += " WHERE timestamp >= ?"
            parameters = (since.astimezone(timezone.utc).isoformat(timespec="seconds"),)
        query += " ORDER BY id ASC LIMIT ?"
        parameters = parameters + (int(limit),)

        with self._database.connect() as connection:
            rows = connection.execute(query, parameters).fetchall()

        return tuple(self._row_to_event(row) for row in rows)

    def verify(self) -> None:
        """Verify the audit hash chain.

        Uses AuditLogVerifier so tests and administrative tools can run the same
        verification logic directly.
        """

        AuditLogVerifier(self._database).verify()

    def purge_before(self, cutoff: datetime) -> int:
        """Delete old events and rebuild the remaining hash chain.

        Purging is destructive. Rows older than cutoff are removed, and the surviving
        rows are re-linked so verification still passes. The chain no longer proves
        continuity across the purge boundary.
        """

        if cutoff.tzinfo is None:
            raise ValueError("AuditLog.purge_before requires a timezone-aware cutoff")

        cutoff_iso = cutoff.astimezone(timezone.utc).isoformat(timespec="seconds")
        with self._append_lock:
            with self._database.connect() as connection:
                cursor = connection.execute(
                    "DELETE FROM audit_event WHERE timestamp < ?", (cutoff_iso,)
                )
                deleted = int(cursor.rowcount or 0)
                if deleted:
                    self._rebuild_chain(connection)
                    _LOG.info("audit purge deleted=%d cutoff=%s", deleted, cutoff_iso)
        return deleted

    @staticmethod
    def _rebuild_chain(connection: sqlite3.Connection) -> None:
        """Rebuild the hash chain for surviving rows.

        After a purge, the remaining rows may point to deleted events. This method
        makes the earliest surviving row the new chain start and relinks each later
        row to the row before it. Only previous_hash and event_hash are rewritten.

        This preserves verification after the purge, but it does not preserve proof
        across the purge boundary. Export the chain head first if that proof is needed.
        """

        rows = connection.execute(
            "SELECT id, timestamp, actor, action, outcome, resource,"
            " details_json FROM audit_event ORDER BY id ASC"
        ).fetchall()
        previous_hash: str | None = None
        for row in rows:
            new_hash = _hash_event(
                timestamp=row["timestamp"],
                actor=row["actor"],
                action=row["action"],
                outcome=row["outcome"],
                resource=row["resource"],
                details_json=row["details_json"],
                previous_hash=previous_hash,
            )
            connection.execute(
                "UPDATE audit_event SET previous_hash = ?, event_hash = ?" " WHERE id = ?",
                (previous_hash, new_hash, row["id"]),
            )
            previous_hash = new_hash

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    @staticmethod
    def _latest_hash(connection: sqlite3.Connection) -> str | None:
        row = connection.execute(
            "SELECT event_hash FROM audit_event ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return str(row["event_hash"])

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> AuditEvent:
        return AuditEvent(
            timestamp=datetime.fromisoformat(row["timestamp"]),
            actor=row["actor"],
            action=AuditAction(row["action"]),
            outcome=AuditOutcome(row["outcome"]),
            resource=row["resource"],
            details=json.loads(row["details_json"]),
        )


class AuditLogVerifier:
    """Verify the audit hash chain.

    Kept separate from AuditLog so tests and administrative scripts can verify a
    database without using the write path.
    """

    def __init__(self, database: Database) -> None:
        self._database = database

    def verify(self) -> None:
        """Verify each row's stored hash.

        An empty table is valid. The first row must have no previous_hash, and each
        later row must point to the event_hash of the row before it.
        """

        with self._database.connect() as connection:
            rows = connection.execute(
                "SELECT id, timestamp, actor, action, outcome, resource,"
                " details_json, previous_hash, event_hash"
                " FROM audit_event ORDER BY id ASC"
            ).fetchall()

        expected_previous: str | None = None
        for row in rows:
            if row["previous_hash"] != expected_previous:
                raise AuditIntegrityError(
                    f"audit_event id={row['id']} previous_hash mismatch:"
                    f" expected {expected_previous!r},"
                    f" found {row['previous_hash']!r}"
                )
            recomputed = _hash_event(
                timestamp=row["timestamp"],
                actor=row["actor"],
                action=row["action"],
                outcome=row["outcome"],
                resource=row["resource"],
                details_json=row["details_json"],
                previous_hash=row["previous_hash"],
            )
            if recomputed != row["event_hash"]:
                raise AuditIntegrityError(
                    f"audit_event id={row['id']} event_hash mismatch:"
                    f" expected {recomputed!r}, found {row['event_hash']!r}"
                )
            expected_previous = row["event_hash"]


def _dev_dump_audit_log(database: Database) -> None:
    """Print the audit table for local debugging.

    This helper is excluded from __all__ and should not be used by application
    code. Use AuditLog.read instead.
    """

    with database.connect() as connection:
        rows = connection.execute(
            "SELECT id, timestamp, actor, action, outcome, resource,"
            " details_json, previous_hash, event_hash"
            " FROM audit_event ORDER BY id ASC"
        ).fetchall()

    header = (
        f"{'id':>4}  {'timestamp':<25}  {'actor':<16}  {'action':<24}"
        f"  {'outcome':<8}  resource  details"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['id']:>4}  {row['timestamp']:<25}  {row['actor']:<16}"
            f"  {row['action']:<24}  {row['outcome']:<8}"
            f"  {row['resource'] or '-'}  {row['details_json']}"
        )


# --------------------------------------------------------------------------- #
# Public surface
# --------------------------------------------------------------------------- #


__all__ = [
    "AuditAction",
    "AuditEvent",
    "AuditIntegrityError",
    "AuditLog",
    "AuditLogVerifier",
    "AuditOutcome",
    "Database",
    "DatabaseError",
    "Migration",
    "MigrationError",
    "SchemaMigrator",
    "get_logger",
]
