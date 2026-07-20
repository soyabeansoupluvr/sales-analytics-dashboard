"""Tests for the M14 audit trail.

These tests cover enums, audit events, log writes, verification, and the local
debug dump. Each test uses a private Settings instance with a temporary SQLite
file. AuditLog uses the bundled migration, so the tested schema matches the
one shipped in src/schema.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.config import Settings
from src.logs import (
    AuditAction,
    AuditEvent,
    AuditIntegrityError,
    AuditLog,
    AuditLogVerifier,
    AuditOutcome,
    Database,
    _dev_dump_audit_log,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _database(tmp_path: Path, filename: str = "audit.db") -> Database:
    settings = Settings(sqlite_url=f"sqlite:///{(tmp_path / filename).as_posix()}")
    return Database(settings=settings)


def _event(
    *,
    timestamp: datetime | None = None,
    actor: str = "alice",
    action: AuditAction = AuditAction.LOGIN_SUCCESS,
    outcome: AuditOutcome = AuditOutcome.SUCCESS,
    resource: str | None = "session:1",
    details: dict | None = None,
) -> AuditEvent:
    return AuditEvent(
        timestamp=timestamp or datetime(2026, 7, 8, 12, 0, tzinfo=timezone.utc),
        actor=actor,
        action=action,
        outcome=outcome,
        resource=resource,
        details=details or {"reason": "unit-test"},
    )


# --------------------------------------------------------------------------- #
# AuditEvent
# --------------------------------------------------------------------------- #


class TestAuditEvent:
    def test_normalizes_timestamp_to_utc(self) -> None:
        eastern = timezone(timedelta(hours=-5))
        event = _event(timestamp=datetime(2026, 7, 8, 7, 0, tzinfo=eastern))
        assert event.timestamp.tzinfo == timezone.utc
        assert event.timestamp == datetime(2026, 7, 8, 12, 0, tzinfo=timezone.utc)

    def test_rejects_naive_timestamp(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            AuditEvent(
                timestamp=datetime(2026, 7, 8, 12, 0),  # no tzinfo
                actor="alice",
                action=AuditAction.LOGIN_SUCCESS,
                outcome=AuditOutcome.SUCCESS,
            )

    def test_rejects_empty_actor(self) -> None:
        with pytest.raises(ValueError, match="actor"):
            AuditEvent(
                timestamp=datetime(2026, 7, 8, 12, 0, tzinfo=timezone.utc),
                actor="",
                action=AuditAction.LOGIN_SUCCESS,
                outcome=AuditOutcome.SUCCESS,
            )

    def test_rejects_wrong_action_type(self) -> None:
        with pytest.raises(TypeError, match="AuditAction"):
            AuditEvent(
                timestamp=datetime(2026, 7, 8, 12, 0, tzinfo=timezone.utc),
                actor="alice",
                action="login_success",  # type: ignore[arg-type]
                outcome=AuditOutcome.SUCCESS,
            )

    def test_rejects_wrong_outcome_type(self) -> None:
        with pytest.raises(TypeError, match="AuditOutcome"):
            AuditEvent(
                timestamp=datetime(2026, 7, 8, 12, 0, tzinfo=timezone.utc),
                actor="alice",
                action=AuditAction.LOGIN_SUCCESS,
                outcome="success",  # type: ignore[arg-type]
            )

    def test_details_are_immutable_after_construction(self) -> None:
        original = {"reason": "click"}
        event = _event(details=original)
        original["reason"] = "changed"
        assert event.details["reason"] == "click"
        with pytest.raises(TypeError):
            event.details["reason"] = "tampered"  # type: ignore[index]


# --------------------------------------------------------------------------- #
# AuditLog.append + hash chain
# --------------------------------------------------------------------------- #


class TestAuditLogAppend:
    def test_first_event_has_null_previous_hash(self, tmp_path: Path) -> None:
        db = _database(tmp_path)
        log = AuditLog(db)

        new_id = log.append(_event())

        assert new_id == 1
        with db.connect() as connection:
            row = connection.execute(
                "SELECT previous_hash, event_hash FROM audit_event WHERE id = ?",
                (new_id,),
            ).fetchone()
        assert row["previous_hash"] is None
        assert isinstance(row["event_hash"], str)
        assert len(row["event_hash"]) == 64  # SHA-256 hex length

    def test_second_event_chains_to_first(self, tmp_path: Path) -> None:
        db = _database(tmp_path)
        log = AuditLog(db)

        first_id = log.append(_event(actor="alice"))
        second_id = log.append(_event(actor="bob"))

        with db.connect() as connection:
            rows = connection.execute(
                "SELECT id, previous_hash, event_hash FROM audit_event" " ORDER BY id ASC"
            ).fetchall()

        assert [r["id"] for r in rows] == [first_id, second_id]
        assert rows[1]["previous_hash"] == rows[0]["event_hash"]
        assert rows[1]["event_hash"] != rows[0]["event_hash"]

    def test_hash_chain_verifies_after_multiple_appends(self, tmp_path: Path) -> None:
        db = _database(tmp_path)
        log = AuditLog(db)
        for actor in ("alice", "bob", "carol", "dave"):
            log.append(_event(actor=actor))

        # Should not raise.
        log.verify()
        AuditLogVerifier(db).verify()

    def test_verify_on_empty_table_passes(self, tmp_path: Path) -> None:
        db = _database(tmp_path)
        AuditLog(db).verify()


# --------------------------------------------------------------------------- #
# AuditLog.read
# --------------------------------------------------------------------------- #


class TestAuditLogRead:
    def test_read_returns_events_in_insertion_order(self, tmp_path: Path) -> None:
        db = _database(tmp_path)
        log = AuditLog(db)
        base = datetime(2026, 7, 8, 12, 0, tzinfo=timezone.utc)
        actors = ("alice", "bob", "carol")
        for offset, actor in enumerate(actors):
            log.append(
                _event(
                    actor=actor,
                    timestamp=base + timedelta(minutes=offset),
                )
            )

        events = log.read()

        assert [event.actor for event in events] == list(actors)

    def test_read_respects_limit(self, tmp_path: Path) -> None:
        db = _database(tmp_path)
        log = AuditLog(db)
        base = datetime(2026, 7, 8, 12, 0, tzinfo=timezone.utc)
        for offset in range(5):
            log.append(_event(timestamp=base + timedelta(minutes=offset)))

        events = log.read(limit=2)

        assert len(events) == 2

    def test_read_respects_since_boundary(self, tmp_path: Path) -> None:
        db = _database(tmp_path)
        log = AuditLog(db)
        base = datetime(2026, 7, 8, 12, 0, tzinfo=timezone.utc)
        for offset in range(4):
            log.append(_event(timestamp=base + timedelta(minutes=offset)))

        events = log.read(since=base + timedelta(minutes=2))

        assert len(events) == 2
        assert all(event.timestamp >= base + timedelta(minutes=2) for event in events)

    def test_read_rejects_naive_since(self, tmp_path: Path) -> None:
        db = _database(tmp_path)
        log = AuditLog(db)
        log.append(_event())

        with pytest.raises(ValueError, match="timezone-aware"):
            log.read(since=datetime(2026, 7, 8, 12, 0))

    def test_read_rejects_non_positive_limit(self, tmp_path: Path) -> None:
        db = _database(tmp_path)
        log = AuditLog(db)
        with pytest.raises(ValueError, match="positive limit"):
            log.read(limit=0)

    def test_read_round_trips_details_and_enums(self, tmp_path: Path) -> None:
        db = _database(tmp_path)
        log = AuditLog(db)
        log.append(
            _event(
                action=AuditAction.PROTECTED_STORE_WRITE,
                outcome=AuditOutcome.DENIED,
                resource="customers/42",
                details={"reason": "no-scope", "attempts": 3},
            )
        )

        (event,) = log.read()

        assert event.action is AuditAction.PROTECTED_STORE_WRITE
        assert event.outcome is AuditOutcome.DENIED
        assert event.resource == "customers/42"
        assert dict(event.details) == {"reason": "no-scope", "attempts": 3}


# --------------------------------------------------------------------------- #
# AuditLogVerifier
# --------------------------------------------------------------------------- #


class TestAuditLogVerifier:
    def test_detects_tampered_details_payload(self, tmp_path: Path) -> None:
        db = _database(tmp_path)
        log = AuditLog(db)
        log.append(_event())
        log.append(_event(actor="bob"))

        with db.connect() as connection:
            connection.execute(
                "UPDATE audit_event SET details_json = ? WHERE id = 1",
                ('{"reason":"tampered"}',),
            )

        with pytest.raises(AuditIntegrityError, match="event_hash mismatch"):
            AuditLogVerifier(db).verify()

    def test_detects_broken_previous_hash_link(self, tmp_path: Path) -> None:
        db = _database(tmp_path)
        log = AuditLog(db)
        log.append(_event(actor="alice"))
        log.append(_event(actor="bob"))

        with db.connect() as connection:
            connection.execute(
                "UPDATE audit_event SET previous_hash = ? WHERE id = 2",
                ("0" * 64,),
            )

        with pytest.raises(AuditIntegrityError, match="previous_hash mismatch"):
            AuditLogVerifier(db).verify()

    def test_detects_deleted_middle_row(self, tmp_path: Path) -> None:
        db = _database(tmp_path)
        log = AuditLog(db)
        log.append(_event(actor="alice"))
        log.append(_event(actor="bob"))
        log.append(_event(actor="carol"))

        with db.connect() as connection:
            connection.execute("DELETE FROM audit_event WHERE id = 2")

        with pytest.raises(AuditIntegrityError, match="previous_hash mismatch"):
            AuditLogVerifier(db).verify()


# --------------------------------------------------------------------------- #
# AuditLog.purge_before
# --------------------------------------------------------------------------- #


class TestAuditLogPurge:
    def test_purge_deletes_rows_before_cutoff(self, tmp_path: Path) -> None:
        db = _database(tmp_path)
        log = AuditLog(db)
        base = datetime(2026, 7, 8, 12, 0, tzinfo=timezone.utc)
        for offset in range(4):
            log.append(_event(timestamp=base + timedelta(minutes=offset)))

        deleted = log.purge_before(base + timedelta(minutes=2))

        assert deleted == 2
        remaining = log.read()
        assert len(remaining) == 2
        assert all(event.timestamp >= base + timedelta(minutes=2) for event in remaining)

    def test_purge_reanchors_chain_so_verify_still_passes(self, tmp_path: Path) -> None:
        db = _database(tmp_path)
        log = AuditLog(db)
        base = datetime(2026, 7, 8, 12, 0, tzinfo=timezone.utc)
        for offset in range(4):
            log.append(_event(timestamp=base + timedelta(minutes=offset)))

        log.purge_before(base + timedelta(minutes=2))

        with db.connect() as connection:
            row = connection.execute(
                "SELECT previous_hash FROM audit_event" " ORDER BY id ASC LIMIT 1"
            ).fetchone()
        assert row["previous_hash"] is None
        log.verify()

    def test_purge_on_empty_table_is_noop(self, tmp_path: Path) -> None:
        db = _database(tmp_path)
        log = AuditLog(db)
        assert log.purge_before(datetime(2026, 7, 8, 12, 0, tzinfo=timezone.utc)) == 0

    def test_purge_rejects_naive_cutoff(self, tmp_path: Path) -> None:
        db = _database(tmp_path)
        log = AuditLog(db)
        with pytest.raises(ValueError, match="timezone-aware"):
            log.purge_before(datetime(2026, 7, 8, 12, 0))


# --------------------------------------------------------------------------- #
# _dev_dump_audit_log
# --------------------------------------------------------------------------- #


class TestDevDumpAuditLog:
    def test_dumps_events_to_stdout(self, tmp_path: Path) -> None:
        db = _database(tmp_path)
        log = AuditLog(db)
        log.append(_event(actor="alice", action=AuditAction.LOGIN_SUCCESS))
        log.append(_event(actor="bob", action=AuditAction.ACCESS_DENIED))

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            _dev_dump_audit_log(db)

        output = buffer.getvalue()
        assert "alice" in output
        assert "bob" in output
        assert "login_success" in output
        assert "access_denied" in output

    def test_dumps_empty_table_without_error(self, tmp_path: Path) -> None:
        db = _database(tmp_path)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            _dev_dump_audit_log(db)
        # Header line is always printed even when the table is empty.
        assert "timestamp" in buffer.getvalue()


# --------------------------------------------------------------------------- #
# Enum coverage
# --------------------------------------------------------------------------- #


class TestEnums:
    def test_audit_action_values_are_lowercase_snake_case(self) -> None:
        for member in AuditAction:
            assert member.value == member.value.lower()
            assert " " not in member.value

    def test_audit_outcome_has_three_values(self) -> None:
        assert {member.value for member in AuditOutcome} == {
            "success",
            "failure",
            "denied",
        }
