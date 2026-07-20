"""M8 - Tests for access control and audit.

Covers the role-hierarchy predicate, the full authorize() checkpoint (with audit-journal
side effects), and the small-group suppression gate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.access import (
    AccessError,
    Actor,
    Decision,
    authorize,
    check,
    enforce_small_group,
)
from src.config import Settings
from src.logs import AuditAction, AuditLog, AuditOutcome, Database


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

_KEY_HEX = "ab" * 32


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings pointing at an isolated sqlite file per test."""

    db_path = tmp_path / "audit.db"
    return Settings(
        sqlite_url=f"sqlite:///{db_path.as_posix()}",
        pseudonym_key=_KEY_HEX,
        data_processed_dir=tmp_path / "processed",
        data_raw_dir=tmp_path / "raw",
        log_dir=tmp_path / "logs",
    )


@pytest.fixture
def audit_log(settings: Settings) -> AuditLog:
    return AuditLog(Database(settings))


@pytest.fixture
def viewer() -> Actor:
    return Actor(username="alice", role="viewer")


@pytest.fixture
def analyst() -> Actor:
    return Actor(username="bob", role="analyst")


@pytest.fixture
def admin() -> Actor:
    return Actor(username="carol", role="admin")


# --------------------------------------------------------------------------- #
# check()
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "actor_role,minimum_role,expected",
    [
        ("admin", "viewer", True),
        ("admin", "analyst", True),
        ("admin", "admin", True),
        ("analyst", "viewer", True),
        ("analyst", "analyst", True),
        ("analyst", "admin", False),
        ("viewer", "viewer", True),
        ("viewer", "analyst", False),
        ("viewer", "admin", False),
    ],
)
def test_check_respects_role_hierarchy(actor_role: str, minimum_role: str, expected: bool) -> None:
    assert check(actor_role, minimum_role) is expected  # type: ignore[arg-type]


def test_check_rejects_unknown_actor_role() -> None:
    with pytest.raises(AccessError, match="actor role"):
        check("superuser", "viewer")  # type: ignore[arg-type]


def test_check_rejects_unknown_minimum_role() -> None:
    with pytest.raises(AccessError, match="minimum role"):
        check("viewer", "root")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Actor
# --------------------------------------------------------------------------- #


def test_actor_requires_non_empty_username() -> None:
    with pytest.raises(ValueError, match="username"):
        Actor(username="", role="viewer")


def test_actor_requires_known_role() -> None:
    with pytest.raises(ValueError, match="role"):
        Actor(username="alice", role="root")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# authorize()
# --------------------------------------------------------------------------- #


def test_authorize_allows_viewer_to_read_revenue_view(
    viewer: Actor, settings: Settings, audit_log: AuditLog
) -> None:
    decision = authorize(viewer, "revenue", settings, audit_log=audit_log)
    assert decision == Decision(allowed=True, reason="role_ok")


def test_authorize_denies_viewer_from_customers_view(
    viewer: Actor, settings: Settings, audit_log: AuditLog
) -> None:
    decision = authorize(viewer, "customers", settings, audit_log=audit_log)
    assert decision == Decision(allowed=False, reason="role_insufficient")


def test_authorize_allows_analyst_on_customers_view(
    analyst: Actor, settings: Settings, audit_log: AuditLog
) -> None:
    decision = authorize(analyst, "customers", settings, audit_log=audit_log)
    assert decision.allowed is True


def test_authorize_writes_success_audit_entry_on_allow(
    viewer: Actor, settings: Settings, audit_log: AuditLog
) -> None:
    authorize(viewer, "revenue", settings, audit_log=audit_log)
    events = audit_log.read(limit=10)
    assert len(events) == 1
    row = events[0]
    assert row.action == AuditAction.PROTECTED_STORE_READ
    assert row.outcome == AuditOutcome.SUCCESS
    assert row.actor == "alice"
    assert row.resource == "revenue"
    assert row.details["reason"] == "role_ok"


def test_authorize_writes_denied_audit_entry_on_deny(
    viewer: Actor, settings: Settings, audit_log: AuditLog
) -> None:
    authorize(viewer, "customers", settings, audit_log=audit_log)
    events = audit_log.read(limit=10)
    assert len(events) == 1
    row = events[0]
    assert row.action == AuditAction.ACCESS_DENIED
    assert row.outcome == AuditOutcome.FAILURE
    assert row.resource == "customers"
    assert row.details["reason"] == "role_insufficient"


def test_authorize_rejects_unknown_view(
    viewer: Actor, settings: Settings, audit_log: AuditLog
) -> None:
    with pytest.raises(AccessError, match="unknown view"):
        authorize(viewer, "kitchen_sink", settings, audit_log=audit_log)  # type: ignore[arg-type]


def test_authorize_records_actor_role_in_details(
    admin: Actor, settings: Settings, audit_log: AuditLog
) -> None:
    authorize(admin, "customers", settings, audit_log=audit_log)
    row = audit_log.read(limit=1)[0]
    assert row.details["actor_role"] == "admin"
    assert row.details["minimum_role"] == "analyst"


# --------------------------------------------------------------------------- #
# enforce_small_group()
# --------------------------------------------------------------------------- #


def test_enforce_small_group_allows_cohort_at_threshold(settings: Settings) -> None:
    assert enforce_small_group(5, settings) is True


def test_enforce_small_group_allows_cohort_above_threshold(settings: Settings) -> None:
    assert enforce_small_group(9999, settings) is True


def test_enforce_small_group_suppresses_cohort_below_threshold(settings: Settings) -> None:
    assert enforce_small_group(4, settings) is False


def test_enforce_small_group_accepts_explicit_threshold_override(settings: Settings) -> None:
    assert enforce_small_group(3, settings, threshold=3) is True
    assert enforce_small_group(2, settings, threshold=3) is False


def test_enforce_small_group_requires_settings_or_threshold() -> None:
    with pytest.raises(AccessError, match="settings or threshold"):
        enforce_small_group(3)


def test_enforce_small_group_rejects_non_positive_threshold(settings: Settings) -> None:
    with pytest.raises(AccessError, match="threshold must be positive"):
        enforce_small_group(3, settings, threshold=0)


def test_enforce_small_group_rejects_negative_cohort(settings: Settings) -> None:
    with pytest.raises(AccessError, match="cohort_size must be non-negative"):
        enforce_small_group(-1, settings)


def test_enforce_small_group_audits_suppression_when_actor_supplied(
    viewer: Actor, settings: Settings, audit_log: AuditLog
) -> None:
    result = enforce_small_group(
        2, settings, actor=viewer, resource="customers", audit_log=audit_log
    )
    assert result is False
    events = audit_log.read(limit=10)
    assert len(events) == 1
    row = events[0]
    assert row.action == AuditAction.ACCESS_DENIED
    assert row.details["reason"] == "small_group_suppressed"
    assert row.details["cohort_size"] == 2
    assert row.details["threshold"] == 5
    assert row.resource == "customers"


def test_enforce_small_group_does_not_audit_when_allowed(
    viewer: Actor, settings: Settings, audit_log: AuditLog
) -> None:
    enforce_small_group(10, settings, actor=viewer, resource="customers", audit_log=audit_log)
    assert len(audit_log.read(limit=10)) == 0


def test_enforce_small_group_silent_without_actor(settings: Settings, audit_log: AuditLog) -> None:
    """Internal probes (no actor) suppress but do not audit."""

    result = enforce_small_group(2, settings, audit_log=audit_log)
    assert result is False
    assert len(audit_log.read(limit=10)) == 0
