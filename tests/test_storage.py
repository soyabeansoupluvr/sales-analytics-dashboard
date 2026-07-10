"""M7 - Tests for the protected store.

Covers round-trip encryption/decryption, snapshot lifecycle, tamper detection, role-based column
filtering, view filtering, M8 authorization delegation, and audit-journal side effects.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.access import Actor
from src.config import Settings
from src.logs import AuditAction, AuditLog, AuditOutcome, Database
from src.storage import (
    StorageError,
    list_snapshots,
    read_snapshot,
    read_view,
    write_snapshot,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

_KEY_HEX = "ab" * 32
_ALT_KEY_HEX = "cd" * 32


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        sqlite_url=f"sqlite:///{(tmp_path / 'audit.db').as_posix()}",
        pseudonym_key=_KEY_HEX,
        data_processed_dir=tmp_path / "processed",
        data_raw_dir=tmp_path / "raw",
        log_dir=tmp_path / "logs",
    )


@pytest.fixture
def alt_settings(tmp_path: Path) -> Settings:
    """Same paths as Settings but a different pseudonym_key."""

    return Settings(
        sqlite_url=f"sqlite:///{(tmp_path / 'audit.db').as_posix()}",
        pseudonym_key=_ALT_KEY_HEX,
        data_processed_dir=tmp_path / "processed",
        data_raw_dir=tmp_path / "raw",
        log_dir=tmp_path / "logs",
    )


@pytest.fixture
def audit_log(settings: Settings) -> AuditLog:
    return AuditLog(Database(settings))


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    """A pseudonymized-shape frame with the columns downstream views expect."""

    return pd.DataFrame(
        {
            "InvoiceNo": ["A1", "A2", "A3", "A4"],
            "StockCode": ["SKU1", "SKU2", "SKU1", "SKU3"],
            "Description": ["Widget", "Gadget", "Widget", "Gizmo"],
            "Quantity": [1, 2, 3, 4],
            "InvoiceDate": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
            "UnitPrice": [1.5, 2.5, 1.5, 3.0],
            "CustomerID": ["ps_a", "ps_b", "ps_a", "ps_c"],
            "Country": ["UK", "UK", "US", "UK"],
        }
    )


@pytest.fixture
def admin() -> Actor:
    return Actor(username="carol", role="admin")


@pytest.fixture
def analyst() -> Actor:
    return Actor(username="bob", role="analyst")


@pytest.fixture
def viewer() -> Actor:
    return Actor(username="alice", role="viewer")


# --------------------------------------------------------------------------- #
# Write + read round-trip
# --------------------------------------------------------------------------- #


def test_write_snapshot_persists_encrypted_file(
    sample_frame: pd.DataFrame, settings: Settings, audit_log: AuditLog
) -> None:
    path = write_snapshot(sample_frame, settings, audit_log=audit_log)
    assert path.exists()
    assert path.suffix == ".enc"
    assert not path.read_bytes().startswith(b"PAR1")


def test_read_snapshot_returns_original_frame(
    sample_frame: pd.DataFrame, settings: Settings, audit_log: AuditLog
) -> None:
    write_snapshot(sample_frame, settings, snapshot_id="20260101T000000Z", audit_log=audit_log)
    restored = read_snapshot(settings, snapshot_id="20260101T000000Z")
    pd.testing.assert_frame_equal(
        restored.reset_index(drop=True), sample_frame.reset_index(drop=True)
    )


def test_read_snapshot_defaults_to_latest(
    sample_frame: pd.DataFrame, settings: Settings, audit_log: AuditLog
) -> None:
    write_snapshot(sample_frame, settings, snapshot_id="20260101T000000Z", audit_log=audit_log)
    later = sample_frame.assign(Quantity=[9, 9, 9, 9])
    write_snapshot(later, settings, snapshot_id="20260201T000000Z", audit_log=audit_log)
    restored = read_snapshot(settings)  # no snapshot_id -> latest
    assert (restored["Quantity"] == 9).all()


def test_list_snapshots_returns_sorted_ids(
    sample_frame: pd.DataFrame, settings: Settings, audit_log: AuditLog
) -> None:
    write_snapshot(sample_frame, settings, snapshot_id="20260201T000000Z", audit_log=audit_log)
    write_snapshot(sample_frame, settings, snapshot_id="20260101T000000Z", audit_log=audit_log)
    write_snapshot(sample_frame, settings, snapshot_id="20260301T000000Z", audit_log=audit_log)
    assert list_snapshots(settings) == [
        "20260101T000000Z",
        "20260201T000000Z",
        "20260301T000000Z",
    ]


def test_list_snapshots_empty_when_no_snapshots(settings: Settings) -> None:
    assert list_snapshots(settings) == []


def test_read_snapshot_missing_raises(settings: Settings) -> None:
    with pytest.raises(StorageError, match="not found"):
        read_snapshot(settings, snapshot_id="20260101T000000Z")


def test_read_snapshot_raises_when_none_exist(settings: Settings) -> None:
    with pytest.raises(StorageError, match="no snapshots"):
        read_snapshot(settings)


def test_write_snapshot_rejects_non_dataframe(settings: Settings, audit_log: AuditLog) -> None:
    with pytest.raises(StorageError, match="DataFrame"):
        write_snapshot([1, 2, 3], settings, audit_log=audit_log)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Tamper + wrong key
# --------------------------------------------------------------------------- #


def test_read_snapshot_rejects_wrong_key(
    sample_frame: pd.DataFrame,
    settings: Settings,
    alt_settings: Settings,
    audit_log: AuditLog,
) -> None:
    write_snapshot(sample_frame, settings, snapshot_id="20260101T000000Z", audit_log=audit_log)
    with pytest.raises(StorageError, match="integrity check"):
        read_snapshot(alt_settings, snapshot_id="20260101T000000Z")


def test_read_snapshot_detects_tampering(
    sample_frame: pd.DataFrame, settings: Settings, audit_log: AuditLog
) -> None:
    path = write_snapshot(
        sample_frame, settings, snapshot_id="20260101T000000Z", audit_log=audit_log
    )
    # Flip a byte roughly in the middle of the file.
    data = bytearray(path.read_bytes())
    middle = len(data) // 2
    data[middle] ^= 0x01
    path.write_bytes(bytes(data))

    with pytest.raises(StorageError, match="integrity check"):
        read_snapshot(settings, snapshot_id="20260101T000000Z")


# --------------------------------------------------------------------------- #
# read_view: authorization + column filtering
# --------------------------------------------------------------------------- #


def test_read_view_denies_viewer_on_customers_view(
    sample_frame: pd.DataFrame,
    settings: Settings,
    viewer: Actor,
    audit_log: AuditLog,
) -> None:
    write_snapshot(sample_frame, settings, snapshot_id="s1", audit_log=audit_log)
    with pytest.raises(StorageError, match="not permitted"):
        read_view("customers", viewer, settings, snapshot_id="s1", audit_log=audit_log)


def test_read_view_allows_analyst_on_customers_view(
    sample_frame: pd.DataFrame,
    settings: Settings,
    analyst: Actor,
    audit_log: AuditLog,
) -> None:
    write_snapshot(sample_frame, settings, snapshot_id="s1", audit_log=audit_log)
    out = read_view("customers", analyst, settings, snapshot_id="s1", audit_log=audit_log)
    assert "CustomerID" in out.columns


def test_read_view_filters_customer_id_for_viewer(
    sample_frame: pd.DataFrame,
    settings: Settings,
    viewer: Actor,
    audit_log: AuditLog,
) -> None:
    write_snapshot(sample_frame, settings, snapshot_id="s1", audit_log=audit_log)
    out = read_view("revenue", viewer, settings, snapshot_id="s1", audit_log=audit_log)
    assert "CustomerID" not in out.columns


def test_read_view_revenue_view_columns_for_analyst(
    sample_frame: pd.DataFrame,
    settings: Settings,
    analyst: Actor,
    audit_log: AuditLog,
) -> None:
    write_snapshot(sample_frame, settings, snapshot_id="s1", audit_log=audit_log)
    out = read_view("revenue", analyst, settings, snapshot_id="s1", audit_log=audit_log)
    assert set(out.columns) == {
        "InvoiceNo",
        "InvoiceDate",
        "Quantity",
        "UnitPrice",
        "Country",
    }


def test_read_view_products_view_columns(
    sample_frame: pd.DataFrame,
    settings: Settings,
    admin: Actor,
    audit_log: AuditLog,
) -> None:
    write_snapshot(sample_frame, settings, snapshot_id="s1", audit_log=audit_log)
    out = read_view("products", admin, settings, snapshot_id="s1", audit_log=audit_log)
    assert set(out.columns) == {"StockCode", "Description", "Quantity", "UnitPrice"}


def test_read_view_unknown_view_raises_storage_error(
    sample_frame: pd.DataFrame,
    settings: Settings,
    admin: Actor,
    audit_log: AuditLog,
) -> None:
    write_snapshot(sample_frame, settings, snapshot_id="s1", audit_log=audit_log)
    with pytest.raises(StorageError, match="unknown view"):
        read_view(  # type: ignore[arg-type]
            "kitchen_sink", admin, settings, snapshot_id="s1", audit_log=audit_log
        )


# --------------------------------------------------------------------------- #
# Audit-journal side effects
# --------------------------------------------------------------------------- #


def test_write_snapshot_writes_audit_entry(
    sample_frame: pd.DataFrame,
    settings: Settings,
    audit_log: AuditLog,
) -> None:
    write_snapshot(
        sample_frame,
        settings,
        snapshot_id="20260101T000000Z",
        audit_log=audit_log,
    )
    events = audit_log.read(limit=10)
    assert len(events) == 1
    row = events[0]
    assert row.action == AuditAction.PROTECTED_STORE_WRITE
    assert row.outcome == AuditOutcome.SUCCESS
    assert row.resource == "20260101T000000Z"
    assert row.actor == "pipeline"
    assert row.details["rows"] == 4
    assert row.details["columns"] == 8


def test_read_view_denial_writes_only_the_authorize_entry(
    sample_frame: pd.DataFrame,
    settings: Settings,
    viewer: Actor,
    audit_log: AuditLog,
) -> None:
    write_snapshot(sample_frame, settings, snapshot_id="s1", audit_log=audit_log)
    # Clear the write audit event by re-reading the log length before/after.
    baseline = len(audit_log.read(limit=100))
    with pytest.raises(StorageError):
        read_view("customers", viewer, settings, snapshot_id="s1", audit_log=audit_log)
    events = audit_log.read(limit=100)
    assert len(events) == baseline + 1
    latest = events[-1]
    assert latest.action == AuditAction.ACCESS_DENIED
    assert latest.resource == "customers"
    assert latest.actor == "alice"


def test_read_view_success_writes_access_read_audit(
    sample_frame: pd.DataFrame,
    settings: Settings,
    analyst: Actor,
    audit_log: AuditLog,
) -> None:
    write_snapshot(sample_frame, settings, snapshot_id="s1", audit_log=audit_log)
    baseline = len(audit_log.read(limit=100))
    read_view("customers", analyst, settings, snapshot_id="s1", audit_log=audit_log)
    events = audit_log.read(limit=100)
    assert len(events) == baseline + 1
    latest = events[-1]
    assert latest.action == AuditAction.PROTECTED_STORE_READ
    assert latest.resource == "customers"
    assert latest.actor == "bob"


def test_write_snapshot_records_actor_when_supplied(
    sample_frame: pd.DataFrame,
    settings: Settings,
    admin: Actor,
    audit_log: AuditLog,
) -> None:
    write_snapshot(
        sample_frame,
        settings,
        actor=admin,
        snapshot_id="s1",
        audit_log=audit_log,
    )
    row = audit_log.read(limit=1)[0]
    assert row.actor == "carol"
    assert row.details["actor_role"] == "admin"
