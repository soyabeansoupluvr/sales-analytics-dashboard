"""M7 - Protected Store.

Persists cleaned, pseudonymized data as encrypted parquet snapshots and serves role-scoped views
to the analytics layer. Reads are checked by M8 Access Control before data is returned.

Design notes:

* Snapshots use parquet so analytical data stays columnar and typed.
* Snapshot files are encrypted with Fernet. The Fernet key is derived from Settings.pseudonym_key
  with HKDF-SHA256, keeping it separate from the HMAC key used by M6.
* Snapshot names use UTC timestamps such as 20260709T204500Z. list_snapshots returns them in
  ascending order, so the latest snapshot is the last item.
* read_view calls M8 authorization before decrypting. If the role is not allowed,
  M7 raises StorageError.
* M7 expects M6 to remove raw customer identifiers before storage. M7 and M8 then enforce role
  boundaries around the remaining protected data.
* Views are pandas DataFrame filters rather than SQLite views. For this   project, that satisfies
  the role-scoped view contract without extra database plumbing.
"""

from __future__ import annotations

import base64
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from src.access import AccessError, Actor, ViewName, authorize
from src.logs import AuditAction, AuditEvent, AuditLog, AuditOutcome, Database

if TYPE_CHECKING:
    import pandas as pd

    from src.config import Settings


# --------------------------------------------------------------------------- #
# Contracts and constants
# --------------------------------------------------------------------------- #

_SNAPSHOT_SUBDIR = "snapshots"
_SNAPSHOT_SUFFIX = ".parquet.enc"
_SNAPSHOT_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"

_HKDF_INFO = b"sales-analytics-dashboard.protected-store-v1"
_HKDF_LENGTH = 32  # 32 bytes = 256-bit Fernet key material

# Columns each role is permitted to observe in a decrypted snapshot. The manager (aka "viewer")
# gets aggregate-safe columns only; per-customer breakdowns require analyst or admin.
_ROLE_COLUMNS: dict[str, set[str]] = {
    "admin": set(),  # empty set == "all columns"
    "analyst": set(),  # empty set == "all columns"
    "viewer": {  # aggregate view: no per-customer axis
        "InvoiceNo",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "Country",
    },
}

# Additional per-view column subsets applied on top of the role filter.
# Keys are the four canonical view names from M8.
_VIEW_COLUMNS: dict[ViewName, set[str] | None] = {
    "revenue": {"InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice", "Country"},
    "products": {"StockCode", "Description", "Quantity", "UnitPrice"},
    "time": {"InvoiceDate", "Quantity", "UnitPrice"},
    "customers": None,  # None == "all columns the role can see"
}


class StorageError(Exception):
    """Raised for M7 contract violations (missing snapshot, denied view, etc)."""


# --------------------------------------------------------------------------- #
# Key derivation
# --------------------------------------------------------------------------- #


def _fernet_from_settings(settings: "Settings") -> Fernet:
    """Derive a Fernet key from Settings.pseudonym_key.

    Fernet requires a URL-safe base64-encoded 32-byte key. HKDF-SHA256 produces that key
    material while keeping the storage encryption key separate from the M6 HMAC key.
    """

    hex_key = settings.require_pseudonym_key()
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=_HKDF_LENGTH,
        salt=None,
        info=_HKDF_INFO,
    )
    derived = hkdf.derive(bytes.fromhex(hex_key))
    return Fernet(base64.urlsafe_b64encode(derived))


# --------------------------------------------------------------------------- #
# Snapshot lifecycle
# --------------------------------------------------------------------------- #


def _snapshots_dir(settings: "Settings") -> Path:
    return Path(settings.data_processed_dir) / _SNAPSHOT_SUBDIR


def _snapshot_path(settings: "Settings", snapshot_id: str) -> Path:
    return _snapshots_dir(settings) / f"{snapshot_id}{_SNAPSHOT_SUFFIX}"


def _current_snapshot_id() -> str:
    return datetime.now(timezone.utc).strftime(_SNAPSHOT_TIMESTAMP_FORMAT)


def list_snapshots(settings: "Settings") -> list[str]:
    """Return snapshot ids in ascending chronological order."""

    directory = _snapshots_dir(settings)
    if not directory.is_dir():
        return []
    ids = [
        entry.name.removesuffix(_SNAPSHOT_SUFFIX)
        for entry in directory.iterdir()
        if entry.is_file() and entry.name.endswith(_SNAPSHOT_SUFFIX)
    ]
    ids.sort()
    return ids


def _latest_snapshot(settings: "Settings") -> str:
    ids = list_snapshots(settings)
    if not ids:
        raise StorageError("no snapshots available")
    return ids[-1]


# --------------------------------------------------------------------------- #
# Write path
# --------------------------------------------------------------------------- #


def write_snapshot(
    frame: "pd.DataFrame",
    settings: "Settings",
    *,
    actor: Actor | None = None,
    snapshot_id: str | None = None,
    audit_log: AuditLog | None = None,
) -> Path:
    """Write an encrypted parquet snapshot and audit the write.

    Args:
        frame: Pseudonymized DataFrame to store. Callers are responsible for running M6
            before calling M7.
        settings: Application settings used for the destination directory and
            Fernet key derivation.
        actor: Actor recorded on the audit event. Defaults to a pipeline actor.
        snapshot_id: Explicit snapshot id. Defaults to the current UTC timestamp.
        audit_log: Optional audit log injection point for tests.

    Returns:
        Absolute path to the encrypted snapshot.
    """

    import pandas as pd  # local import: keeps import graph light for tests

    if not isinstance(frame, pd.DataFrame):
        raise StorageError("write_snapshot requires a pandas DataFrame")

    snapshot_id = snapshot_id or _current_snapshot_id()
    target = _snapshot_path(settings, snapshot_id)
    target.parent.mkdir(parents=True, exist_ok=True)

    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False, engine="pyarrow")
    ciphertext = _fernet_from_settings(settings).encrypt(buffer.getvalue())
    target.write_bytes(ciphertext)

    _audit(
        settings,
        audit_log,
        actor=actor,
        action=AuditAction.PROTECTED_STORE_WRITE,
        outcome=AuditOutcome.SUCCESS,
        resource=snapshot_id,
        details={"rows": int(len(frame)), "columns": int(frame.shape[1])},
    )
    return target


# --------------------------------------------------------------------------- #
# Read path
# --------------------------------------------------------------------------- #


def read_snapshot(
    settings: "Settings",
    *,
    snapshot_id: str | None = None,
) -> "pd.DataFrame":
    """Return the full decrypted frame for a snapshot.

    This is for administrative tools and tests. Application code should use read_view
    so role checks and view filtering are applied.
    """

    import pandas as pd

    snapshot_id = snapshot_id or _latest_snapshot(settings)
    path = _snapshot_path(settings, snapshot_id)
    if not path.is_file():
        raise StorageError(f"snapshot not found: {snapshot_id}")

    try:
        plaintext = _fernet_from_settings(settings).decrypt(path.read_bytes())
    except InvalidToken as exc:
        raise StorageError(
            f"snapshot {snapshot_id} failed integrity check (wrong key or " "tampered file)"
        ) from exc
    return pd.read_parquet(io.BytesIO(plaintext), engine="pyarrow")


def read_view(
    view: ViewName,
    actor: Actor,
    settings: "Settings",
    *,
    snapshot_id: str | None = None,
    audit_log: AuditLog | None = None,
) -> "pd.DataFrame":
    """Return a role-scoped view of a snapshot.

    M8 authorizes the actor and view before anything is decrypted. If access is denied,
    StorageError is raised. If access is allowed, the snapshot is decrypted and filtered
    by role and view before being returned.
    """

    if view not in _VIEW_COLUMNS:
        raise StorageError(f"unknown view: {view!r}")

    log = audit_log if audit_log is not None else AuditLog(Database(settings))
    try:
        decision = authorize(actor, view, settings, audit_log=log)
    except AccessError as exc:
        raise StorageError(str(exc)) from exc

    if not decision.allowed:
        raise StorageError(f"role {actor.role!r} is not permitted to read view {view!r}")

    frame = read_snapshot(settings, snapshot_id=snapshot_id)
    return _apply_view_filters(frame, actor.role, view)


# --------------------------------------------------------------------------- #
# Filtering helpers
# --------------------------------------------------------------------------- #


def _apply_view_filters(
    frame: "pd.DataFrame",
    role: str,
    view: ViewName,
) -> "pd.DataFrame":
    """Return the intersection of the role filter and the view filter."""

    allowed_by_role = _ROLE_COLUMNS.get(role, set())
    allowed_by_view = _VIEW_COLUMNS.get(view)

    columns = set(frame.columns)
    if allowed_by_role:  # non-empty set == restrict
        columns &= allowed_by_role
    if allowed_by_view is not None:
        columns &= allowed_by_view
    if not columns:
        return frame.iloc[:, 0:0].copy()

    # Preserve the original column order.
    ordered = [c for c in frame.columns if c in columns]
    return frame.loc[:, ordered].copy()


# --------------------------------------------------------------------------- #
# Audit helper
# --------------------------------------------------------------------------- #


def _audit(
    settings: "Settings",
    audit_log: AuditLog | None,
    *,
    actor: Actor | None,
    action: AuditAction,
    outcome: AuditOutcome,
    resource: str | None,
    details: dict[str, object],
) -> None:
    log = audit_log if audit_log is not None else AuditLog(Database(settings))
    actor_name = actor.username if actor is not None else "pipeline"
    actor_role: str = actor.role if actor is not None else "pipeline"
    log.append(
        AuditEvent(
            timestamp=datetime.now(timezone.utc),
            actor=actor_name,
            action=action,
            outcome=outcome,
            resource=resource,
            details={"actor_role": actor_role, **details},
        )
    )


__all__ = [
    "StorageError",
    "list_snapshots",
    "read_snapshot",
    "read_view",
    "write_snapshot",
]
