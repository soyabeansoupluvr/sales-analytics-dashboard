"""M6 - Pseudonymization.

Replaces raw CustomerID values with keyed HMAC-SHA256 pseudonyms so analytics and
protected storage do not use the original customer identifiers.

Design notes:

* Keys come from Settings and are never written to logs.
* Pseudonyms are deterministic for a given key, so grouping and RFM analysis  still work.
* Different keys produce different pseudonyms, which limits linkage across key  rotations.
* Null CustomerID values should be dropped upstream, but this module passes them through
  unchanged as a defensive fallback.

Entry points:

* pseudonymize_column applies a supplied key to one column.
* pseudonymize_customers reads the key from Settings and rewrites CustomerID.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from src.config import Settings


# --------------------------------------------------------------------------- #
# Contracts
# --------------------------------------------------------------------------- #

_CUSTOMER_ID_COLUMN: str = "CustomerID"


class PseudonymizationError(Exception):
    """Raised for M6 pseudonymization failures.

    Used for contract violations such as a missing input column, separate from generic
    argument errors.
    """


# --------------------------------------------------------------------------- #
# Low-level primitive
# --------------------------------------------------------------------------- #


def pseudonymize_column(
    frame: "pd.DataFrame",
    *,
    column: str = _CUSTOMER_ID_COLUMN,
    key: bytes,
    output_column: str | None = None,
    drop_source: bool = False,
) -> "pd.DataFrame":
    """Return a copy of frame with one column pseudonymized.

    Args:
        frame: Input DataFrame. It is not modified.
        column: Column containing values to pseudonymize. Defaults to CustomerID.
        key: Raw HMAC key bytes. Must be non-empty.
        output_column: Destination column. Defaults to replacing column.
        drop_source: When output_column differs from column, remove the original column.

    Returns:
        A copy with pseudonyms written to output_column.

    Raises:
        PseudonymizationError: If column is missing.
        ValueError: If key is empty.
    """

    import pandas as pd  # local import keeps import graph light for tests

    if not isinstance(key, (bytes, bytearray)) or len(key) == 0:
        raise ValueError("pseudonymize_column requires a non-empty bytes key")
    if column not in frame.columns:
        raise PseudonymizationError(f"pseudonymize_column: input frame has no {column!r} column")

    target = column if output_column is None else output_column
    out = frame.copy()

    source = out[column]
    pseudonyms = source.map(lambda value: _hmac_hex(value, key) if pd.notna(value) else value)
    out[target] = pseudonyms

    if drop_source and target != column:
        out = out.drop(columns=[column])

    return out


# --------------------------------------------------------------------------- #
# Pipeline orchestrator
# --------------------------------------------------------------------------- #


def pseudonymize_customers(
    frame: "pd.DataFrame",
    settings: "Settings",
) -> "pd.DataFrame":
    """Rewrite CustomerID with keyed HMAC pseudonyms.

    Called before the frame is handed to the protected store. The HMAC key is read
    from Settings, and ConfigError is raised if the key is missing.

    The returned frame still has a CustomerID column, but its values are replaced
    with pseudonyms so downstream grouping and joins can continue without renaming.

    Null CustomerID values pass through unchanged. They should already be removed
    upstream, but passthrough is safer than creating a pseudonym from a missing
    value.
    """

    hex_key = settings.require_pseudonym_key()
    key_bytes = bytes.fromhex(hex_key)
    return pseudonymize_column(frame, column=_CUSTOMER_ID_COLUMN, key=key_bytes)


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _hmac_hex(value: object, key: bytes) -> str:
    """Return the HMAC-SHA256 hex digest for value.

    value is converted to a string before encoding so the helper can handle
    customer identifiers that arrive as numbers. Nulls should be filtered before
    calling this function.
    """

    encoded = str(value).encode("utf-8")
    return hmac.new(key, encoded, hashlib.sha256).hexdigest()


__all__ = [
    "PseudonymizationError",
    "pseudonymize_column",
    "pseudonymize_customers",
]
