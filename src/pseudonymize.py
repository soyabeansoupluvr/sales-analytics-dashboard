"""M6 — Pseudonymization.

Replaces the raw ``CustomerID`` with a keyed HMAC-SHA256 pseudonym so that
the analytics layer never sees a raw identifier. Keys are stored in an
environment-scoped vault (see M11 config), rotated quarterly, and never
written to logs.
"""
from __future__ import annotations

import hmac
import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def pseudonymize_column(frame: "pd.DataFrame", column: str, key: bytes) -> "pd.DataFrame":
    """Return a copy of ``frame`` with ``column`` replaced by an HMAC pseudonym.

    The pseudonym is deterministic per key, so grouping and RFM aggregation
    still work, but it is one-way and unlinkable across key rotations.
    """
    raise NotImplementedError("Implemented in feature/m6-pseudonymize")


def _hmac_hex(value: str, key: bytes) -> str:
    return hmac.new(key, value.encode("utf-8"), hashlib.sha256).hexdigest()
