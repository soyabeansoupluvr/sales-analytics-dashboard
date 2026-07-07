"""M7 — Protected Store.

Persists the cleaned, pseudonymized frame as encrypted parquet snapshots and
serves role-scoped SQLite views to the analytics layer. Every read is
brokered by M8 Access Control so raw identifiers never leave this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def write_snapshot(frame: "pd.DataFrame", target: Path) -> None:
    """Persist ``frame`` as an encrypted parquet snapshot."""
    raise NotImplementedError("Implemented in feature/m7-storage")


def read_view(view_name: str, role: str) -> "pd.DataFrame":
    """Return a role-scoped view. Non-admin roles never see pseudonym keys."""
    raise NotImplementedError("Implemented in feature/m7-storage")
