"""M10 — Cleaning / ETL.

Flags cancellations (invoice numbers starting with ``C``), pairs them with
their originals, handles returns, derives ``Revenue = Quantity * UnitPrice``,
and drops rows with a null ``CustomerID``. Emits a cleaned frame consumed by
M6 Pseudonymization.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def clean(frame: "pd.DataFrame") -> "pd.DataFrame":
    """Return a cleaned frame with a derived Revenue column."""
    raise NotImplementedError("Implemented in feature/m10-cleaning")
