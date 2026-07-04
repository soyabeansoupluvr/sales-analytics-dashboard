"""M9 — Ingestion & Validation.

Accepts CSV or XLSX uploads, validates schema, rejects executable content
(macros, formulas beginning with ``=``, mismatched MIME types), and emits a
data-quality report used by M10 Cleaning / ETL.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


REQUIRED_COLUMNS = (
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
)


def ingest(source: Path) -> "pd.DataFrame":
    """Load a UCI-schema retail file into a validated DataFrame.

    Raises :class:`ValueError` when required columns are missing or when a
    cell begins with ``=`` (macro-injection defense per OWASP A03).
    """
    raise NotImplementedError("Implemented in feature/m9-ingestion")
