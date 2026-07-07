"""Tests for M9 Ingestion and Validation.

The tests cover the highest-value branches needed for the CI coverage floor without using the full
UCI Online Retail dataset. Each test creates a small CSV or XLSX file, runs it through Ingestor,
and checks the result or the specific exception raised.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src import ingestion
from src.ingestion import (
    ContentInspector,
    DataQualityReport,
    FileKind,
    FileTooLargeError,
    FormulaInjectionError,
    IngestionResult,
    Ingestor,
    IssueSeverity,
    MimeMismatchError,
    SchemaValidator,
    SchemaViolationError,
    UnsupportedFileTypeError,
    ValidationIssue,
)

REQUIRED = list(ingestion.REQUIRED_COLUMNS)


def _valid_row(**overrides: object) -> dict[str, object]:
    """Return a minimal row that satisfies the M9 schema."""

    base: dict[str, object] = {
        "InvoiceNo": "536365",
        "StockCode": "85123A",
        "Description": "WHITE HANGING HEART T-LIGHT HOLDER",
        "Quantity": "6",
        "InvoiceDate": "12/1/2010 8:26",
        "UnitPrice": "2.55",
        "CustomerID": "17850",
        "Country": "United Kingdom",
    }
    base.update(overrides)
    return base


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    frame = pd.DataFrame(rows, columns=REQUIRED)
    frame.to_csv(path, index=False)
    return path


def _write_xlsx(path: Path, rows: list[dict[str, object]]) -> Path:
    frame = pd.DataFrame(rows, columns=REQUIRED)
    frame.to_excel(path, index=False, engine="openpyxl")
    return path


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


def test_ingest_csv_returns_dataframe_with_expected_columns(tmp_path: Path) -> None:
    source = _write_csv(tmp_path / "retail.csv", [_valid_row(), _valid_row(InvoiceNo="536366")])

    frame = ingestion.ingest(source)

    assert list(frame.columns) == REQUIRED
    assert len(frame) == 2


def test_ingestor_result_carries_report_source_and_kind(tmp_path: Path) -> None:
    source = _write_csv(tmp_path / "retail.csv", [_valid_row()])

    result = Ingestor().ingest(source)

    assert isinstance(result, IngestionResult)
    assert isinstance(result.report, DataQualityReport)
    assert result.kind is FileKind.CSV
    assert result.source == source
    assert result.report.total_rows == 1
    assert result.report.error_count == 0
    assert result.report.has_errors is False


def test_ingest_xlsx_round_trip(tmp_path: Path) -> None:
    source = _write_xlsx(tmp_path / "retail.xlsx", [_valid_row()])

    result = Ingestor().ingest(source)

    assert result.kind is FileKind.XLSX
    assert len(result.frame) == 1


# --------------------------------------------------------------------------- #
# Rejections — each maps to one exception subclass in the M9 hierarchy
# --------------------------------------------------------------------------- #


def test_unsupported_extension_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "retail.txt"
    source.write_text("not a spreadsheet")

    with pytest.raises(UnsupportedFileTypeError):
        Ingestor().ingest(source)


def test_mime_mismatch_between_extension_and_magic_bytes(tmp_path: Path) -> None:
    # A file with the ``.csv`` extension but a ZIP magic-byte header is a
    # content-type spoofing attempt and must be rejected.
    source = tmp_path / "spoofed.csv"
    source.write_bytes(b"PK\x03\x04" + b"\x00" * 32)

    with pytest.raises(MimeMismatchError):
        Ingestor().ingest(source)


def test_schema_violation_when_required_column_missing(tmp_path: Path) -> None:
    row = _valid_row()
    del row["Country"]
    source = tmp_path / "no_country.csv"
    pd.DataFrame([row]).to_csv(source, index=False)

    with pytest.raises(SchemaViolationError):
        Ingestor().ingest(source)


def test_formula_injection_in_text_column_is_rejected(tmp_path: Path) -> None:
    payload = _valid_row(Description="=cmd|' /C calc'!A1")
    source = _write_csv(tmp_path / "malicious.csv", [payload])

    with pytest.raises(FormulaInjectionError):
        Ingestor().ingest(source)


def test_signed_number_in_numeric_column_is_allowed(tmp_path: Path) -> None:
    # A leading ``-`` in ``Quantity`` is a signed-number literal, not a
    # formula trigger — the ingestor must accept it.
    source = _write_csv(tmp_path / "signed.csv", [_valid_row(Quantity="-6")])

    result = Ingestor().ingest(source)

    assert result.report.error_count == 0


def test_file_size_cap_is_enforced(tmp_path: Path) -> None:
    source = _write_csv(tmp_path / "retail.csv", [_valid_row()])
    ingestor = Ingestor(max_file_bytes=8)  # forces the cap to trip

    with pytest.raises(FileTooLargeError):
        ingestor.ingest(source)


def test_row_cap_is_enforced(tmp_path: Path) -> None:
    source = _write_csv(tmp_path / "retail.csv", [_valid_row(), _valid_row()])
    ingestor = Ingestor(max_rows=1)

    with pytest.raises(FileTooLargeError):
        ingestor.ingest(source)


# --------------------------------------------------------------------------- #
# Direct stage tests — cheap and target branches the composition may skip
# --------------------------------------------------------------------------- #


def test_schema_validator_flags_duplicate_columns() -> None:
    frame = pd.DataFrame(
        columns=[*REQUIRED, "Country"],  # duplicate trailing column
    )

    with pytest.raises(SchemaViolationError):
        SchemaValidator().validate(frame)


def test_content_inspector_permissive_returns_issues_instead_of_raising() -> None:
    frame = pd.DataFrame(
        {
            "InvoiceNo": ["=1+1"],
            "StockCode": ["85123A"],
            "Description": ["Widget"],
            "Quantity": ["1"],
            "InvoiceDate": ["12/1/2010 8:26"],
            "UnitPrice": ["1.00"],
            "CustomerID": ["17850"],
            "Country": ["United Kingdom"],
        }
    )

    issues = ContentInspector(strict=False).inspect(frame)

    assert len(issues) == 1
    issue = issues[0]
    assert isinstance(issue, ValidationIssue)
    assert issue.severity is IssueSeverity.ERROR
    assert issue.column == "InvoiceNo"


def test_data_quality_report_counts_are_derived_from_issues() -> None:
    err = ValidationIssue("X", "err", IssueSeverity.ERROR)
    warn = ValidationIssue("Y", "warn", IssueSeverity.WARNING)
    report = DataQualityReport(total_rows=2, accepted_rows=1, issues=(err, warn))

    assert report.has_errors is True
    assert report.error_count == 1
    assert report.warning_count == 1
