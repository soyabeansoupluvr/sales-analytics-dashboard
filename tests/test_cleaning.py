"""Tests for M10 Cleaning / ETL.

The tests build small synthetic DataFrames with the string-based output from M9 and cover each
pipeline stage: type coercion, null filtering, cancellation and return pairing, revenue
calculation, and non-product tagging. Together, they meet the CI coverage floor without loading
 the full UCI sample.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import cleaning
from src.cleaning import (
    CancellationTagger,
    Cleaner,
    CleaningError,
    CleaningReport,
    CleaningResult,
    NonProductFilter,
    NullFilter,
    RevenueDeriver,
    StageDelta,
    TypeCoercer,
)
from src.ingestion import REQUIRED_COLUMNS

REQUIRED = list(REQUIRED_COLUMNS)


def _string_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Return a DataFrame that mirrors M9's post-ingestion dtype: all strings."""

    frame = pd.DataFrame(rows, columns=REQUIRED)
    for column in REQUIRED:
        frame[column] = frame[column].astype("string")
    return frame


def _row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "InvoiceNo": "536365",
        "StockCode": "85123A",
        "Description": "Widget",
        "Quantity": "6",
        "InvoiceDate": "12/1/2010 8:26",
        "UnitPrice": "2.55",
        "CustomerID": "17850",
        "Country": "United Kingdom",
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# End-to-end pipeline
# --------------------------------------------------------------------------- #


def test_clean_returns_result_with_derived_columns() -> None:
    frame = _string_frame([_row(), _row(InvoiceNo="536366", Quantity="2", UnitPrice="1.50")])

    result = Cleaner().clean(frame)

    assert isinstance(result, CleaningResult)
    assert isinstance(result.report, CleaningReport)
    assert result.report.input_rows == 2
    assert result.report.output_rows == 2
    # Derived columns land on the output frame.
    for column in ("Revenue", "IsCancellation", "IsAdjustment", "IsReturn", "IsNonProduct"):
        assert column in result.frame.columns


def test_clean_convenience_function_returns_only_frame() -> None:
    frame = _string_frame([_row()])
    out = cleaning.clean(frame)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 1


def test_clean_reports_stage_deltas_in_pipeline_order() -> None:
    frame = _string_frame([_row(), _row(InvoiceNo="536366")])

    result = Cleaner().clean(frame)

    stage_names = [delta.stage for delta in result.report.stages]
    assert stage_names == [
        "type_coercion",
        "null_customer_drop",
        "cancellation_tagging",
        "revenue_derivation",
        "non_product_tagging",
    ]


# --------------------------------------------------------------------------- #
# Structural failures
# --------------------------------------------------------------------------- #


def test_clean_rejects_non_dataframe_input() -> None:
    with pytest.raises(CleaningError):
        Cleaner().clean("not a dataframe")  # type: ignore[arg-type]


def test_clean_rejects_missing_required_columns() -> None:
    frame = pd.DataFrame({"InvoiceNo": ["1"], "StockCode": ["A"]})

    with pytest.raises(CleaningError):
        Cleaner().clean(frame)


def test_clean_rejects_duplicate_column_names() -> None:
    frame = _string_frame([_row()])
    frame = pd.concat([frame, frame[["Country"]]], axis=1)  # duplicates ``Country``

    with pytest.raises(CleaningError):
        Cleaner().clean(frame)


# --------------------------------------------------------------------------- #
# Individual stages
# --------------------------------------------------------------------------- #


def test_null_filter_drops_rows_without_customer_id() -> None:
    frame = _string_frame(
        [
            _row(CustomerID="17850"),
            _row(InvoiceNo="X", CustomerID=""),
            _row(InvoiceNo="Y", CustomerID="not-a-number"),
        ]
    )
    coerced, _ = TypeCoercer().apply(frame)

    filtered, delta = NullFilter().apply(coerced)

    assert len(filtered) == 1
    assert delta.rows_in == 3
    assert delta.rows_out == 1
    assert delta.details["customer_id_rows_dropped"] == 2


def test_cancellation_tagger_flags_cancellations_adjustments_and_returns() -> None:
    frame = _string_frame(
        [
            _row(InvoiceNo="536365", Quantity="6"),
            _row(InvoiceNo="C536365", Quantity="-6"),  # cancellation
            _row(InvoiceNo="A536365", Quantity="-1"),  # adjustment
            _row(InvoiceNo="536366", Quantity="-2"),  # return
            _row(InvoiceNo="ABC123", Quantity="5"),  # not a cancellation despite ``C``
        ]
    )
    coerced, _ = TypeCoercer().apply(frame)
    coerced = coerced[coerced["CustomerID"].notna()]  # NullFilter would do this

    tagged, delta = CancellationTagger().apply(coerced)

    assert tagged.loc[tagged["InvoiceNo"] == "C536365", "IsCancellation"].iloc[0]
    assert tagged.loc[tagged["InvoiceNo"] == "A536365", "IsAdjustment"].iloc[0]
    assert tagged.loc[tagged["InvoiceNo"] == "536366", "IsReturn"].iloc[0]
    # ``ABC123`` starts with ``A`` in position two but not position one, so
    # neither the cancellation nor the adjustment pattern should match.
    assert not tagged.loc[tagged["InvoiceNo"] == "ABC123", "IsCancellation"].iloc[0]
    assert not tagged.loc[tagged["InvoiceNo"] == "ABC123", "IsAdjustment"].iloc[0]
    assert delta.details["cancellations"] == 1
    assert delta.details["adjustments"] == 1
    # Two rows carry a negative quantity that is not an adjustment (the
    # cancellation row and the pure return row), so both are counted.
    assert delta.details["returns"] == 2


def test_cancellation_tagger_pairs_return_to_original() -> None:
    frame = _string_frame(
        [
            _row(InvoiceNo="536365", InvoiceDate="12/1/2010 8:26", Quantity="6"),
            _row(InvoiceNo="C536365", InvoiceDate="12/2/2010 9:00", Quantity="-6"),
        ]
    )
    coerced, _ = TypeCoercer().apply(frame)

    tagged, delta = CancellationTagger().apply(coerced)

    paired = tagged.loc[tagged["IsCancellation"], "PairedInvoiceNo"].iloc[0]
    assert paired == "536365"
    assert delta.details["paired_cancellations"] == 1


def test_revenue_deriver_multiplies_quantity_and_unit_price() -> None:
    frame = _string_frame([_row(Quantity="3", UnitPrice="2.00")])
    coerced, _ = TypeCoercer().apply(frame)
    tagged, _ = CancellationTagger().apply(coerced)

    with_revenue, delta = RevenueDeriver().apply(tagged)

    assert float(with_revenue["Revenue"].iloc[0]) == pytest.approx(6.00)
    assert delta.details["negative_revenue_rows"] == 0


def test_non_product_filter_tags_postage_and_manual_codes() -> None:
    frame = _string_frame(
        [
            _row(StockCode="85123A"),  # product
            _row(InvoiceNo="536366", StockCode="POST"),  # postage
            _row(InvoiceNo="536367", StockCode="M"),  # manual adjustment
        ]
    )
    coerced, _ = TypeCoercer().apply(frame)
    tagged, _ = CancellationTagger().apply(coerced)
    with_revenue, _ = RevenueDeriver().apply(tagged)

    flagged, delta = NonProductFilter().apply(with_revenue)

    assert not bool(flagged.loc[flagged["StockCode"] == "85123A", "IsNonProduct"].iloc[0])
    assert bool(flagged.loc[flagged["StockCode"] == "POST", "IsNonProduct"].iloc[0])
    assert bool(flagged.loc[flagged["StockCode"] == "M", "IsNonProduct"].iloc[0])
    assert delta.details["non_product_rows"] == 2


def test_stage_delta_details_are_immutable() -> None:
    delta = StageDelta("stage", 1, 1, {"k": 1})
    with pytest.raises(TypeError):
        delta.details["k"] = 2  # type: ignore[index]
