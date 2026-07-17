"""Tests for the M3 metrics module.

Covers the five public M3 surfaces and the frame validation contract:
revenue summary, product metrics, time series, country metrics, and repeat
rate. Also verifies audit integration when an AuditLog is supplied.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.access import Actor
from src.analytics import (
    AnalyticsError,
    country_metrics,
    product_metrics,
    repeat_rate,
    revenue_summary,
    time_series,
)
from src.config import Settings
from src.logs import AuditAction, AuditLog, AuditOutcome, Database


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

_KEY_HEX = "ab" * 32


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
def audit_log(settings: Settings) -> AuditLog:
    return AuditLog(Database(settings))


@pytest.fixture
def analyst() -> Actor:
    return Actor(username="bob", role="analyst")


def _cleaned_row(**overrides: object) -> dict[str, object]:
    """Return one row conforming to the M10 output contract."""

    base: dict[str, object] = {
        "InvoiceNo": "A1",
        "StockCode": "SKU1",
        "Description": "Widget",
        "Quantity": 1,
        "InvoiceDate": pd.Timestamp("2024-01-15"),
        "UnitPrice": 10.0,
        "CustomerID": "C1",
        "Country": "UK",
        "IsReturn": False,
        "IsAdjustment": False,
        "Revenue": 10.0,
    }
    base.update(overrides)
    return base


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    """A realistic mix: two customers, one return, one adjustment, two countries."""

    return pd.DataFrame(
        [
            _cleaned_row(
                InvoiceNo="A1",
                CustomerID="C1",
                Country="UK",
                StockCode="SKU1",
                Description="Widget",
                Quantity=3,
                UnitPrice=10.0,
                Revenue=30.0,
                InvoiceDate=pd.Timestamp("2024-01-15"),
            ),
            _cleaned_row(
                InvoiceNo="A2",
                CustomerID="C2",
                Country="US",
                StockCode="SKU2",
                Description="Gadget",
                Quantity=2,
                UnitPrice=5.0,
                Revenue=10.0,
                InvoiceDate=pd.Timestamp("2024-02-20"),
            ),
            _cleaned_row(
                InvoiceNo="A3",
                CustomerID="C1",
                Country="UK",
                StockCode="SKU1",
                Description="Widget",
                Quantity=5,
                UnitPrice=10.0,
                Revenue=50.0,
                InvoiceDate=pd.Timestamp("2024-03-10"),
            ),
            # A return of two widgets, paired against A1
            _cleaned_row(
                InvoiceNo="C1",
                CustomerID="C1",
                Country="UK",
                StockCode="SKU1",
                Description="Widget",
                Quantity=-2,
                UnitPrice=10.0,
                Revenue=-20.0,
                IsReturn=True,
                InvoiceDate=pd.Timestamp("2024-03-25"),
            ),
            # An adjustment (bank charges) - excluded from every metric
            _cleaned_row(
                InvoiceNo="A4",
                CustomerID=None,
                Country="UK",
                StockCode="BANK CHARGES",
                Description="Fee",
                Quantity=1,
                UnitPrice=15.0,
                Revenue=15.0,
                IsAdjustment=True,
                InvoiceDate=pd.Timestamp("2024-03-30"),
            ),
        ]
    )


@pytest.fixture
def empty_frame() -> pd.DataFrame:
    """An empty frame that still conforms to the column contract."""

    return pd.DataFrame(
        {
            "InvoiceNo": pd.Series(dtype="string"),
            "StockCode": pd.Series(dtype="string"),
            "Description": pd.Series(dtype="string"),
            "Quantity": pd.Series(dtype="Int64"),
            "InvoiceDate": pd.Series(dtype="datetime64[ns]"),
            "UnitPrice": pd.Series(dtype="Float64"),
            "CustomerID": pd.Series(dtype="string"),
            "Country": pd.Series(dtype="string"),
            "IsReturn": pd.Series(dtype="boolean"),
            "IsAdjustment": pd.Series(dtype="boolean"),
            "Revenue": pd.Series(dtype="Float64"),
        }
    )


# --------------------------------------------------------------------------- #
# Frame validation
# --------------------------------------------------------------------------- #


def test_revenue_summary_rejects_non_dataframe() -> None:
    with pytest.raises(AnalyticsError, match="pandas DataFrame"):
        revenue_summary([1, 2, 3])  # type: ignore[arg-type]


def test_revenue_summary_reports_missing_columns() -> None:
    partial = pd.DataFrame({"InvoiceNo": ["A1"], "Revenue": [1.0]})
    with pytest.raises(AnalyticsError, match="missing required columns"):
        revenue_summary(partial)


def test_all_public_functions_validate_columns(empty_frame: pd.DataFrame) -> None:
    partial = empty_frame.drop(columns=["Revenue"])
    for fn in (revenue_summary, product_metrics, time_series, country_metrics, repeat_rate):
        with pytest.raises(AnalyticsError, match="missing required columns"):
            fn(partial)


# --------------------------------------------------------------------------- #
# revenue_summary
# --------------------------------------------------------------------------- #


def test_revenue_summary_returns_gross_net_and_returns(sample_frame: pd.DataFrame) -> None:
    result = revenue_summary(sample_frame)
    assert result["gross_revenue"] == 90.0  # 30 + 10 + 50
    assert result["returns_value"] == -20.0
    assert result["net_revenue"] == 70.0  # 90 - 20
    assert result["orders"] == 3
    assert result["aov"] == 30.0  # 90 / 3
    assert result["line_items"] == 4  # excludes the adjustment row
    assert result["adjustments"] == 1


def test_revenue_summary_returns_zero_for_empty_frame(empty_frame: pd.DataFrame) -> None:
    result = revenue_summary(empty_frame)
    assert result["gross_revenue"] == 0.0
    assert result["net_revenue"] == 0.0
    assert result["returns_value"] == 0.0
    assert result["orders"] == 0
    assert result["aov"] is None
    assert result["line_items"] == 0
    assert result["adjustments"] == 0


def test_revenue_summary_excludes_adjustments_from_revenue(sample_frame: pd.DataFrame) -> None:
    total_raw = float(sample_frame["Revenue"].sum())
    result = revenue_summary(sample_frame)
    assert total_raw == 85.0  # sanity check on the fixture
    assert result["net_revenue"] == 70.0
    assert result["net_revenue"] != total_raw


def test_revenue_summary_all_returns_frame_yields_negative_net() -> None:
    frame = pd.DataFrame(
        [
            _cleaned_row(InvoiceNo="C1", IsReturn=True, Quantity=-1, Revenue=-10.0),
            _cleaned_row(InvoiceNo="C2", IsReturn=True, Quantity=-2, Revenue=-20.0),
        ]
    )
    result = revenue_summary(frame)
    assert result["gross_revenue"] == 0.0
    assert result["returns_value"] == -30.0
    assert result["net_revenue"] == -30.0
    assert result["orders"] == 0
    assert result["aov"] is None


def test_revenue_summary_returns_python_native_types(sample_frame: pd.DataFrame) -> None:
    result = revenue_summary(sample_frame)
    assert type(result["gross_revenue"]) is float
    assert type(result["orders"]) is int
    assert type(result["line_items"]) is int


# --------------------------------------------------------------------------- #
# product_metrics
# --------------------------------------------------------------------------- #


def test_product_metrics_top_by_revenue_orders_descending(sample_frame: pd.DataFrame) -> None:
    result = product_metrics(sample_frame)
    top = result["top_by_revenue"]
    assert list(top["StockCode"]) == ["SKU1", "SKU2"]
    assert list(top["Revenue"]) == [80.0, 10.0]  # SKU1: 30+50; SKU2: 10


def test_product_metrics_top_by_quantity_orders_descending(sample_frame: pd.DataFrame) -> None:
    result = product_metrics(sample_frame)
    top = result["top_by_quantity"]
    assert list(top["StockCode"]) == ["SKU1", "SKU2"]
    assert list(top["Quantity"]) == [8, 2]  # SKU1: 3+5; SKU2: 2


def test_product_metrics_return_rate_computed_per_sku(sample_frame: pd.DataFrame) -> None:
    result = product_metrics(sample_frame)
    # Return rate is returned quantity divided by sold quantity per SKU.
    rr = result["return_rate"].set_index("StockCode")["return_rate"]
    assert float(rr["SKU1"]) == pytest.approx(0.25)
    assert float(rr["SKU2"]) == pytest.approx(0.0)


def test_product_metrics_top_n_limits_result(sample_frame: pd.DataFrame) -> None:
    result = product_metrics(sample_frame, top_n=1)
    assert len(result["top_by_revenue"]) == 1
    assert result["top_by_revenue"].iloc[0]["StockCode"] == "SKU1"


def test_product_metrics_rejects_non_positive_top_n(sample_frame: pd.DataFrame) -> None:
    with pytest.raises(AnalyticsError, match="positive"):
        product_metrics(sample_frame, top_n=0)


def test_product_metrics_empty_frame_returns_empty_frames(empty_frame: pd.DataFrame) -> None:
    result = product_metrics(empty_frame)
    assert result["top_by_revenue"].empty
    assert result["top_by_quantity"].empty
    assert result["return_rate"].empty
    assert list(result["top_by_revenue"].columns) == [
        "StockCode",
        "Description",
        "Revenue",
        "Quantity",
    ]


# --------------------------------------------------------------------------- #
# time_series
# --------------------------------------------------------------------------- #


def test_time_series_monthly_has_row_per_month(sample_frame: pd.DataFrame) -> None:
    result = time_series(sample_frame)
    monthly = result["monthly"]
    assert len(monthly) == 3
    assert list(monthly["month"]) == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-02-01"),
        pd.Timestamp("2024-03-01"),
    ]


def test_time_series_monthly_uses_net_revenue(sample_frame: pd.DataFrame) -> None:
    result = time_series(sample_frame)
    monthly = result["monthly"].set_index("month")["revenue"]
    # March: 50 sale + (-20) return = 30
    assert float(monthly[pd.Timestamp("2024-03-01")]) == pytest.approx(30.0)


def test_time_series_monthly_orders_excludes_returns(sample_frame: pd.DataFrame) -> None:
    result = time_series(sample_frame)
    monthly = result["monthly"].set_index("month")["orders"]
    # March has one positive invoice (A3) even though there's also a return.
    assert int(monthly[pd.Timestamp("2024-03-01")]) == 1


def test_time_series_weekday_has_seven_rows(sample_frame: pd.DataFrame) -> None:
    result = time_series(sample_frame)
    weekday = result["weekday"]
    assert list(weekday["weekday"]) == [0, 1, 2, 3, 4, 5, 6]
    assert set(weekday.columns) == {"weekday", "revenue"}


def test_time_series_empty_frame_returns_empty_frames(empty_frame: pd.DataFrame) -> None:
    result = time_series(empty_frame)
    assert result["monthly"].empty
    assert result["weekday"].empty


def test_time_series_ignores_rows_with_missing_date() -> None:
    frame = pd.DataFrame(
        [
            _cleaned_row(InvoiceNo="A1", Revenue=10.0, InvoiceDate=pd.NaT),
            _cleaned_row(InvoiceNo="A2", Revenue=20.0, InvoiceDate=pd.Timestamp("2024-05-01")),
        ]
    )
    monthly = time_series(frame)["monthly"]
    assert len(monthly) == 1
    assert float(monthly.iloc[0]["revenue"]) == pytest.approx(20.0)


# --------------------------------------------------------------------------- #
# country_metrics
# --------------------------------------------------------------------------- #


def test_country_metrics_sorted_by_revenue(sample_frame: pd.DataFrame) -> None:
    result = country_metrics(sample_frame)
    assert list(result["Country"]) == ["UK", "US"]
    # UK net = 30 + 50 - 20 = 60; US = 10
    assert float(result.iloc[0]["revenue"]) == pytest.approx(60.0)
    assert float(result.iloc[1]["revenue"]) == pytest.approx(10.0)


def test_country_metrics_orders_excludes_returns(sample_frame: pd.DataFrame) -> None:
    result = country_metrics(sample_frame).set_index("Country")
    assert int(result.loc["UK", "orders"]) == 2
    assert int(result.loc["US", "orders"]) == 1


def test_country_metrics_empty_frame(empty_frame: pd.DataFrame) -> None:
    result = country_metrics(empty_frame)
    assert result.empty
    assert list(result.columns) == ["Country", "revenue", "orders"]


# --------------------------------------------------------------------------- #
# repeat_rate
# --------------------------------------------------------------------------- #


def test_repeat_rate_counts_customers_with_two_or_more_orders(
    sample_frame: pd.DataFrame,
) -> None:
    result = repeat_rate(sample_frame)
    assert result["customers"] == 2
    assert result["repeat_customers"] == 1
    assert result["repeat_rate"] == 0.5


def test_repeat_rate_excludes_returns_when_counting_orders() -> None:
    # C1 has one sale + one return -> counted as one order, not two.
    frame = pd.DataFrame(
        [
            _cleaned_row(InvoiceNo="A1", CustomerID="C1", Revenue=10.0),
            _cleaned_row(
                InvoiceNo="C1", CustomerID="C1", IsReturn=True, Quantity=-1, Revenue=-10.0
            ),
        ]
    )
    result = repeat_rate(frame)
    assert result["customers"] == 1
    assert result["repeat_customers"] == 0
    assert result["repeat_rate"] == 0.0


def test_repeat_rate_ignores_rows_without_customer_id() -> None:
    frame = pd.DataFrame(
        [
            _cleaned_row(InvoiceNo="A1", CustomerID=None),
            _cleaned_row(InvoiceNo="A2", CustomerID=None),
        ]
    )
    result = repeat_rate(frame)
    assert result["customers"] == 0
    assert result["repeat_customers"] == 0
    assert result["repeat_rate"] is None


def test_repeat_rate_empty_frame(empty_frame: pd.DataFrame) -> None:
    result = repeat_rate(empty_frame)
    assert result == {"customers": 0, "repeat_customers": 0, "repeat_rate": None}


# --------------------------------------------------------------------------- #
# Audit integration
# --------------------------------------------------------------------------- #


def test_revenue_summary_writes_audit_event_when_log_supplied(
    sample_frame: pd.DataFrame, analyst: Actor, audit_log: AuditLog
) -> None:
    revenue_summary(sample_frame, actor=analyst, audit_log=audit_log)
    events = audit_log.read(limit=10)
    assert len(events) == 1
    row = events[0]
    assert row.action == AuditAction.PROTECTED_STORE_READ
    assert row.outcome == AuditOutcome.SUCCESS
    assert row.actor == "bob"
    assert row.resource == "revenue_summary"
    assert row.details["actor_role"] == "analyst"
    assert row.details["gross_revenue"] == 90.0
    assert row.details["orders"] == 3


def test_no_audit_event_when_log_not_supplied(
    sample_frame: pd.DataFrame,
    audit_log: AuditLog,
) -> None:
    revenue_summary(sample_frame)

    assert len(audit_log.read(limit=1)) == 0


def test_all_public_functions_emit_one_event_each(
    sample_frame: pd.DataFrame, analyst: Actor, audit_log: AuditLog
) -> None:
    revenue_summary(sample_frame, actor=analyst, audit_log=audit_log)
    product_metrics(sample_frame, actor=analyst, audit_log=audit_log)
    time_series(sample_frame, actor=analyst, audit_log=audit_log)
    country_metrics(sample_frame, actor=analyst, audit_log=audit_log)
    repeat_rate(sample_frame, actor=analyst, audit_log=audit_log)

    events = audit_log.read(limit=20)
    assert len(events) == 5
    resources = [row.resource for row in events]
    assert set(resources) == {
        "revenue_summary",
        "product_metrics",
        "time_series",
        "country_metrics",
        "repeat_rate",
    }
    for row in events:
        assert row.action == AuditAction.PROTECTED_STORE_READ
        assert row.outcome == AuditOutcome.SUCCESS
        assert row.actor == "bob"


def test_audit_actor_defaults_to_pipeline_when_actor_omitted(
    sample_frame: pd.DataFrame, audit_log: AuditLog
) -> None:
    revenue_summary(sample_frame, audit_log=audit_log)
    row = audit_log.read(limit=1)[0]
    assert row.actor == "pipeline"
    assert row.details["actor_role"] == "pipeline"
