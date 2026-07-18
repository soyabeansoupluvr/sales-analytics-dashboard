"""Tests for the M5 visualization module.

Covers the six chart helpers and their ExplainedChart payloads. Tests verify
figure shape, explanation text, exclusions, malformed payload handling, and
the pure-rendering contract.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import pytest

from src.analytics import (
    country_metrics,
    product_metrics,
    repeat_rate,
    revenue_summary,
    time_series,
)
from src.visualization import (
    ChartKind,
    ExplainedChart,
    VisualizationError,
    country_bar,
    repeat_rate_gauge,
    revenue_by_month,
    revenue_by_weekday,
    rfm_scatter,
    top_products_bar,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _cleaned_row(**overrides: object) -> dict[str, object]:
    """Row conforming to the M10 output contract."""

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
    """A realistic mix: two countries, several SKUs, one return, one adjustment."""

    rows = []
    # Positive activity across three months for four SKUs and two countries.
    for i in range(1, 25):
        rows.append(
            _cleaned_row(
                InvoiceNo=f"INV{i:03d}",
                StockCode=f"SKU{i % 4:02d}",
                Quantity=i,
                InvoiceDate=pd.Timestamp("2024-01-01") + pd.Timedelta(days=i * 3),
                CustomerID=f"pseudo-{i % 6:03d}",
                Country="UK" if i % 3 else "France",
                Revenue=float(i * 10),
            )
        )
    # One return and one adjustment to exercise the exclusion counters.
    rows.append(
        _cleaned_row(
            InvoiceNo="R001",
            StockCode="SKU01",
            Quantity=-2,
            InvoiceDate=pd.Timestamp("2024-02-15"),
            CustomerID="pseudo-001",
            Country="UK",
            IsReturn=True,
            Revenue=-40.0,
        )
    )
    rows.append(
        _cleaned_row(
            InvoiceNo="ADJ01",
            StockCode="BANK",
            Description="Bank charge",
            Quantity=1,
            InvoiceDate=pd.Timestamp("2024-02-20"),
            CustomerID=None,
            Country="UK",
            IsAdjustment=True,
            Revenue=25.0,
        )
    )
    frame = pd.DataFrame(rows)
    frame["InvoiceDate"] = pd.to_datetime(frame["InvoiceDate"])
    return frame


@pytest.fixture
def rfm_frame() -> pd.DataFrame:
    """A synthetic RFM output as returned by rfm_segments."""

    return pd.DataFrame(
        {
            "CustomerID": [f"pseudo-{i:03d}" for i in range(1, 13)],
            "recency": [1, 5, 30, 60, 90, 120, 150, 200, 5, 45, 75, 15],
            "frequency": [8, 6, 5, 4, 3, 2, 2, 1, 7, 3, 2, 6],
            "monetary": [
                1200.0,
                900.0,
                400.0,
                350.0,
                200.0,
                180.0,
                150.0,
                90.0,
                1100.0,
                300.0,
                220.0,
                800.0,
            ],
            "cluster": [0, 0, 1, 1, 2, 2, 2, 3, 0, 1, 2, 0],
        }
    )


# --------------------------------------------------------------------------- #
# revenue_by_month
# --------------------------------------------------------------------------- #


def test_revenue_by_month_returns_explained_chart(sample_frame: pd.DataFrame) -> None:
    payload = time_series(sample_frame)
    chart = revenue_by_month(payload)
    assert isinstance(chart, ExplainedChart)
    assert chart.kind is ChartKind.REVENUE_BY_MONTH
    assert isinstance(chart.figure, go.Figure)
    assert len(chart.figure.data) == 1
    assert chart.formula.startswith("Monthly net revenue")
    assert chart.filters["adjustments"] == "excluded"
    assert chart.row_count == len(payload["monthly"])


def test_revenue_by_month_uses_revenue_payload_for_exclusions(
    sample_frame: pd.DataFrame,
) -> None:
    ts = time_series(sample_frame)
    rev = revenue_summary(sample_frame)
    chart = revenue_by_month(ts, revenue_payload=rev)
    # sample_frame has 1 adjustment row and 1 return (returns_value = -40).
    assert chart.exclusions["adjustments"] == 1
    assert chart.exclusions["returns_value"] == -40


def test_revenue_by_month_rejects_bad_payload() -> None:
    with pytest.raises(VisualizationError, match="missing required key"):
        revenue_by_month({"weekday": pd.DataFrame()})


# --------------------------------------------------------------------------- #
# revenue_by_weekday
# --------------------------------------------------------------------------- #


def test_revenue_by_weekday_labels_days(sample_frame: pd.DataFrame) -> None:
    payload = time_series(sample_frame)
    chart = revenue_by_weekday(payload)
    assert chart.kind is ChartKind.REVENUE_BY_WEEKDAY
    # Bar chart always has 7 categories, even when some weekdays are zero.
    assert chart.row_count == 7
    # x tick labels are weekday names, not integers.
    x_values = list(chart.figure.data[0].x)
    assert "Monday" in x_values
    assert "Sunday" in x_values


def test_revenue_by_weekday_rejects_non_mapping() -> None:
    with pytest.raises(VisualizationError, match="must be a mapping"):
        revenue_by_weekday([1, 2, 3])  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# top_products_bar
# --------------------------------------------------------------------------- #


def test_top_products_bar_by_revenue(sample_frame: pd.DataFrame) -> None:
    payload = product_metrics(sample_frame, top_n=5)
    chart = top_products_bar(payload, metric="revenue")
    assert chart.kind is ChartKind.TOP_PRODUCTS_BAR
    assert chart.figure.data[0].orientation == "h"
    assert "revenue" in chart.formula.lower()


def test_top_products_bar_by_quantity(sample_frame: pd.DataFrame) -> None:
    payload = product_metrics(sample_frame, top_n=5)
    chart = top_products_bar(payload, metric="quantity")
    assert "quantity" in chart.formula.lower()


def test_top_products_bar_rejects_unknown_metric(sample_frame: pd.DataFrame) -> None:
    payload = product_metrics(sample_frame)
    with pytest.raises(VisualizationError, match="metric must be"):
        top_products_bar(payload, metric="profit")


# --------------------------------------------------------------------------- #
# country_bar
# --------------------------------------------------------------------------- #


def test_country_bar_sorts_by_revenue(sample_frame: pd.DataFrame) -> None:
    payload = country_metrics(sample_frame)
    chart = country_bar(payload)
    assert chart.kind is ChartKind.COUNTRY_BAR
    # Horizontal bars are drawn bottom-up ascending, so the last y label is the top-ranked country.
    y_values = list(chart.figure.data[0].y)
    assert y_values[-1] == "UK"


def test_country_bar_caps_at_fifteen_countries() -> None:
    frame = pd.DataFrame(
        {
            "Country": [f"C{i:02d}" for i in range(1, 22)],
            "revenue": list(range(2100, 0, -100)),
            "orders": [1] * 21,
        }
    )
    chart = country_bar(frame)
    assert chart.row_count == 15
    assert chart.exclusions["lower_ranked_countries"] == 6


def test_country_bar_rejects_non_dataframe() -> None:
    with pytest.raises(VisualizationError, match="expects a DataFrame"):
        country_bar({"Country": ["UK"], "revenue": [1.0], "orders": [1]})  # type: ignore[arg-type]


def test_country_bar_rejects_missing_columns() -> None:
    with pytest.raises(VisualizationError, match="missing required columns"):
        country_bar(pd.DataFrame({"Country": ["UK"], "revenue": [1.0]}))


# --------------------------------------------------------------------------- #
# repeat_rate_gauge
# --------------------------------------------------------------------------- #


def test_repeat_rate_gauge_renders_percentage(sample_frame: pd.DataFrame) -> None:
    payload = repeat_rate(sample_frame)
    chart = repeat_rate_gauge(payload)
    assert chart.kind is ChartKind.REPEAT_RATE_GAUGE
    # Indicator carries the value on data[0].value.
    assert 0 <= float(chart.figure.data[0].value) <= 100


def test_repeat_rate_gauge_handles_none_rate() -> None:
    chart = repeat_rate_gauge({"customers": 0, "repeat_customers": 0, "repeat_rate": None})
    assert float(chart.figure.data[0].value) == 0.0
    assert chart.row_count == 0


def test_repeat_rate_gauge_rejects_incomplete_payload() -> None:
    with pytest.raises(VisualizationError, match="missing required keys"):
        repeat_rate_gauge({"customers": 5})


# --------------------------------------------------------------------------- #
# rfm_scatter
# --------------------------------------------------------------------------- #


def test_rfm_scatter_encodes_three_dimensions(rfm_frame: pd.DataFrame) -> None:
    chart = rfm_scatter(rfm_frame)
    assert chart.kind is ChartKind.RFM_SCATTER
    assert len(chart.figure.data) == rfm_frame["cluster"].nunique()
    assert chart.row_count == len(rfm_frame)
    assert chart.exclusions["clusters_rendered"] == rfm_frame["cluster"].nunique()


def test_rfm_scatter_handles_empty_frame() -> None:
    empty = pd.DataFrame(columns=["CustomerID", "recency", "frequency", "monetary", "cluster"])
    chart = rfm_scatter(empty)
    assert chart.row_count == 0
    assert chart.exclusions["clusters_rendered"] == 0
    # Empty figures still carry an empty-state title.
    assert "no data" in chart.figure.layout.title.text.lower()


def test_rfm_scatter_rejects_non_dataframe() -> None:
    with pytest.raises(VisualizationError, match="expects the DataFrame"):
        rfm_scatter({"CustomerID": [], "cluster": []})  # type: ignore[arg-type]


def test_rfm_scatter_rejects_missing_columns() -> None:
    frame = pd.DataFrame({"CustomerID": ["a"], "recency": [1]})
    with pytest.raises(VisualizationError, match="missing required columns"):
        rfm_scatter(frame)


# --------------------------------------------------------------------------- #
# ExplainedChart / ChartKind contract
# --------------------------------------------------------------------------- #


def test_explained_chart_is_frozen(sample_frame: pd.DataFrame) -> None:
    """Frozen dataclass so callers cannot mutate the audit trail."""

    chart = revenue_by_month(time_series(sample_frame))
    with pytest.raises((AttributeError, TypeError)):
        chart.formula = "tampered"  # type: ignore[misc]


def test_chart_kind_serializes_to_string() -> None:
    """ChartKind is a str-mixin enum so payloads flatten cleanly."""

    assert ChartKind.REVENUE_BY_MONTH == "revenue_by_month"
    assert str(ChartKind.RFM_SCATTER.value) == "rfm_scatter"
