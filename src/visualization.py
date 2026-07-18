"""M5 - Visualization and explanation.

Renders M3 metric outputs and M4 segmentation results as labeled Plotly figures. Each chart is
returned as an ExplainedChart that includes the figure, the formula behind it, applied filters,
and excluded row counts.

Design notes:

* Chart helpers accept the return shapes produced by src.analytics. They do not recompute metrics.
* M5 has no audit surface. It renders already-audited data, so auditing chart construction would
  double-count usage.
* Helpers return ExplainedChart rather than bare Plotly figures so charts stay paired with their
  explanations.
* Formula text is written for non-engineer viewers and avoids code fragments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final, Mapping

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class VisualizationError(Exception):
    """Raised when a chart helper receives a malformed metric payload."""


class ChartKind(str, Enum):
    """Enumeration of the M5 chart surfaces.

    Inherits from str so chart types serialize cleanly without manually unwrapping .value.
    """

    REVENUE_BY_MONTH = "revenue_by_month"
    REVENUE_BY_WEEKDAY = "revenue_by_weekday"
    TOP_PRODUCTS_BAR = "top_products_bar"
    COUNTRY_BAR = "country_bar"
    REPEAT_RATE_GAUGE = "repeat_rate_gauge"
    RFM_SCATTER = "rfm_scatter"


@dataclass(frozen=True)
class ExplainedChart:
    """Plotly figure bundled with its explanation.

    Attributes:
        kind: M5 chart surface that produced this record.
        figure: Plotly figure ready for st.plotly_chart.
        formula: Plain-English statement of the computation.
        filters: Applied filter conditions by name.
        exclusions: Row counts excluded from the chart, keyed by cause.
        row_count: Number of rows behind the visualization after filters and exclusions.

    The presentation layer should display formula, filters, and exclusions alongside the chart.
    """

    kind: ChartKind
    figure: go.Figure
    formula: str
    filters: Mapping[str, str] = field(default_factory=dict)
    exclusions: Mapping[str, int] = field(default_factory=dict)
    row_count: int = 0


# --------------------------------------------------------------------------- #
# Design constants
# --------------------------------------------------------------------------- #

# pandas uses Monday=0 for weekday numbers.
_WEEKDAY_LABELS: Final[tuple[str, ...]] = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)

# Shared chart colors from the design foundation.
_ACCENT: Final[str] = "#20808D"
_NEUTRAL: Final[str] = "#7A7974"

# Plotly template used for readable default charts.
_TEMPLATE: Final[str] = "plotly_white"


# --------------------------------------------------------------------------- #
# Chart helpers - one per M3/M4 surface
# --------------------------------------------------------------------------- #


def revenue_by_month(
    time_payload: Mapping[str, Any],
    *,
    revenue_payload: Mapping[str, Any] | None = None,
) -> ExplainedChart:
    """Render the monthly revenue line chart from time_series output.

    Args:
        time_payload: Output from src.analytics.time_series. Must include monthly with month,
            revenue, and orders columns.
        revenue_payload: Optional output from src.analytics.revenue_summary. When supplied,
            adjustments and returns populate the exclusion record.

    Returns:
        ExplainedChart labeled REVENUE_BY_MONTH.
    """

    monthly = _require_frame(time_payload, "monthly", "time_series")
    figure = _make_line(
        monthly,
        x="month",
        y="revenue",
        title="Monthly revenue",
        y_axis_title="Net revenue",
        x_axis_title="Month",
    )
    return ExplainedChart(
        kind=ChartKind.REVENUE_BY_MONTH,
        figure=figure,
        formula=(
            "Monthly net revenue is the sum of Revenue per calendar month, "
            "with adjustment rows excluded and return rows netted in."
        ),
        filters={
            "adjustments": "excluded",
            "returns": "netted into monthly totals",
            "missing_dates": "excluded",
        },
        exclusions=_revenue_exclusions(revenue_payload),
        row_count=int(len(monthly)),
    )


def revenue_by_weekday(
    time_payload: Mapping[str, Any],
    *,
    revenue_payload: Mapping[str, Any] | None = None,
) -> ExplainedChart:
    """Render the weekday revenue bar chart from time_series output."""

    weekday = _require_frame(time_payload, "weekday", "time_series")
    # Add weekday names for readable hover labels.
    labeled = weekday.copy()
    labeled["day"] = labeled["weekday"].map(
        {i: _WEEKDAY_LABELS[i] for i in range(len(_WEEKDAY_LABELS))}
    )
    figure = _make_bar(
        labeled,
        x="day",
        y="revenue",
        title="Revenue by weekday",
        y_axis_title="Net revenue",
        x_axis_title="Day of week",
    )
    return ExplainedChart(
        kind=ChartKind.REVENUE_BY_WEEKDAY,
        figure=figure,
        formula=(
            "Weekday net revenue is the sum of Revenue per day of week, "
            "with adjustment rows excluded and return rows netted in."
        ),
        filters={
            "adjustments": "excluded",
            "returns": "netted into daily totals",
            "missing_dates": "excluded",
        },
        exclusions=_revenue_exclusions(revenue_payload),
        row_count=int(len(labeled)),
    )


def top_products_bar(
    product_payload: Mapping[str, Any],
    *,
    metric: str = "revenue",
) -> ExplainedChart:
    """Render the top products bar chart from product_metrics output.

    Args:
        product_payload: Output from src.analytics.product_metrics.
        metric: Ranking to plot. Must be revenue or quantity.

    Returns:
        ExplainedChart labeled TOP_PRODUCTS.
    """

    if metric not in {"revenue", "quantity"}:
        raise VisualizationError("metric must be 'revenue' or 'quantity'; got " + repr(metric))
    key = "top_by_revenue" if metric == "revenue" else "top_by_quantity"
    top = _require_frame(product_payload, key, "product_metrics")

    # Sort ascending so the highest-ranked products appear at the top of the horizontal bar chart.
    value_column = "Revenue" if metric == "revenue" else "Quantity"
    labels = top.sort_values(value_column, ascending=True)
    figure = _make_hbar(
        labels,
        y="StockCode",
        x=value_column,
        title=f"Top products by {metric}",
        x_axis_title=("Revenue" if metric == "revenue" else "Quantity"),
        y_axis_title="Stock code",
    )
    return ExplainedChart(
        kind=ChartKind.TOP_PRODUCTS_BAR,
        figure=figure,
        formula=(
            f"Top products by {metric} rank SKUs on positive-revenue rows only; "
            "returns are counted separately in the return-rate table and "
            "adjustments are excluded."
        ),
        filters={
            "adjustments": "excluded",
            "returns": "excluded from ranking",
        },
        exclusions={},
        row_count=int(len(labels)),
    )


def country_bar(country_payload: pd.DataFrame) -> ExplainedChart:
    """Render the per-country revenue bar chart from country_metrics."""

    if not isinstance(country_payload, pd.DataFrame):
        raise VisualizationError("country_bar expects a DataFrame from country_metrics")
    required = {"Country", "revenue", "orders"}
    missing = required.difference(country_payload.columns)
    if missing:
        raise VisualizationError(
            "country_metrics frame missing required columns: " + ", ".join(sorted(missing))
        )

    # Limit the chart to 15 countries so labels stay legible. Remaining countries are reported as
    # other markets in the exclusion note.
    max_bars = 15
    ranked = country_payload.sort_values("revenue", ascending=False)
    hidden = max(0, len(ranked) - max_bars)
    top = ranked.head(max_bars).sort_values("revenue", ascending=True)

    figure = _make_hbar(
        top,
        y="Country",
        x="revenue",
        title="Revenue by country",
        x_axis_title="Net revenue",
        y_axis_title="Country",
    )
    return ExplainedChart(
        kind=ChartKind.COUNTRY_BAR,
        figure=figure,
        formula=(
            "Per-country net revenue is the sum of Revenue grouped by Country, "
            "with adjustment rows excluded and return rows netted in. "
            "Rows with a missing country drop out of the aggregation."
        ),
        filters={
            "adjustments": "excluded",
            "returns": "netted into country totals",
            "missing_country": "excluded",
        },
        exclusions={"lower_ranked_countries": hidden},
        row_count=int(len(top)),
    )


def repeat_rate_gauge(repeat_payload: Mapping[str, Any]) -> ExplainedChart:
    """Render the repeat-rate KPI as a Plotly indicator.

    repeat_rate values of None are shown as a zero-value gauge with an explanatory caption, so the
    presentation layer does not need special handling.
    """

    _require_keys(repeat_payload, {"customers", "repeat_customers", "repeat_rate"}, "repeat_rate")

    customers = int(repeat_payload["customers"])
    rate = repeat_payload["repeat_rate"]
    display_rate = 0.0 if rate is None else float(rate)

    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(display_rate * 100, 2),
            number={"suffix": "%", "font": {"size": 44}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": _ACCENT},
                "bgcolor": "#F7F6F2",
                "borderwidth": 0,
            },
            title={"text": "Repeat customer rate"},
        )
    )
    figure.update_layout(
        template=_TEMPLATE,
        margin=dict(l=40, r=40, t=60, b=20),
        height=280,
    )
    return ExplainedChart(
        kind=ChartKind.REPEAT_RATE_GAUGE,
        figure=figure,
        formula=(
            "Repeat rate is the share of identified customers who placed two or more "
            "positive orders."
        ),
        filters={
            "returns": "excluded",
            "guest_checkouts": "excluded",
        },
        exclusions={"unidentified_customers": _guest_count(repeat_payload)},
        row_count=customers,
    )


def rfm_scatter(segments: pd.DataFrame) -> ExplainedChart:
    """Render RFM segments as a monetary-versus-recency scatter plot.

    Points are colored by cluster, and frequency is encoded as marker size so all three RFM
    dimensions appear in one chart. Empty input returns an empty chart with row_count set to zero.
    """

    if not isinstance(segments, pd.DataFrame):
        raise VisualizationError("rfm_scatter expects the DataFrame returned by rfm_segments")
    required = {"CustomerID", "recency", "frequency", "monetary", "cluster"}
    missing = required.difference(segments.columns)
    if missing:
        raise VisualizationError(
            "rfm_segments frame missing required columns: " + ", ".join(sorted(missing))
        )

    if segments.empty:
        figure = go.Figure()
        figure.update_layout(
            template=_TEMPLATE,
            title="RFM segments (no data)",
            xaxis_title="Recency (days)",
            yaxis_title="Monetary",
            height=420,
        )
    else:
        figure = px.scatter(
            segments,
            x="recency",
            y="monetary",
            color=segments["cluster"].astype(str),
            size="frequency",
            hover_data={"CustomerID": True, "cluster": True, "frequency": True},
            title="RFM segments",
            labels={
                "recency": "Recency (days)",
                "monetary": "Monetary",
                "color": "Cluster",
            },
            template=_TEMPLATE,
        )
        figure.update_layout(height=420, legend_title_text="Cluster")

    cluster_count = int(segments["cluster"].nunique()) if not segments.empty else 0
    return ExplainedChart(
        kind=ChartKind.RFM_SCATTER,
        figure=figure,
        formula=(
            "RFM segments group customers by recency, frequency, and monetary value. "
            "The chart plots recency against monetary value, sizes markers by frequency, "
            "and colors markers by segment."
        ),
        filters={
            "guest_checkouts": "excluded from clustering",
            "adjustments": "excluded from monetary",
            "returns": "netted into monetary",
        },
        exclusions={"small_group_clusters": 0, "clusters_rendered": cluster_count},
        row_count=int(len(segments)),
    )


# --------------------------------------------------------------------------- #
# Internal figure builders
# --------------------------------------------------------------------------- #


def _make_line(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    title: str,
    x_axis_title: str,
    y_axis_title: str,
) -> go.Figure:
    figure = px.line(
        frame,
        x=x,
        y=y,
        title=title,
        template=_TEMPLATE,
        markers=True,
    )
    figure.update_traces(line_color=_ACCENT, marker_color=_ACCENT)
    figure.update_layout(
        xaxis_title=x_axis_title,
        yaxis_title=y_axis_title,
        height=380,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return figure


def _make_bar(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    title: str,
    x_axis_title: str,
    y_axis_title: str,
) -> go.Figure:
    figure = px.bar(
        frame,
        x=x,
        y=y,
        title=title,
        template=_TEMPLATE,
    )
    figure.update_traces(marker_color=_ACCENT)
    figure.update_layout(
        xaxis_title=x_axis_title,
        yaxis_title=y_axis_title,
        height=380,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return figure


def _make_hbar(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    title: str,
    x_axis_title: str,
    y_axis_title: str,
) -> go.Figure:
    figure = px.bar(
        frame,
        x=x,
        y=y,
        orientation="h",
        title=title,
        template=_TEMPLATE,
    )
    figure.update_traces(marker_color=_ACCENT)
    figure.update_layout(
        xaxis_title=x_axis_title,
        yaxis_title=y_axis_title,
        height=max(320, 24 * len(frame) + 120),
        margin=dict(l=80, r=20, t=60, b=40),
    )
    return figure


# --------------------------------------------------------------------------- #
# Payload validation helpers
# --------------------------------------------------------------------------- #


def _require_frame(payload: Mapping[str, Any], key: str, source: str) -> pd.DataFrame:
    """Return payload[key] or raise a descriptive VisualizationError."""

    if not isinstance(payload, Mapping):
        raise VisualizationError(
            f"{source} payload must be a mapping; got {type(payload).__name__}"
        )
    if key not in payload:
        raise VisualizationError(f"{source} payload is missing required key {key!r}")
    value = payload[key]
    if not isinstance(value, pd.DataFrame):
        raise VisualizationError(
            f"{source} payload[{key!r}] must be a DataFrame; got {type(value).__name__}"
        )
    return value


def _require_keys(payload: Mapping[str, Any], required: set[str], source: str) -> None:
    if not isinstance(payload, Mapping):
        raise VisualizationError(
            f"{source} payload must be a mapping; got {type(payload).__name__}"
        )
    missing = required.difference(payload.keys())
    if missing:
        raise VisualizationError(
            f"{source} payload missing required keys: " + ", ".join(sorted(missing))
        )


def _revenue_exclusions(
    revenue_payload: Mapping[str, Any] | None,
) -> Mapping[str, int]:
    """Return revenue exclusions from a revenue_summary payload.

    Returns an empty mapping when no revenue payload is supplied.
    """

    if revenue_payload is None:
        return {}
    exclusions: dict[str, int] = {}
    if "adjustments" in revenue_payload:
        exclusions["adjustments"] = int(revenue_payload["adjustments"])
    # returns_value is monetary impact, not a row count.
    if "returns_value" in revenue_payload:
        exclusions["returns_value"] = int(round(float(revenue_payload["returns_value"])))
    return exclusions


def _guest_count(repeat_payload: Mapping[str, Any]) -> int:
    """Return unidentified customer count from repeat payload."""

    raw = repeat_payload.get("unidentified_customers", 0)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0
