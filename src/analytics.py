"""M3 Metrics and KPI, and M4 Segmentation analytics.

Hosts the analytics services declared in the architecture diagram. M3 is implemented here. M4 keeps
a locked stub signature so the segmentation branch can replace it without changing M3.

Design notes:

* Public M3 functions accept cleaned M10 DataFrames and validate the required analytics columns
  before computing results.
* Missing required columns raise AnalyticsError.
* Return values are plain dictionaries and pandas DataFrames. This module has no Streamlit or
  storage coupling.
* Revenue summaries report both gross revenue and net revenue. Gross revenue uses positive
  revenue rows and excludes adjustments. Net revenue includes returns and excludes adjustments.
* Empty inputs return zero-filled results with AOV set to None.
* When an AuditLog is supplied, each public metric call writes one PROTECTED_STORE_READ
  event so metric use can be traced.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Final

import pandas as pd

from src.logs import AuditAction, AuditEvent, AuditLog, AuditOutcome

if TYPE_CHECKING:
    from src.access import Actor


# --------------------------------------------------------------------------- #
# Contracts and constants
# --------------------------------------------------------------------------- #

_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "InvoiceNo",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "CustomerID",
        "Country",
        "IsReturn",
        "IsAdjustment",
        "Revenue",
    }
)

_DEFAULT_TOP_N: Final[int] = 10


class AnalyticsError(Exception):
    """Raised when an analytics function receives a malformed frame."""


# --------------------------------------------------------------------------- #
# Frame validation
# --------------------------------------------------------------------------- #


def _require_columns(frame: pd.DataFrame) -> None:
    """Fail fast if the cleaned-frame contract is violated."""

    if not isinstance(frame, pd.DataFrame):
        raise AnalyticsError("analytics functions require a pandas DataFrame")

    missing = _REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        # Sort for deterministic error messages that read cleanly in a diff.
        missing_list = ", ".join(sorted(missing))
        raise AnalyticsError(f"missing required columns: {missing_list}")


def _non_adjustment(frame: pd.DataFrame) -> pd.DataFrame:
    """Return rows that are not adjustment lines.

    Adjustments are excluded from revenue metrics. Returns are kept so they can subtract from
    net revenue and be reported separately.
    """

    mask = ~frame["IsAdjustment"].fillna(False).astype(bool)
    return frame.loc[mask]


# --------------------------------------------------------------------------- #
# M3.1 - Headline revenue summary
# --------------------------------------------------------------------------- #


def revenue_summary(
    frame: pd.DataFrame,
    *,
    actor: "Actor | None" = None,
    audit_log: AuditLog | None = None,
) -> dict[str, Any]:
    """Return headline revenue KPIs for a cleaned frame.

    Args:
        frame: A cleaned DataFrame.
        actor: Actor recorded on the audit event.
        audit_log: Optional audit log. When supplied, one PROTECTED_STORE_READ event is written.

    Returns:
        Dictionary containing gross_revenue, net_revenue, returns_value, orders, aov
        (average order value), line_items, and adjustments. aov is None when there are no orders.
        Numeric values are converted to native Python types.
    """

    _require_columns(frame)

    non_adj = _non_adjustment(frame)
    positives = non_adj.loc[~non_adj["IsReturn"].fillna(False).astype(bool)]
    returns = non_adj.loc[non_adj["IsReturn"].fillna(False).astype(bool)]

    gross_revenue = _to_float(positives["Revenue"].sum(skipna=True))
    returns_value = _to_float(returns["Revenue"].sum(skipna=True))
    net_revenue = gross_revenue + returns_value  # returns_value is <= 0

    # Order count is distinct positive invoices (returns are refunds against prior orders,
    # not new orders).
    orders = int(positives["InvoiceNo"].nunique(dropna=True))
    aov: float | None = round(gross_revenue / orders, 2) if orders else None

    result = {
        "gross_revenue": round(gross_revenue, 2),
        "net_revenue": round(net_revenue, 2),
        "returns_value": round(returns_value, 2),
        "orders": orders,
        "aov": aov,
        "line_items": int(len(non_adj)),
        "adjustments": int(len(frame) - len(non_adj)),
    }
    _audit(audit_log, actor=actor, resource="revenue_summary", details=result)
    return result


# --------------------------------------------------------------------------- #
# M3.2 - Product metrics
# --------------------------------------------------------------------------- #


def product_metrics(
    frame: pd.DataFrame,
    *,
    top_n: int = _DEFAULT_TOP_N,
    actor: "Actor | None" = None,
    audit_log: AuditLog | None = None,
) -> dict[str, Any]:
    """Return top product rankings and return rates.

    Args:
        frame: A cleaned DataFrame.
        top_n: Number of rows in each ranking.
        actor: Actor recorded on the audit event.
        audit_log: Optional audit log.

    Returns:
        Dictionary containing top_by_revenue, top_by_quantity, and return_rate.
        Product rankings are DataFrames with StockCode, Description, Revenue, and Quantity.
        return_rate contains StockCode and return_rate.

    Raises:
        ValueError: If top_n is not positive.
    """

    _require_columns(frame)
    if top_n <= 0:
        raise AnalyticsError("top_n must be a positive integer")

    non_adj = _non_adjustment(frame).copy()
    if non_adj.empty:
        empty_top = pd.DataFrame(columns=["StockCode", "Description", "Revenue", "Quantity"])
        empty_returns = pd.DataFrame(columns=["StockCode", "return_rate"])
        result = {
            "top_by_revenue": empty_top,
            "top_by_quantity": empty_top.copy(),
            "return_rate": empty_returns,
        }
        _audit(audit_log, actor=actor, resource="product_metrics", details={"top_n": top_n})
        return result

    # Positive rows drive top-N rankings. Returns are refund credits, not sales, so they are
    # excluded from top-product calculations.
    positives = non_adj.loc[~non_adj["IsReturn"].fillna(False).astype(bool)]

    # Capture the first non-null description for each StockCode. Some cleaned rows may have
    # blank descriptions when other rows for the product have one.
    description = positives.groupby("StockCode", dropna=True)["Description"].agg(
        lambda s: next((v for v in s if isinstance(v, str) and v), "")
    )

    grouped = positives.groupby("StockCode", dropna=True).agg(
        Revenue=("Revenue", "sum"),
        Quantity=("Quantity", "sum"),
    )
    grouped["Description"] = description
    grouped = grouped.reset_index()[["StockCode", "Description", "Revenue", "Quantity"]]

    top_by_revenue = (
        grouped.sort_values(["Revenue", "StockCode"], ascending=[False, True])
        .head(top_n)
        .reset_index(drop=True)
    )
    top_by_quantity = (
        grouped.sort_values(["Quantity", "StockCode"], ascending=[False, True])
        .head(top_n)
        .reset_index(drop=True)
    )

    # Return rate: absolute returned quantity divided by sold quantity.
    sold = positives.groupby("StockCode", dropna=True)["Quantity"].sum().astype("Float64")
    returned = (
        non_adj.loc[non_adj["IsReturn"].fillna(False).astype(bool)]
        .groupby("StockCode", dropna=True)["Quantity"]
        .sum()
        .abs()
        .astype("Float64")
    )
    aligned = sold.to_frame("sold").join(returned.rename("returned"), how="left").fillna(0)
    aligned["return_rate"] = (aligned["returned"] / aligned["sold"]).where(
        aligned["sold"] > 0, other=pd.NA
    )
    return_rate = (
        aligned.reset_index()[["StockCode", "return_rate"]]
        .sort_values("StockCode")
        .reset_index(drop=True)
    )

    result = {
        "top_by_revenue": top_by_revenue,
        "top_by_quantity": top_by_quantity,
        "return_rate": return_rate,
    }
    _audit(
        audit_log,
        actor=actor,
        resource="product_metrics",
        details={"top_n": top_n, "distinct_products": int(len(grouped))},
    )
    return result


# --------------------------------------------------------------------------- #
# M3.3 - Time series
# --------------------------------------------------------------------------- #


def time_series(
    frame: pd.DataFrame,
    *,
    actor: "Actor | None" = None,
    audit_log: AuditLog | None = None,
) -> dict[str, Any]:
    """Return revenue and order counts aggregated over time.

    Args:
        frame: A cleaned DataFrame.
        actor: Optional actor for audit.
        audit_log: Optional audit log.

    Returns:
        Dictionary containing monthly and weekday DataFrames. monthly contains month, revenue, and
        orders columns. weekday contains weekday and revenue columns ordered from Monday through
        Sunday.
    """

    _require_columns(frame)

    non_adj = _non_adjustment(frame).copy()
    # Coerce dates before dropping rows. Missing dates drop those rows from the time series but
    # should not raise. Cleaning already coerces InvoiceDate.
    dated = non_adj.dropna(subset=["InvoiceDate"]).copy()
    if dated.empty:
        result = {
            "monthly": pd.DataFrame(columns=["month", "revenue", "orders"]),
            "weekday": pd.DataFrame(columns=["weekday", "revenue"]),
        }
        _audit(audit_log, actor=actor, resource="time_series", details={"rows": 0})
        return result

    dated["_month"] = dated["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
    dated["_weekday"] = dated["InvoiceDate"].dt.weekday.astype("Int8")

    monthly_revenue = dated.groupby("_month")["Revenue"].sum().astype(float)
    positives = dated.loc[~dated["IsReturn"].fillna(False).astype(bool)]
    monthly_orders = (
        positives.groupby(positives["InvoiceDate"].dt.to_period("M").dt.to_timestamp())["InvoiceNo"]
        .nunique()
        .astype("int64")
    )
    monthly = (
        monthly_revenue.to_frame("revenue")
        .join(monthly_orders.rename("orders"), how="left")
        .fillna({"orders": 0})
        .reset_index()
        .rename(columns={"_month": "month"})
    )
    monthly["orders"] = monthly["orders"].astype("int64")
    monthly["revenue"] = monthly["revenue"].round(2)

    weekday = (
        dated.groupby("_weekday")["Revenue"]
        .sum()
        .reindex(range(7), fill_value=0.0)
        .astype(float)
        .round(2)
        .reset_index()
        .rename(columns={"_weekday": "weekday", "Revenue": "revenue"})
    )
    weekday["weekday"] = weekday["weekday"].astype("int64")

    result = {"monthly": monthly, "weekday": weekday}
    _audit(
        audit_log,
        actor=actor,
        resource="time_series",
        details={"months": int(len(monthly)), "rows": int(len(dated))},
    )
    return result


# --------------------------------------------------------------------------- #
# M3.4 - Country metrics
# --------------------------------------------------------------------------- #


def country_metrics(
    frame: pd.DataFrame,
    *,
    actor: "Actor | None" = None,
    audit_log: AuditLog | None = None,
) -> pd.DataFrame:
    """Return net revenue and order counts per country, sorted by revenue."""

    _require_columns(frame)

    non_adj = _non_adjustment(frame).copy()
    if non_adj.empty:
        empty = pd.DataFrame(columns=["Country", "revenue", "orders"])
        _audit(audit_log, actor=actor, resource="country_metrics", details={"countries": 0})
        return empty

    revenue = non_adj.groupby("Country", dropna=True)["Revenue"].sum().astype(float)
    positives = non_adj.loc[~non_adj["IsReturn"].fillna(False).astype(bool)]
    orders = positives.groupby("Country", dropna=True)["InvoiceNo"].nunique().astype("int64")
    result = (
        revenue.to_frame("revenue")
        .join(orders.rename("orders"), how="left")
        .fillna({"orders": 0})
        .reset_index()
    )
    result["orders"] = result["orders"].astype("int64")
    result["revenue"] = result["revenue"].round(2)
    result = result.sort_values(["revenue", "Country"], ascending=[False, True]).reset_index(
        drop=True
    )

    _audit(
        audit_log,
        actor=actor,
        resource="country_metrics",
        details={"countries": int(len(result))},
    )
    return result


# --------------------------------------------------------------------------- #
# M3.5 - Repeat purchase rate
# --------------------------------------------------------------------------- #


def repeat_rate(
    frame: pd.DataFrame,
    *,
    actor: "Actor | None" = None,
    audit_log: AuditLog | None = None,
) -> dict[str, Any]:
    """Return the fraction of customers with more than one positive order.

    Args:
        frame: A cleaned DataFrame. Rows without CustomerID are excluded because repeat rate
            is undefined for guest orders.
        actor: Optional actor for audit.
        audit_log: Optional audit log.

    Returns:
        Dictionary containing customers, repeat_customers, and repeat_rate. repeat_rate is None
        when there are no identified customers.
    """

    _require_columns(frame)

    non_adj = _non_adjustment(frame)
    positives = non_adj.loc[~non_adj["IsReturn"].fillna(False).astype(bool)]
    identified = positives.dropna(subset=["CustomerID"])
    if identified.empty:
        result_empty: dict[str, Any] = {
            "customers": 0,
            "repeat_customers": 0,
            "repeat_rate": None,
        }
        _audit(audit_log, actor=actor, resource="repeat_rate", details=result_empty)
        return result_empty

    orders_per_customer = identified.groupby("CustomerID")["InvoiceNo"].nunique()
    customers = int(len(orders_per_customer))
    repeat_customers = int((orders_per_customer >= 2).sum())
    rate = round(repeat_customers / customers, 4) if customers else None

    result = {
        "customers": customers,
        "repeat_customers": repeat_customers,
        "repeat_rate": rate,
    }
    _audit(audit_log, actor=actor, resource="repeat_rate", details=result)
    return result


# --------------------------------------------------------------------------- #
# M4 - Segmentation (stub, unchanged)
# --------------------------------------------------------------------------- #


def rfm_segments(frame: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    """Compute RFM scores and K-Means segments.

    Returns:
        DataFrame with CustomerID, recency, frequency, monetary, and cluster columns.
        Clusters below the small-group threshold are removed.
        Implemented in feature/m4-segmentation.
    """

    raise NotImplementedError("Implemented in feature/m4-segmentation")


# --------------------------------------------------------------------------- #
# Audit + coercion helpers
# --------------------------------------------------------------------------- #


def _audit(
    audit_log: AuditLog | None,
    *,
    actor: "Actor | None",
    resource: str,
    details: dict[str, Any],
) -> None:
    """Write a single PROTECTED_STORE_READ event when an audit log is bound."""

    if audit_log is None:
        return

    # DataFrames are useful in return values but noisy in the audit journal; keep the recorded
    # details lightweight and JSON-serializable.
    serializable: dict[str, Any] = {
        "actor_role": actor.role if actor is not None else "pipeline",
    }
    for key, value in details.items():
        if isinstance(value, (int, float, str)) or value is None:
            serializable[key] = value

    audit_log.append(
        AuditEvent(
            timestamp=datetime.now(timezone.utc),
            actor=actor.username if actor is not None else "pipeline",
            action=AuditAction.PROTECTED_STORE_READ,
            outcome=AuditOutcome.SUCCESS,
            resource=resource,
            details=serializable,
        )
    )


def _to_float(value: Any) -> float:
    """Coerce numpy / nullable scalars to native Python floats.

    ``DataFrame.sum`` on an empty or all-NA column returns ``0`` in some
    dtypes and ``NA`` in others. Presentation code expects a plain ``0.0``
    for the "no revenue yet" case.
    """

    if value is None or pd.isna(value):
        return 0.0
    return float(value)


__all__ = [
    "AnalyticsError",
    "country_metrics",
    "product_metrics",
    "repeat_rate",
    "revenue_summary",
    "rfm_segments",
    "time_series",
]
