"""M3 Metrics and KPI, and M4 Segmentation analytics.

Hosts the analytics services declared in the architecture diagram. M3 produces per-metric
DataFrames and dictionaries from a cleaned M7 frame. M4 fits K-Means on the M3 RFM table and
left-merges a per-customer sensitivity frame so a single DataFrame carries both the segmentation
and its fairness posture.

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

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.access import enforce_small_group
from src.logs import AuditAction, AuditEvent, AuditLog, AuditOutcome

if TYPE_CHECKING:
    from src.access import Actor
    from src.config import Settings


# --------------------------------------------------------------------------- #
# Contracts and constants
# --------------------------------------------------------------------------- #

# Per-metric column requirements. Each metric declares the columns it
# actually reads from the cleaned frame. The presentation layer (M1 + M2)
# routes a role-scoped frame from M7 to a metric, so a viewer that lacks
# CustomerID must still be able to run revenue_summary, product_metrics,
# time_series, and country_metrics without a spurious missing-column
# error. rfm_segments and repeat_rate name CustomerID explicitly and will
# correctly refuse a viewer-scoped frame.
_REQUIRED_BY_METRIC: Final[dict[str, frozenset[str]]] = {
    "revenue_summary": frozenset(
        {"InvoiceNo", "IsReturn", "IsAdjustment", "Revenue"}
    ),
    "product_metrics": frozenset(
        {"StockCode", "Description", "Quantity", "IsReturn", "IsAdjustment", "Revenue"}
    ),
    "time_series": frozenset(
        {"InvoiceNo", "InvoiceDate", "IsReturn", "IsAdjustment", "Revenue"}
    ),
    "country_metrics": frozenset(
        {"InvoiceNo", "Country", "IsReturn", "IsAdjustment", "Revenue"}
    ),
    "repeat_rate": frozenset(
        {"InvoiceNo", "CustomerID", "IsReturn", "IsAdjustment"}
    ),
    "rfm_segments": frozenset(
        {"CustomerID", "InvoiceDate", "InvoiceNo", "IsReturn", "IsAdjustment", "Revenue"}
    ),
}

_DEFAULT_TOP_N: Final[int] = 10

# M4 constants. random_state keeps K-Means deterministic so audit records can be reproduced.
# n_init is explicit to avoid future-default changes in sklearn.
_DEFAULT_K: Final[int] = 4
_KMEANS_RANDOM_STATE: Final[int] = 20260716
_KMEANS_N_INIT: Final[int] = 10
_MIN_CUSTOMERS_FOR_CLUSTERING: Final[int] = 10
_RFM_COLUMNS: Final[tuple[str, ...]] = ("recency", "frequency", "monetary")

# M4 fairness sensitivity check. The holdout refit uses its own seed so the
# holdout partition is reproducible without disturbing the main fit's seed.
_HOLDOUT_RANDOM_STATE: Final[int] = 202607111528
_MIN_HOLDOUT_SIZE: Final[int] = 2
_MIN_TRAIN_FOR_HOLDOUT: Final[int] = _MIN_CUSTOMERS_FOR_CLUSTERING

# Sensitivity columns appended to the RFM output by a left join on CustomerID.
# Order matches the merge target so the final frame reads recency, frequency,
# monetary, cluster, then the five stability fields.
_STABILITY_COLUMNS: Final[tuple[str, ...]] = (
    "stability_score",
    "stability_holdout_size",
    "stability_threshold",
    "stability_flag",
    "stability_reason",
)


class AnalyticsError(Exception):
    """Raised when an analytics function receives a malformed frame."""


# --------------------------------------------------------------------------- #
# Frame validation
# --------------------------------------------------------------------------- #


def _require_columns(frame: pd.DataFrame, metric: str) -> None:
    """Fail fast if the cleaned-frame contract is violated for this metric.

    Each metric declares its own column requirements in _REQUIRED_BY_METRIC`. The error message
    names the metric so a viewer-scoped frame denied by rfm_segments produces a clear cause.
    """

    if not isinstance(frame, pd.DataFrame):
        raise AnalyticsError("analytics functions require a pandas DataFrame")

    required = _REQUIRED_BY_METRIC.get(metric)
    if required is None:
        raise AnalyticsError(f"unknown metric: {metric!r}")

    missing = required - set(frame.columns)
    if missing:
        # Sort for deterministic error messages that read cleanly in a diff.
        missing_list = ", ".join(sorted(missing))
        raise AnalyticsError(f"{metric} missing required columns: {missing_list}")


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

    _require_columns(frame, "revenue_summary")

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

    _require_columns(frame, "product_metrics")
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

    _require_columns(frame, "time_series")

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

    _require_columns(frame, "country_metrics")

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

    _require_columns(frame, "repeat_rate")

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
# M4 - RFM segmentation
# --------------------------------------------------------------------------- #


def rfm_segments(
    frame: pd.DataFrame,
    k: int = _DEFAULT_K,
    *,
    settings: "Settings | None" = None,
    threshold: int | None = None,
    actor: "Actor | None" = None,
    audit_log: AuditLog | None = None,
) -> pd.DataFrame:
    """Compute RFM scores and K-Means segments with a fairness-sensitivity join.

    Args:
        frame: Cleaned and pseudonymized DataFrame from M7 output.
        k: Number of clusters to fit. Must be at least 2 and less than the number of identified
            customers.
        settings: Application settings. Required unless threshold is supplied.
        threshold: Explicit small-group threshold override.
        actor: Actor recorded on audit events.
        audit_log: Optional audit log. When supplied, the segmentation run is audited, and
            suppressed clusters are audited as access denials.

    Returns:
        DataFrame with RFM, cluster, and stability columns. The output includes CustomerID,
        recency, frequency, monetary, cluster, stability_score, stability_holdout_size,
        stability_threshold, stability_flag, and stability_reason.

        Clusters below the small-group threshold are removed before the join. If there are
        too few identified customers to cluster, an empty ten-column DataFrame is returned
        and clustering is not attempted.

    Raises:
        AnalyticsError: If the frame contract is violated, k is out of range, or too few customers
            are present for the requested k.
    """

    _require_columns(frame, "rfm_segments")
    if k < 2:
        raise AnalyticsError("k must be >= 2")

    holdout_fraction = _resolve_holdout_fraction(settings)
    stability_threshold = _resolve_stability_threshold(settings)

    rfm = _compute_rfm(frame)
    if rfm.empty or len(rfm) < _MIN_CUSTOMERS_FOR_CLUSTERING:
        _audit(
            audit_log,
            actor=actor,
            resource="rfm_segments",
            details={
                "customers": int(len(rfm)),
                "suppressed_reason": "too_few_customers",
                "stability_score": None,
                "stability_holdout_size": 0,
                "stability_flag": None,
                "stability_reason": "too_few_customers",
            },
        )
        return _merge_stability(
            _empty_rfm_output(),
            customer_ids=(),
            score=None,
            holdout_size=0,
            threshold=stability_threshold,
            flag=None,
            reason="too_few_customers",
        )

    if k >= len(rfm):
        raise AnalyticsError(f"k={k} is not less than the number of customers ({len(rfm)})")

    # Standardize before K-Means so no dimension dominates the Euclidean distance. Monetary
    # value, in particular, has a much larger scale than recency-in-days or invoice counts.
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm[list(_RFM_COLUMNS)].to_numpy())

    model = KMeans(
        n_clusters=k,
        n_init=_KMEANS_N_INIT,
        random_state=_KMEANS_RANDOM_STATE,
    )
    labels = model.fit_predict(scaled)
    rfm = rfm.copy()
    rfm["cluster"] = labels.astype("int64")

    # Silhouette requires at least two distinct labels. If K-Means collapses to one cluster,
    # report the result without raising.
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        _audit(
            audit_log,
            actor=actor,
            resource="rfm_segments",
            details={
                "k": int(k),
                "customers": int(len(rfm)),
                "kept_customers": int(len(rfm)),
                "silhouette": None,
                "clusters_suppressed": 0,
                "suppressed_reason": "degenerate_single_cluster",
                "stability_score": None,
                "stability_holdout_size": 0,
                "stability_flag": None,
                "stability_reason": "degenerate_single_cluster",
            },
        )
        return _merge_stability(
            rfm[["CustomerID", *_RFM_COLUMNS, "cluster"]],
            customer_ids=rfm["CustomerID"].tolist(),
            score=None,
            holdout_size=0,
            threshold=stability_threshold,
            flag=None,
            reason="degenerate_single_cluster",
        )

    silhouette = float(silhouette_score(scaled, labels))

    stability_score, holdout_size, stability_reason = _compute_stability(
        scaled=scaled,
        full_labels=labels,
        k=k,
        holdout_fraction=holdout_fraction,
    )
    stability_flag: bool | None
    if stability_score is None:
        stability_flag = None
    else:
        stability_flag = bool(stability_score >= stability_threshold)

    # Suppress clusters below the configured small-group threshold. Each dropped cluster is
    # audited by enforce_small_group as ACCESS_DENIED.
    cluster_sizes = rfm["cluster"].value_counts().to_dict()
    suppressed: list[int] = []
    kept_clusters: list[int] = []
    for cluster_id, size in sorted(cluster_sizes.items()):
        allowed = enforce_small_group(
            int(size),
            settings,
            threshold=threshold,
            actor=actor,
            resource=f"rfm_cluster_{cluster_id}",
            audit_log=audit_log,
        )
        if allowed:
            kept_clusters.append(int(cluster_id))
        else:
            suppressed.append(int(cluster_id))

    if suppressed:
        rfm = rfm.loc[rfm["cluster"].isin(kept_clusters)].reset_index(drop=True)

    _audit(
        audit_log,
        actor=actor,
        resource="rfm_segments",
        details={
            "k": int(k),
            "customers": int(sum(cluster_sizes.values())),
            "kept_customers": int(len(rfm)),
            "silhouette": round(silhouette, 4),
            "clusters_suppressed": len(suppressed),
            "stability_score": (
                None if stability_score is None else round(stability_score, 4)
            ),
            "stability_holdout_size": int(holdout_size),
            "stability_flag": stability_flag,
            "stability_reason": stability_reason,
        },
    )
    return _merge_stability(
        rfm[["CustomerID", *_RFM_COLUMNS, "cluster"]],
        customer_ids=rfm["CustomerID"].tolist(),
        score=stability_score,
        holdout_size=holdout_size,
        threshold=stability_threshold,
        flag=stability_flag,
        reason=stability_reason,
    )


# --------------------------------------------------------------------------- #
# RFM helpers
# --------------------------------------------------------------------------- #


def _compute_rfm(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the per-customer RFM table (before scaling and clustering).

    Recency is measured in whole days from the frame's maximum non-return
    invoice date to each customer's most recent positive invoice date.
    Frequency counts distinct positive invoices; monetary sums Revenue on
    non-adjustment rows (returns net out).
    """

    non_adj = _non_adjustment(frame)
    identified = non_adj.dropna(subset=["CustomerID", "InvoiceDate"])
    if identified.empty:
        return pd.DataFrame(columns=["CustomerID", *_RFM_COLUMNS]).astype(
            {"recency": "int64", "frequency": "int64", "monetary": "float64"}
        )

    positives = identified.loc[~identified["IsReturn"].fillna(False).astype(bool)]
    if positives.empty:
        return pd.DataFrame(columns=["CustomerID", *_RFM_COLUMNS]).astype(
            {"recency": "int64", "frequency": "int64", "monetary": "float64"}
        )

    reference_date = positives["InvoiceDate"].max()

    last_purchase = positives.groupby("CustomerID", dropna=True)["InvoiceDate"].max()
    recency = (reference_date - last_purchase).dt.days.astype("int64")

    frequency = positives.groupby("CustomerID", dropna=True)["InvoiceNo"].nunique().astype("int64")

    monetary = identified.groupby("CustomerID", dropna=True)["Revenue"].sum().astype("float64")

    rfm = (
        recency.to_frame("recency")
        .join(frequency.rename("frequency"))
        .join(monetary.rename("monetary"))
        .reset_index()
    )
    # Drop customers with missing or non-finite monetary totals. K-Means cannot fit on
    # non-finite values.
    rfm = rfm.replace([np.inf, -np.inf], np.nan).dropna(subset=list(_RFM_COLUMNS))
    return rfm.reset_index(drop=True)


def _empty_rfm_output() -> pd.DataFrame:
    """Return an empty five-column RFM frame with the documented dtypes.

    The merge helper adds the five sensitivity columns on top of this base.
    """

    return pd.DataFrame(
        {
            "CustomerID": pd.Series(dtype="object"),
            "recency": pd.Series(dtype="int64"),
            "frequency": pd.Series(dtype="int64"),
            "monetary": pd.Series(dtype="float64"),
            "cluster": pd.Series(dtype="int64"),
        }
    )


# --------------------------------------------------------------------------- #
# M4 fairness sensitivity helpers
# --------------------------------------------------------------------------- #


def _resolve_holdout_fraction(settings: "Settings | None") -> float:
    """Return the configured holdout fraction, falling back to 0.1 without settings."""

    if settings is None:
        return 0.1
    return float(settings.rfm_holdout_fraction)


def _resolve_stability_threshold(settings: "Settings | None") -> float:
    """Return the configured stability threshold, falling back to 0.7 without settings."""

    if settings is None:
        return 0.7
    return float(settings.rfm_stability_threshold)


def _compute_stability(
    *,
    scaled: np.ndarray,
    full_labels: np.ndarray,
    k: int,
    holdout_fraction: float,
) -> tuple[float | None, int, str | None]:
    """Return clustering stability from a holdout refit.

    Refit K-Means on the training partition, predict labels for all customers,
    and compare those labels with the original fit using adjusted Rand index.

    Returns:
        Tuple of score, holdout_size, and reason. score is None when the check
        is skipped. reason is None on a normal run.
    """

    n_customers = int(scaled.shape[0])
    holdout_size = int(round(n_customers * holdout_fraction))
    train_size = n_customers - holdout_size

    if holdout_size < _MIN_HOLDOUT_SIZE or train_size < _MIN_TRAIN_FOR_HOLDOUT or train_size <= k:
        return None, 0, "insufficient_customers_for_holdout"

    rng = np.random.default_rng(_HOLDOUT_RANDOM_STATE)
    permutation = rng.permutation(n_customers)
    train_index = permutation[:train_size]
    train_matrix = scaled[train_index]

    refit_model = KMeans(
        n_clusters=k,
        n_init=_KMEANS_N_INIT,
        random_state=_KMEANS_RANDOM_STATE,
    )
    refit_model.fit(train_matrix)
    refit_labels = refit_model.predict(scaled)

    if np.unique(refit_labels).size < 2:
        # Degenerate refit. Treat the stability check as skipped.
        return None, holdout_size, "insufficient_customers_for_holdout"

    score = float(adjusted_rand_score(full_labels, refit_labels))
    return score, holdout_size, None


def _merge_stability(
    segments: pd.DataFrame,
    *,
    customer_ids: "list[Any] | tuple[Any, ...]",
    score: float | None,
    holdout_size: int,
    threshold: float,
    flag: bool | None,
    reason: str | None,
) -> pd.DataFrame:
    """Join stability fields onto each segmented customer.

    Stability values describe the whole fit, but storing them per row keeps the
    fields attached through downstream filters, groupbys, concats, and merges.
    """

    sensitivity = _build_sensitivity_frame(
        customer_ids=customer_ids,
        score=score,
        holdout_size=holdout_size,
        threshold=threshold,
        flag=flag,
        reason=reason,
    )
    merged = segments.merge(sensitivity, on="CustomerID", how="left")
    return merged


def _build_sensitivity_frame(
    *,
    customer_ids: "list[Any] | tuple[Any, ...]",
    score: float | None,
    holdout_size: int,
    threshold: float,
    flag: bool | None,
    reason: str | None,
) -> pd.DataFrame:
    """Build one stability row per customer."""

    row_count = len(customer_ids)
    reason_text = "" if reason is None else str(reason)
    if row_count == 0:
        return pd.DataFrame(
            {
                "CustomerID": pd.Series(dtype="object"),
                "stability_score": pd.Series(dtype="float64"),
                "stability_holdout_size": pd.Series(dtype="int64"),
                "stability_threshold": pd.Series(dtype="float64"),
                "stability_flag": pd.Series(dtype="object"),
                "stability_reason": pd.Series(dtype="object"),
            }
        )
    score_value = float("nan") if score is None else float(score)
    return pd.DataFrame(
        {
            "CustomerID": list(customer_ids),
            "stability_score": pd.Series(
                [score_value] * row_count, dtype="float64"
            ),
            "stability_holdout_size": pd.Series(
                [int(holdout_size)] * row_count, dtype="int64"
            ),
            "stability_threshold": pd.Series(
                [float(threshold)] * row_count, dtype="float64"
            ),
            "stability_flag": pd.Series(
                [None if flag is None else bool(flag)] * row_count,
                dtype="object",
            ),
            "stability_reason": pd.Series(
                [reason_text] * row_count, dtype="object"
            ),
        }
    )


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

    Pandas reductions can return numpy or nullable scalar types. Presentation code
    expects plain floats, including 0.0 for empty or all-missing revenue totals.
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
