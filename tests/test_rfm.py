"""Tests for M4 RFM segmentation.

Covers rfm_segments and _compute_rfm: input validation, RFM computation, K-Means determinism,
output shape, small-group suppression, audit integration, and edge cases for empty inputs,
all-return inputs, too few customers, and invalid k values.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.access import Actor
from src.analytics import AnalyticsError, rfm_segments
from src.config import Settings
from src.logs import AuditAction, AuditLog, AuditOutcome, Database


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

_KEY_HEX = "cd" * 32


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
    return Actor(username="drew", role="analyst")


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


def _synthetic_frame(
    n_customers: int = 40,
    seed: int = 0,
    *,
    reference: pd.Timestamp = pd.Timestamp("2024-12-01"),
) -> pd.DataFrame:
    """Build a synthetic cleaned frame with n_customers pseudonyms."""

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for cid in range(1, n_customers + 1):
        n_inv = int(rng.integers(3, 9))
        for i in range(n_inv):
            offset = int(rng.integers(1, 300))
            d = reference - pd.Timedelta(days=offset)
            qty = int(rng.integers(1, 10))
            price = float(round(rng.uniform(2, 50), 2))
            rows.append(
                _cleaned_row(
                    InvoiceNo=f"INV{cid:04d}{i:02d}",
                    StockCode=f"SKU{int(rng.integers(1, 20)):03d}",
                    Quantity=qty,
                    UnitPrice=price,
                    CustomerID=f"pseudo-{cid:04d}",
                    InvoiceDate=d,
                    Revenue=qty * price,
                )
            )
    frame = pd.DataFrame(rows)
    frame["InvoiceDate"] = pd.to_datetime(frame["InvoiceDate"])
    return frame


# --------------------------------------------------------------------------- #
# Frame validation
# --------------------------------------------------------------------------- #


def test_rfm_segments_requires_frame_contract() -> None:
    with pytest.raises(AnalyticsError, match="missing required columns"):
        rfm_segments(pd.DataFrame({"foo": [1, 2, 3]}))


def test_rfm_segments_rejects_k_below_two() -> None:
    frame = _synthetic_frame(n_customers=20)
    with pytest.raises(AnalyticsError, match="k must be >= 2"):
        rfm_segments(frame, k=1, threshold=1)


def test_rfm_segments_rejects_k_at_or_above_customer_count() -> None:
    frame = _synthetic_frame(n_customers=15)
    # 15 customers but k=15 leaves no room for a valid silhouette.
    with pytest.raises(AnalyticsError, match="not less than the number of customers"):
        rfm_segments(frame, k=15, threshold=1)


# --------------------------------------------------------------------------- #
# RFM computation semantics
# --------------------------------------------------------------------------- #


def test_rfm_computes_recency_from_max_positive_date() -> None:
    """Recency is measured in days from the frame's max positive date."""

    rows = []
    # 15 customers, each with a single invoice at a distinct offset from ref
    for cid in range(1, 16):
        rows.append(
            _cleaned_row(
                InvoiceNo=f"INV{cid:03d}",
                CustomerID=f"pseudo-{cid:03d}",
                InvoiceDate=pd.Timestamp("2024-12-01") - pd.Timedelta(days=cid),
                Quantity=cid,
                UnitPrice=1.0,
                Revenue=float(cid),
            )
        )
    frame = pd.DataFrame(rows)
    frame["InvoiceDate"] = pd.to_datetime(frame["InvoiceDate"])

    out = rfm_segments(frame, k=3, threshold=1)
    assert len(out) == 15
    # The most recent customer (cid=1, offset=1) has recency 0: they define the reference date.
    most_recent = out.loc[out["CustomerID"] == "pseudo-001", "recency"].iloc[0]
    assert most_recent == 0
    # The oldest single-invoice customer (cid=15, offset=15) has recency = 14 days (15 - 1).
    oldest = out.loc[out["CustomerID"] == "pseudo-015", "recency"].iloc[0]
    assert oldest == 14


def test_rfm_frequency_counts_distinct_invoices_not_line_items() -> None:
    """Frequency is a count of distinct positive invoices, not rows."""

    rows = []
    # 12 solo customers so the frame is large enough to cluster
    for cid in range(2, 14):
        rows.append(
            _cleaned_row(
                InvoiceNo=f"SOLO{cid:03d}",
                CustomerID=f"pseudo-{cid:03d}",
                InvoiceDate=pd.Timestamp("2024-06-01"),
                Quantity=1,
                UnitPrice=1.0,
                Revenue=1.0,
            )
        )
    # Customer 001: two line items on the same invoice + a second invoice.
    # Frequency should equal 2, not 3.
    rows.extend(
        [
            _cleaned_row(
                InvoiceNo="A1",
                CustomerID="pseudo-001",
                StockCode="SKU1",
                Quantity=1,
                UnitPrice=1.0,
                Revenue=1.0,
                InvoiceDate=pd.Timestamp("2024-03-01"),
            ),
            _cleaned_row(
                InvoiceNo="A1",
                CustomerID="pseudo-001",
                StockCode="SKU2",
                Quantity=1,
                UnitPrice=1.0,
                Revenue=1.0,
                InvoiceDate=pd.Timestamp("2024-03-01"),
            ),
            _cleaned_row(
                InvoiceNo="A2",
                CustomerID="pseudo-001",
                StockCode="SKU1",
                Quantity=1,
                UnitPrice=1.0,
                Revenue=1.0,
                InvoiceDate=pd.Timestamp("2024-04-01"),
            ),
        ]
    )
    frame = pd.DataFrame(rows)
    frame["InvoiceDate"] = pd.to_datetime(frame["InvoiceDate"])

    out = rfm_segments(frame, k=2, threshold=1)
    freq = int(out.loc[out["CustomerID"] == "pseudo-001", "frequency"].iloc[0])
    assert freq == 2


def test_rfm_monetary_nets_returns() -> None:
    """Monetary should net returns; adjustments should be excluded entirely."""

    rows: list[dict[str, object]] = []
    # Filler so we have enough customers to cluster.
    for cid in range(2, 14):
        rows.append(
            _cleaned_row(
                InvoiceNo=f"F{cid}",
                CustomerID=f"pseudo-{cid:03d}",
                Quantity=1,
                UnitPrice=5.0,
                Revenue=5.0,
                InvoiceDate=pd.Timestamp("2024-05-01"),
            )
        )
    # Target customer: 100 gross, -30 return, +999 adjustment (should be ignored). Net
    # monetary should equal 70.
    rows.extend(
        [
            _cleaned_row(
                InvoiceNo="P1",
                CustomerID="pseudo-001",
                Quantity=10,
                UnitPrice=10.0,
                Revenue=100.0,
                InvoiceDate=pd.Timestamp("2024-04-01"),
            ),
            _cleaned_row(
                InvoiceNo="R1",
                CustomerID="pseudo-001",
                Quantity=-3,
                UnitPrice=10.0,
                Revenue=-30.0,
                IsReturn=True,
                InvoiceDate=pd.Timestamp("2024-04-15"),
            ),
            _cleaned_row(
                InvoiceNo="ADJ1",
                CustomerID="pseudo-001",
                Quantity=1,
                UnitPrice=999.0,
                Revenue=999.0,
                IsAdjustment=True,
                InvoiceDate=pd.Timestamp("2024-04-20"),
            ),
        ]
    )
    frame = pd.DataFrame(rows)
    frame["InvoiceDate"] = pd.to_datetime(frame["InvoiceDate"])

    out = rfm_segments(frame, k=2, threshold=1)
    monetary = float(out.loc[out["CustomerID"] == "pseudo-001", "monetary"].iloc[0])
    assert monetary == pytest.approx(70.0)


# --------------------------------------------------------------------------- #
# K-Means shape and determinism
# --------------------------------------------------------------------------- #


def test_rfm_produces_expected_output_shape() -> None:
    frame = _synthetic_frame(n_customers=40, seed=1)
    out = rfm_segments(frame, k=4, threshold=1)
    assert list(out.columns) == [
        "CustomerID",
        "recency",
        "frequency",
        "monetary",
        "cluster",
        "stability_score",
        "stability_holdout_size",
        "stability_threshold",
        "stability_flag",
        "stability_reason",
    ]
    assert out["cluster"].dtype == "int64"
    assert out["recency"].dtype == "int64"
    assert out["frequency"].dtype == "int64"
    assert out["monetary"].dtype == "float64"
    assert out["stability_score"].dtype == "float64"
    assert out["stability_holdout_size"].dtype == "int64"
    assert out["stability_threshold"].dtype == "float64"
    assert out["stability_flag"].dtype == object
    assert out["stability_reason"].dtype == object
    assert len(out) == 40


def test_rfm_is_deterministic_across_runs() -> None:
    """Two calls with the same input produce byte-identical cluster labels."""

    frame = _synthetic_frame(n_customers=40, seed=2)
    a = rfm_segments(frame, k=4, threshold=1)
    b = rfm_segments(frame, k=4, threshold=1)
    pd.testing.assert_frame_equal(
        a.sort_values("CustomerID").reset_index(drop=True),
        b.sort_values("CustomerID").reset_index(drop=True),
    )


def test_rfm_default_k_equals_four() -> None:
    frame = _synthetic_frame(n_customers=40, seed=3)
    out = rfm_segments(frame, threshold=1)
    # Up to k labels; not all clusters need be populated but there should be no label
    # outside [0, 3].
    labels = set(int(c) for c in out["cluster"].unique().tolist())
    assert labels.issubset({0, 1, 2, 3})
    assert len(labels) >= 2  # a real segmentation, not degenerate


# --------------------------------------------------------------------------- #
# Small-group suppression (fairness)
# --------------------------------------------------------------------------- #


def test_rfm_suppresses_small_clusters(
    settings: Settings, audit_log: AuditLog, analyst: Actor
) -> None:
    """Clusters below the threshold are dropped; larger clusters remain."""

    frame = _synthetic_frame(n_customers=40, seed=0)
    out = rfm_segments(
        frame,
        k=4,
        settings=settings,
        threshold=5,
        actor=analyst,
        audit_log=audit_log,
    )

    kept_sizes = out["cluster"].value_counts()
    # Every surviving cluster meets the threshold.
    assert (kept_sizes >= 5).all()
    events = audit_log.read(limit=100)
    denials = [e for e in events if e.action == AuditAction.ACCESS_DENIED]
    # The synthetic seed=0 frame yields a 3-member cluster that must be suppressed at
    # threshold=5 - confirm at least one denial fired.
    assert denials, "expected at least one small-group denial in this fixture"
    assert all(e.resource.startswith("rfm_cluster_") for e in denials)


def test_rfm_high_threshold_suppresses_all_clusters(
    settings: Settings, audit_log: AuditLog, analyst: Actor
) -> None:
    """A threshold above the total customer count suppresses everything."""

    frame = _synthetic_frame(n_customers=40, seed=4)
    out = rfm_segments(
        frame,
        k=4,
        settings=settings,
        threshold=1000,
        actor=analyst,
        audit_log=audit_log,
    )
    assert out.empty
    # Column contract still holds on the empty return.
    assert list(out.columns) == [
        "CustomerID",
        "recency",
        "frequency",
        "monetary",
        "cluster",
        "stability_score",
        "stability_holdout_size",
        "stability_threshold",
        "stability_flag",
        "stability_reason",
    ]


# --------------------------------------------------------------------------- #
# Audit integration
# --------------------------------------------------------------------------- #


def test_rfm_writes_summary_audit_event(
    settings: Settings, audit_log: AuditLog, analyst: Actor
) -> None:
    frame = _synthetic_frame(n_customers=30, seed=5)
    rfm_segments(
        frame,
        k=4,
        settings=settings,
        threshold=1,
        actor=analyst,
        audit_log=audit_log,
    )
    events = audit_log.read(limit=100)
    summary = [
        e
        for e in events
        if e.action == AuditAction.PROTECTED_STORE_READ and e.resource == "rfm_segments"
    ]
    assert len(summary) == 1
    ev = summary[0]
    assert ev.outcome == AuditOutcome.SUCCESS
    assert ev.actor == "drew"
    assert ev.details["k"] == 4
    assert ev.details["customers"] == 30
    assert ev.details["kept_customers"] == 30
    assert -1.0 <= float(ev.details["silhouette"]) <= 1.0
    assert ev.details["clusters_suppressed"] == 0


def test_rfm_omits_audit_when_no_log_supplied(audit_log: AuditLog) -> None:
    frame = _synthetic_frame(n_customers=30, seed=6)
    rfm_segments(frame, k=3, threshold=1)
    assert len(audit_log.read(limit=1)) == 0


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #


def test_rfm_empty_frame_returns_empty_output(
    settings: Settings, audit_log: AuditLog, analyst: Actor
) -> None:
    empty = pd.DataFrame(
        columns=[
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
        ]
    )
    empty["InvoiceDate"] = pd.to_datetime(empty["InvoiceDate"])

    out = rfm_segments(
        empty,
        k=4,
        settings=settings,
        threshold=1,
        actor=analyst,
        audit_log=audit_log,
    )
    assert out.empty
    assert list(out.columns) == [
        "CustomerID",
        "recency",
        "frequency",
        "monetary",
        "cluster",
        "stability_score",
        "stability_holdout_size",
        "stability_threshold",
        "stability_flag",
        "stability_reason",
    ]

    events = audit_log.read(limit=10)
    reasons = [
        e.details.get("suppressed_reason")
        for e in events
        if e.action == AuditAction.PROTECTED_STORE_READ and e.resource == "rfm_segments"
    ]
    assert "too_few_customers" in reasons


def test_rfm_all_return_frame_returns_empty_output() -> None:
    """A frame containing only returns has no positive activity to cluster."""

    rows = [
        _cleaned_row(
            InvoiceNo=f"R{i}",
            CustomerID=f"pseudo-{i:03d}",
            Quantity=-1,
            UnitPrice=1.0,
            Revenue=-1.0,
            IsReturn=True,
            InvoiceDate=pd.Timestamp("2024-01-01"),
        )
        for i in range(1, 20)
    ]
    frame = pd.DataFrame(rows)
    frame["InvoiceDate"] = pd.to_datetime(frame["InvoiceDate"])

    out = rfm_segments(frame, k=3, threshold=1)
    assert out.empty


def test_rfm_too_few_customers_returns_empty_output() -> None:
    """Fewer than _MIN_CUSTOMERS_FOR_CLUSTERING triggers the early return."""

    rows = [
        _cleaned_row(
            InvoiceNo=f"INV{i}",
            CustomerID=f"pseudo-{i:03d}",
            Quantity=1,
            UnitPrice=5.0,
            Revenue=5.0,
            InvoiceDate=pd.Timestamp("2024-01-01"),
        )
        for i in range(1, 6)  # 5 customers - below the floor of 10
    ]
    frame = pd.DataFrame(rows)
    frame["InvoiceDate"] = pd.to_datetime(frame["InvoiceDate"])

    out = rfm_segments(frame, k=3, threshold=1)
    assert out.empty
    assert list(out.columns) == [
        "CustomerID",
        "recency",
        "frequency",
        "monetary",
        "cluster",
        "stability_score",
        "stability_holdout_size",
        "stability_threshold",
        "stability_flag",
        "stability_reason",
    ]


def test_rfm_ignores_unidentified_customers() -> None:
    """Rows with a null CustomerID are excluded from the RFM table."""

    # Build 15 identified customers with varied RFM footprints so K-Means has real structure to
    # cluster on.
    frame = _synthetic_frame(n_customers=15, seed=7)
    # Add 5 guest-checkout rows (null CustomerID) at various dates.
    guests = pd.DataFrame(
        [
            _cleaned_row(
                InvoiceNo=f"GUEST{i}",
                CustomerID=None,
                Quantity=1,
                UnitPrice=5.0,
                Revenue=5.0,
                InvoiceDate=pd.Timestamp("2024-06-01") + pd.Timedelta(days=i),
            )
            for i in range(1, 6)
        ]
    )
    guests["InvoiceDate"] = pd.to_datetime(guests["InvoiceDate"])
    combined = pd.concat([frame, guests], ignore_index=True)

    out = rfm_segments(combined, k=3, threshold=1)
    assert len(out) == 15
    assert out["CustomerID"].notna().all()


# --------------------------------------------------------------------------- #
# M4 fairness sensitivity columns
# --------------------------------------------------------------------------- #


def test_rfm_populates_stability_columns_on_normal_fit(
    settings: Settings,
) -> None:
    """A normal fit produces per-row stability values that are constant across rows."""

    frame = _synthetic_frame(n_customers=40, seed=11)
    out = rfm_segments(frame, k=4, settings=settings, threshold=1)

    assert not out.empty
    score = out["stability_score"].iloc[0]
    assert pd.notna(score)
    assert -1.0 <= float(score) <= 1.0

    # Stability values describe the fit, so they should be identical per row.
    assert out["stability_score"].nunique(dropna=False) == 1
    assert out["stability_holdout_size"].nunique(dropna=False) == 1
    assert out["stability_threshold"].nunique(dropna=False) == 1
    assert out["stability_flag"].nunique(dropna=False) == 1
    assert out["stability_reason"].nunique(dropna=False) == 1

    assert out["stability_reason"].iloc[0] == ""
    assert isinstance(out["stability_flag"].iloc[0], bool)
    assert int(out["stability_holdout_size"].iloc[0]) >= 2


def test_rfm_stability_flag_respects_configured_threshold() -> None:
    """stability_flag toggles at the configured threshold and rides on every row."""

    frame = _synthetic_frame(n_customers=40, seed=12)

    lenient = Settings(
        sqlite_url="sqlite:///:memory:",
        pseudonym_key=_KEY_HEX,
        rfm_stability_threshold=0.0,
    )
    strict = Settings(
        sqlite_url="sqlite:///:memory:",
        pseudonym_key=_KEY_HEX,
        rfm_stability_threshold=1.0,
    )

    lenient_out = rfm_segments(frame, k=4, settings=lenient, threshold=1)
    strict_out = rfm_segments(frame, k=4, settings=strict, threshold=1)

    # The flag must reflect the score >= threshold comparison the module documents.
    lenient_score = float(lenient_out["stability_score"].iloc[0])
    strict_score = float(strict_out["stability_score"].iloc[0])
    assert lenient_score == strict_score  # same seed, same fit
    assert lenient_out["stability_flag"].iloc[0] is bool(lenient_score >= 0.0)
    assert strict_out["stability_flag"].iloc[0] is bool(strict_score >= 1.0)

    assert float(lenient_out["stability_threshold"].iloc[0]) == pytest.approx(0.0)
    assert float(strict_out["stability_threshold"].iloc[0]) == pytest.approx(1.0)


def test_rfm_too_few_customers_populates_skip_reason() -> None:
    """Too-few-customer outputs still expose the stability column contract."""

    rows = [
        _cleaned_row(
            InvoiceNo=f"INV{i}",
            CustomerID=f"pseudo-{i:03d}",
            Quantity=1,
            UnitPrice=5.0,
            Revenue=5.0,
            InvoiceDate=pd.Timestamp("2024-01-01"),
        )
        for i in range(1, 6)
    ]
    frame = pd.DataFrame(rows)
    frame["InvoiceDate"] = pd.to_datetime(frame["InvoiceDate"])

    out = rfm_segments(frame, k=3, threshold=1)
    assert out.empty
    # An empty frame still declares the sensitivity columns with the documented dtypes.
    assert out["stability_score"].dtype == "float64"
    assert out["stability_holdout_size"].dtype == "int64"
    assert out["stability_flag"].dtype == object
    assert out["stability_reason"].dtype == object


def test_rfm_insufficient_customers_for_holdout_records_skip(
    settings: Settings,
) -> None:
    """When the holdout would starve the refit, the reason column captures that."""

    settings_with_large_holdout = Settings(
        sqlite_url=settings.sqlite_url,
        pseudonym_key=_KEY_HEX,
        data_processed_dir=settings.data_processed_dir,
        data_raw_dir=settings.data_raw_dir,
        log_dir=settings.log_dir,
        rfm_holdout_fraction=0.95,
    )
    frame = _synthetic_frame(n_customers=11, seed=13)

    out = rfm_segments(
        frame,
        k=3,
        settings=settings_with_large_holdout,
        threshold=1,
    )
    assert not out.empty
    reason = str(out["stability_reason"].iloc[0])
    assert reason == "insufficient_customers_for_holdout"
    assert pd.isna(out["stability_score"].iloc[0])
    assert out["stability_flag"].iloc[0] is None
    assert int(out["stability_holdout_size"].iloc[0]) == 0


def test_rfm_stability_columns_survive_filter_and_groupby() -> None:
    """Sensitivity columns survive downstream filters and groupbys."""

    frame = _synthetic_frame(n_customers=40, seed=14)
    out = rfm_segments(frame, k=4, threshold=1)

    kept_clusters = sorted(out["cluster"].unique().tolist())[:2]
    filtered = out.loc[out["cluster"].isin(kept_clusters)].reset_index(drop=True)
    assert "stability_score" in filtered.columns
    assert filtered["stability_score"].nunique(dropna=False) == 1

    grouped = out.groupby("cluster", as_index=False)["stability_score"].first()
    assert "stability_score" in grouped.columns
    assert grouped["stability_score"].nunique(dropna=False) == 1
