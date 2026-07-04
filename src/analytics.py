"""M3 (Metrics & KPI) + M4 (Segmentation / RFM).

Metrics are computed with pandas aggregations; RFM segmentation uses
``sklearn.cluster.KMeans`` with silhouette scoring. Cluster labels are
inspected for fairness (score sensitivity to holdout removal) before being
released to the presentation layer.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def revenue_summary(frame: "pd.DataFrame") -> dict:
    """Return revenue, order count, and AOV for the input frame."""
    raise NotImplementedError("Implemented in feature/m3-metrics")


def rfm_segments(frame: "pd.DataFrame", k: int = 4) -> "pd.DataFrame":
    """Compute RFM scores + K-Means segments.

    Returns a frame with columns ``pseudonym, recency, frequency, monetary,
    cluster``. Fairness sensitivity checks are recorded to the audit journal.
    """
    raise NotImplementedError("Implemented in feature/m4-segmentation")
