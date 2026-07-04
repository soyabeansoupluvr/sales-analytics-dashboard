"""M5 — Visualization & Explain.

Charts are labeled with the underlying formula, the filter that produced
them, and any exclusions (e.g. cancellations, small-group suppression) so
the viewer always knows what is and is not in the picture.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    import plotly.graph_objects as go


def revenue_by_month(frame: "pd.DataFrame"):
    """Return a labeled Plotly figure for monthly revenue."""
    raise NotImplementedError("Implemented in feature/m5-visualization")


def rfm_scatter(segments: "pd.DataFrame"):
    """Return a labeled Plotly scatter for RFM segments."""
    raise NotImplementedError("Implemented in feature/m5-visualization")
