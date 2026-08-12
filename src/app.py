"""Interactive Sales Analytics Dashboard — Streamlit entry point.

Combines M1 UI layout and M2 presentation control. This layer composes the pipeline, storage,
analytics, and visualization modules, but does not compute metrics or build figures directly.

Module flow:

* M9, M10, and M6 ingest, clean, and pseudonymize the retail data.
* M7 stores pseudonymized snapshots and gates reads through M8 access control.
* M3 computes revenue, product, time, country, and repeat-rate metrics.
* M4 computes RFM segments and attaches fairness-sensitivity metadata through stability columns.
* M5 renders metric outputs as ExplainedChart records with Plotly figures, formulas, filters,
  and exclusions.

The layout follows the project design: Revenue, Products, Customers, and Time trends tabs.
The Customers tab is visible to unauthenticated visitors but locked behind the analyst role.

M2 reads the authenticated session, resolves the Actor, reads the role-scoped snapshot view,
applies user-selected filters, and dispatches to the metric and chart helper for the selected
analysis.

Author: Rev. Drew Brown
Course: MSIT 5910 Capstone, University of the People (2026)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Final, Iterable, Mapping

import pandas as pd
import streamlit as st
import streamlit_authenticator as stauth
from dotenv import load_dotenv

from src.access import Actor, Role, ViewName
from src.analytics import (
    country_metrics,
    product_metrics,
    repeat_rate,
    revenue_summary,
    rfm_segments,
    time_series,
)
from src.cleaning import clean
from src.config import Settings, get_settings
from src.credentials import (
    CredentialStore,
    build_authenticator_credentials,
    default_store,
    role_for_username,
)
from src.ingestion import ingest
from src.logs import AuditLog, Database
from src.pseudonymize import pseudonymize_column
from src.storage import StorageError, list_snapshots, read_view, write_snapshot
from src.visualization import (
    ChartKind,
    ExplainedChart,
    revenue_summary_kpis,
    country_bar,
    repeat_rate_gauge,
    revenue_by_month,
    revenue_by_weekday,
    rfm_scatter,
    top_products_bar,
)

# --------------------------------------------------------------------------- #
# Load .env before Settings so PSEUDONYM_KEY is available.
# --------------------------------------------------------------------------- #
load_dotenv()


# --------------------------------------------------------------------------- #
# Controller records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class AnalysisChoice:
    """Analysis selector option."""

    analysis_id: str
    label: str
    m8_view: ViewName
    minimum_role: Role


@dataclass(frozen=True, slots=True)
class TabDefinition:
    """One visible dashboard tab and the analyses it contains."""

    key: ViewName
    label: str
    analyses: tuple[AnalysisChoice, ...]
    minimum_role: Role
    description: str


# Four dashboard tabs from the project design. The Customers tab is visible but locked unless
# the caller has an analyst or admin role. The description text introduces each tab before
# the filter form.
_TAB_CATALOG: Final[tuple[TabDefinition, ...]] = (
    TabDefinition(
        key="revenue",
        label="Revenue",
        analyses=(
            AnalysisChoice("revenue_summary_kpis", "Revenue summary", "revenue", "viewer"),
            AnalysisChoice("country_revenue", "Country revenue", "revenue", "viewer"),
        ),
        minimum_role="viewer",
        description=(
            "Shows overall revenue trends and revenue broken down by country. "
            "Pick an analysis below, optionally narrow the date range or select "
            "specific countries, then select \u201cShow me the numbers.\u201d"
        ),
    ),
    TabDefinition(
        key="products",
        label="Products",
        analyses=(AnalysisChoice("top_products", "Top products by revenue", "products", "viewer"),),
        minimum_role="viewer",
        description=(
            "Shows the top-selling products ranked by revenue. Optionally narrow "
            "the results to specific stock codes, then select \u201cShow me the "
            "numbers.\u201d"
        ),
    ),
    TabDefinition(
        key="customers",
        label="Customers",
        analyses=(
            AnalysisChoice("repeat_rate", "Repeat-purchase rate", "customers", "analyst"),
            AnalysisChoice("customer_segments", "Customer segments (RFM)", "customers", "analyst"),
        ),
        minimum_role="analyst",
        description=(
            "Shows the repeat-purchase rate and RFM customer segments. Available "
            "only to signed-in analysts and administrators. Optionally narrow by "
            "date range, country, product, or customer segment, then select "
            "\u201cShow me the numbers.\u201d"
        ),
    ),
    TabDefinition(
        key="time",
        label="Time trends",
        analyses=(
            AnalysisChoice("revenue_by_month", "Revenue by month", "time", "viewer"),
            AnalysisChoice("revenue_by_weekday", "Revenue by weekday", "time", "viewer"),
        ),
        minimum_role="viewer",
        description=(
            "Shows revenue trends by month and by day of week. Optionally narrow "
            "the date range, then select \u201cShow me the numbers.\u201d"
        ),
    ),
)

_ROLE_RANK: Final[Mapping[Role, int]] = {"viewer": 1, "analyst": 2, "admin": 3}


# --------------------------------------------------------------------------- #
# streamlit-authenticator wiring
# --------------------------------------------------------------------------- #
# Credentials live in app_user. The credential store feeds streamlit-authenticator, so this module
# does not hard-code password hashes or roles.

_COOKIE_NAME: Final[str] = "sales_dashboard_auth"
_COOKIE_KEY: Final[str] = "capstone-demo-cookie-key-not-a-secret"
_COOKIE_EXPIRY_DAYS: Final[float] = 0.0  # No persistent cookies; session ends when browser closes.


# --------------------------------------------------------------------------- #
# Controller helpers - pure functions, importable without a Streamlit runtime.
# --------------------------------------------------------------------------- #


def resolve_actor(role: Role, *, username: str | None = None) -> Actor:
    """Return an Actor for role.

    When username is absent, use a role-shaped local name for the unauthenticated viewer posture.
    Authenticated sessions pass their username through for audit records.
    """

    if role not in _ROLE_RANK:
        raise ValueError(f"unknown role: {role!r}")
    return Actor(username=username or f"{role}@local", role=role)


def tabs_for_role(role: Role) -> tuple[TabDefinition, ...]:
    """Return all tab definitions for role.

    The tab list is always the same. Roles control whether a tab is open or locked,
    not whether it is visible.
    """

    if role not in _ROLE_RANK:
        raise ValueError(f"unknown role: {role!r}")
    return _TAB_CATALOG


def tab_permitted_for_role(tab: TabDefinition, role: Role) -> bool:
    """Return whether role may open any analysis on tab."""

    if role not in _ROLE_RANK:
        raise ValueError(f"unknown role: {role!r}")
    return _ROLE_RANK[role] >= _ROLE_RANK[tab.minimum_role]


def analyses_for_tab(tab: TabDefinition, role: Role) -> tuple[AnalysisChoice, ...]:
    """Return tab analyses available to role.

    Locked tabs return an empty tuple.
    """

    if not tab_permitted_for_role(tab, role):
        return ()
    return tab.analyses


def filter_frame(
    frame: pd.DataFrame,
    *,
    start: date | None = None,
    end: date | None = None,
    countries: Iterable[str] | None = None,
    stock_codes: Iterable[str] | None = None,
    segments: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return a filtered copy of frame.

    Filter axes are optional and independent. Date filters apply to InvoiceDate, countries
    to Country, stock_codes to StockCode, and segments to CustomerID values selected from
    RFM clusters.

    Raises:
        TypeError: If frame is not a DataFrame.
        ValueError: If a requested filter's backing column is missing.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("filter_frame requires a pandas DataFrame")

    result = frame
    if start is not None or end is not None:
        if "InvoiceDate" not in result.columns:
            raise ValueError("date filter requested but frame lacks InvoiceDate")
        dates = pd.to_datetime(result["InvoiceDate"], errors="coerce")
        mask = pd.Series(True, index=result.index)
        if start is not None:
            mask &= dates >= pd.Timestamp(start)
        if end is not None:
            # Include the entire end day.
            mask &= dates < pd.Timestamp(end) + pd.Timedelta(days=1)
        result = result.loc[mask]

    country_list = [c for c in (countries or []) if c]
    if country_list:
        if "Country" not in result.columns:
            raise ValueError("country filter requested but frame lacks Country")
        result = result.loc[result["Country"].isin(country_list)]

    stock_list = [s for s in (stock_codes or []) if s]
    if stock_list:
        if "StockCode" not in result.columns:
            raise ValueError("stock_code filter requested but frame lacks StockCode")
        result = result.loc[result["StockCode"].astype(str).isin(stock_list)]

    segment_list = [c for c in (segments or []) if c]
    if segment_list:
        if "CustomerID" not in result.columns:
            raise ValueError("segment filter requested but frame lacks CustomerID")
        result = result.loc[result["CustomerID"].astype(str).isin(segment_list)]

    return result.copy()


_AnalysisRenderer = Callable[[pd.DataFrame], ExplainedChart]


def _render_revenue_summary_kpis(frame: pd.DataFrame) -> ExplainedChart:
    """Render revenue KPIs using the monthly revenue chart surface.

    The chart carries the revenue_summary payload in its formula and exclusions so the caption
    can show the KPI context.
    """

    return revenue_summary_kpis(revenue_summary(frame))


def _render_revenue_by_month(frame: pd.DataFrame) -> ExplainedChart:
    return revenue_by_month(time_series(frame), revenue_payload=revenue_summary(frame))


def _render_revenue_by_weekday(frame: pd.DataFrame) -> ExplainedChart:
    return revenue_by_weekday(time_series(frame), revenue_payload=revenue_summary(frame))


def _render_top_products(frame: pd.DataFrame) -> ExplainedChart:
    return top_products_bar(product_metrics(frame), metric="revenue")


def _render_country_revenue(frame: pd.DataFrame) -> ExplainedChart:
    return country_bar(country_metrics(frame))


def _render_repeat_rate(frame: pd.DataFrame) -> ExplainedChart:
    return repeat_rate_gauge(repeat_rate(frame))


def _render_customer_segments(frame: pd.DataFrame, settings: Settings) -> ExplainedChart:
    return rfm_scatter(rfm_segments(frame, settings=settings))


def dispatch_analysis(
    analysis_id: str,
    frame: pd.DataFrame,
    *,
    settings: Settings,
) -> ExplainedChart:
    """Route an analysis id to its metric and chart renderer.

    Unknown ids raise ValueError.
    """

    if analysis_id == "revenue_summary_kpis":
        return _render_revenue_summary_kpis(frame)
    if analysis_id == "revenue_by_month":
        return _render_revenue_by_month(frame)
    if analysis_id == "revenue_by_weekday":
        return _render_revenue_by_weekday(frame)
    if analysis_id == "top_products":
        return _render_top_products(frame)
    if analysis_id == "country_revenue":
        return _render_country_revenue(frame)
    if analysis_id == "repeat_rate":
        return _render_repeat_rate(frame)
    if analysis_id == "customer_segments":
        return _render_customer_segments(frame, settings)
    raise ValueError(f"unknown analysis id: {analysis_id!r}")


# --------------------------------------------------------------------------- #
# Snapshot bootstrap
# --------------------------------------------------------------------------- #


def _ensure_snapshot(
    settings: Settings,
    *,
    source: Path,
) -> str:
    """Return the latest snapshot id, creating one from source if needed.

    Existing snapshots are reused. When none exist, source is ingested, cleaned, pseudonymized,
    and stored as a timestamped snapshot.
    """

    existing = list_snapshots(settings)
    if existing:
        return existing[-1]

    raw = ingest(source)
    cleaned = clean(raw)
    key = settings.require_pseudonym_key().encode("utf-8")
    pseudonymized = pseudonymize_column(cleaned, key=key, drop_source=True)

    snapshot_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    audit_log = AuditLog(Database(settings))
    write_snapshot(
        pseudonymized,
        settings,
        snapshot_id=snapshot_id,
        audit_log=audit_log,
    )
    return snapshot_id


# --------------------------------------------------------------------------- #
# Session resolution helpers
# --------------------------------------------------------------------------- #


def _resolve_settings() -> Settings:
    """Return active Settings, honoring the AppTest session override."""

    override = st.session_state.get("__test_settings_override__")
    if isinstance(override, Settings):
        return override
    return get_settings()


def _resolve_credential_store(settings: Settings) -> CredentialStore:
    """Return the credential store, honoring the AppTest session override."""

    override = st.session_state.get("__test_credential_store_override__")
    if override is not None:
        return override
    return default_store(settings)


def _resolve_source(settings: Settings) -> Path:
    """Return the raw data source used for snapshot bootstrap."""

    override = st.session_state.get("__test_source_override__")
    if isinstance(override, (str, Path)):
        return Path(override)
    # Prefer a raw upload when present; otherwise fall back to demo.csv.
    raw_dir = Path(settings.data_raw_dir)
    for suffix in (".xlsx", ".csv"):
        candidates = sorted(raw_dir.glob(f"*{suffix}"))
        if candidates:
            return candidates[0]
    return raw_dir / "demo.csv"


def _country_options(frame: pd.DataFrame) -> list[str]:
    if "Country" not in frame.columns:
        return []
    return sorted(str(c) for c in frame["Country"].dropna().unique())


def _stock_code_options(frame: pd.DataFrame, *, limit: int = 500) -> list[str]:
    """Return StockCodes available for filtering, capped for selector size."""

    if "StockCode" not in frame.columns:
        return []
    codes = sorted({str(c) for c in frame["StockCode"].dropna().unique()})
    return codes[:limit]


def _date_bounds(frame: pd.DataFrame) -> tuple[date, date] | None:
    if "InvoiceDate" not in frame.columns:
        return None
    dates = pd.to_datetime(frame["InvoiceDate"], errors="coerce").dropna()
    if dates.empty:
        return None
    return dates.min().date(), dates.max().date()


_CHART_KIND_LABELS: Final[Mapping[ChartKind, str]] = {
    ChartKind.REVENUE_SUMMARY_KPIS: "Revenue summary",
    ChartKind.REVENUE_BY_MONTH: "Monthly revenue",
    ChartKind.REVENUE_BY_WEEKDAY: "Revenue by weekday",
    ChartKind.TOP_PRODUCTS_BAR: "Top products",
    ChartKind.COUNTRY_BAR: "Revenue by country",
    ChartKind.REPEAT_RATE_GAUGE: "Repeat customer rate",
    ChartKind.RFM_SCATTER: "Customer segments (RFM)",
}

# Per-key sentence templates for chart.exclusions entries that need units or context. Keys not
# listed here use a generic "Key: value." sentence.
_EXCLUSION_PHRASES: Final[Mapping[str, str]] = {
    "adjustments": "{value} adjustment rows were excluded from this result.",
    "returns_value": "Returns reduced revenue by about {value} in this result.",
    "lower_ranked_countries": (
        "{value} smaller countries were grouped out of this chart to keep it readable."
    ),
    "unidentified_customers": (
        "{value} orders came from customers who could not be identified, so they "
        "are not counted as repeat business."
    ),
    "small_group_clusters": "{value} clusters were too small to display and were excluded.",
    "clusters_rendered": "{value} customer segments are shown in this chart.",
}

# RFM stability metadata is already narrated in chart.formula, so these raw keys are omitted from
# the exclusions sentence.
_EXCLUSION_KEYS_NARRATED_IN_FORMULA: Final[frozenset[str]] = frozenset(
    {"stability_score", "stability_flag", "stability_holdout_size", "stability_reason"}
)

# Keys where a falsy value adds caption noise rather than useful context.
_EXCLUSION_KEYS_SKIPPED_WHEN_ZERO: Final[frozenset[str]] = frozenset(
    {
        "adjustments",
        "returns_value",
        "lower_ranked_countries",
        "unidentified_customers",
        "small_group_clusters",
    }
)


def _humanize_key(key: str) -> str:
    """Return a readable phrase for a snake_case key."""

    words = key.replace("_", " ")
    return words[:1].upper() + words[1:]


def _describe_filters(filters: Mapping[str, str]) -> str:
    """Return readable sentences for chart filters.

    Filter values are already verb phrases, so each sentence pairs a
    humanized key with its value.
    """

    return " ".join(f"{_humanize_key(name)} {value}." for name, value in filters.items())


def _describe_exclusions(exclusions: Mapping[str, object]) -> str:
    """Return readable sentences for chart exclusions."""

    sentences: list[str] = []
    for key, value in exclusions.items():
        if key in _EXCLUSION_KEYS_NARRATED_IN_FORMULA:
            continue
        if key in _EXCLUSION_KEYS_SKIPPED_WHEN_ZERO and not value:
            continue
        template = _EXCLUSION_PHRASES.get(key)
        if template is not None:
            sentences.append(template.format(value=value))
        else:
            sentences.append(f"{_humanize_key(key)}: {value}.")
    return " ".join(sentences)


def _render_caption(chart: ExplainedChart) -> None:
    """Render an ExplainedChart caption below its figure."""

    label = _CHART_KIND_LABELS.get(chart.kind, _humanize_key(chart.kind.value))
    st.markdown(f"**{label}** \u2014 {chart.row_count} rows shown.")
    st.markdown(f"**How this is calculated.** {chart.formula}")
    if chart.filters:
        st.markdown(f"**What is filtered.** {_describe_filters(chart.filters)}")
    exclusion_text = _describe_exclusions(chart.exclusions) if chart.exclusions else ""
    if exclusion_text:
        st.markdown(f"**What is left out.** {exclusion_text}")


# --------------------------------------------------------------------------- #
# Streamlit rendering
# --------------------------------------------------------------------------- #


def _authenticate(  # pragma: no cover - Streamlit widget wiring
    authenticator: stauth.Authenticate,
    store: CredentialStore,
) -> tuple[Role, str | None]:
    """Render the sidebar login widget and return role and username.

    Unauthenticated visitors use the viewer role.
    """

    authenticator.login(location="sidebar")
    auth_status = st.session_state.get("authentication_status")
    username = st.session_state.get("username")

    if auth_status is False:
        st.sidebar.error("Username or password is incorrect.")
        return "viewer", None
    if auth_status is None or not username:
        st.sidebar.info("Log in for customer-level analyses. Aggregate views remain open.")
        return "viewer", None
    try:
        role = role_for_username(store, username)
    except ValueError:
        st.sidebar.error(f"Unknown account: {username}")
        return "viewer", None
    record = store.find(username)
    display = record.display_name if record is not None else username
    st.sidebar.success(f"Signed in as {display} ({role}).")
    authenticator.logout(button_name="Log out", location="sidebar")
    return role, username


def _render_tab(  # pragma: no cover - Streamlit rendering path
    tab: TabDefinition,
    *,
    role: Role,
    username: str | None,
    settings: Settings,
    snapshot_id: str,
) -> None:
    """Render one dashboard tab."""

    st.caption(tab.description)

    if not tab_permitted_for_role(tab, role):
        st.info(
            f"The {tab.label} tab is available to authenticated analysts and "
            "administrators. Use the sidebar login to access customer-level analyses."
        )
        return

    analyses = analyses_for_tab(tab, role)
    if not analyses:
        st.warning("No analyses are available on this tab for the current role.")
        return

    # Use the widest role for filter options so controls show the full domain.
    preview_actor = resolve_actor("admin")
    try:
        preview = read_view(tab.key, preview_actor, settings, snapshot_id=snapshot_id)
    except StorageError:
        preview = pd.DataFrame()

    # Render filters only when this view includes their backing columns.
    # Otherwise a user could select a filter that filter_frame would reject.
    has_date = "InvoiceDate" in preview.columns
    has_country = "Country" in preview.columns
    has_stock_code = "StockCode" in preview.columns
    show_segment_filter = tab.key == "customers"

    bounds = _date_bounds(preview) if has_date else None
    country_options = _country_options(preview) if has_country else []
    stock_options = _stock_code_options(preview) if has_stock_code else []

    form_key = f"controls_{tab.key}"
    with st.form(form_key):
        analysis_label = st.selectbox(
            "Analysis",
            options=[a.label for a in analyses],
            index=0,
            key=f"analysis_{tab.key}",
        )
        chosen = next(a for a in analyses if a.label == analysis_label)

        start: date | None = None
        end: date | None = None
        if has_date:
            col1, col2 = st.columns(2)
            with col1:
                start = st.date_input(
                    "Start date",
                    value=bounds[0] if bounds else None,
                    min_value=bounds[0] if bounds else None,
                    max_value=bounds[1] if bounds else None,
                    key=f"start_{tab.key}",
                )
            with col2:
                end = st.date_input(
                    "End date",
                    value=bounds[1] if bounds else None,
                    min_value=bounds[0] if bounds else None,
                    max_value=bounds[1] if bounds else None,
                    key=f"end_{tab.key}",
                )

        countries: list[str] = []
        if has_country:
            countries = st.multiselect(
                "Countries (blank = all)",
                options=country_options,
                default=[],
                key=f"countries_{tab.key}",
            )

        stock_codes: list[str] = []
        if has_stock_code:
            stock_codes = st.multiselect(
                "Products / StockCodes (blank = all)",
                options=stock_options,
                default=[],
                key=f"products_{tab.key}",
                help=(
                    "Up to 500 StockCodes are listed; leave blank for all "
                    "products. StockCode-level detail matches the values in "
                    "the cleaned UCI Online Retail dataset."
                ),
            )

        segments: list[str] = []
        if show_segment_filter:
            segments = _customer_segment_options(
                preview_actor=preview_actor,
                settings=settings,
                snapshot_id=snapshot_id,
                form_key=form_key,
            )
        submitted = st.form_submit_button("Show me the numbers")

    if not submitted:
        st.info("Pick an analysis and any filters above, then submit.")
        return

    actor = resolve_actor(role, username=username)
    try:
        scoped = read_view(chosen.m8_view, actor, settings, snapshot_id=snapshot_id)
    except StorageError as exc:
        st.error(f"Access denied: {exc}")
        return

    try:
        filtered = filter_frame(
            scoped,
            start=start if isinstance(start, date) else None,
            end=end if isinstance(end, date) else None,
            countries=countries,
            stock_codes=stock_codes,
            segments=segments if show_segment_filter else None,
        )
    except ValueError as exc:
        st.error(f"Filter rejected: {exc}")
        return

    if filtered.empty:
        st.warning("No rows match the selected filters.")
        return

    try:
        chart = dispatch_analysis(chosen.analysis_id, filtered, settings=settings)
    except Exception as exc:  # noqa: BLE001 - surface any metric error
        st.error(f"Could not render {chosen.label}: {exc}")
        return

    st.plotly_chart(chart.figure, use_container_width=True)
    _render_caption(chart)


def _customer_segment_options(  # pragma: no cover - depends on live snapshot
    *,
    preview_actor: Actor,
    settings: Settings,
    snapshot_id: str,
    form_key: str,
) -> list[str]:
    """Return selected CustomerID values for chosen RFM clusters.

    Segments are computed from the current preview frame. If no usable clusters
    are available, no segment filter is rendered.
    """

    try:
        scoped = read_view("customers", preview_actor, settings, snapshot_id=snapshot_id)
    except StorageError:
        return []
    try:
        clusters = rfm_segments(scoped, settings=settings)
    except Exception:  # noqa: BLE001 - segmentation is best-effort in a filter
        return []
    if clusters.empty:
        return []

    membership: dict[str, list[str]] = {}
    for cluster_id, group in clusters.groupby("cluster"):
        membership[f"Cluster {int(cluster_id)}"] = [
            str(cid) for cid in group["CustomerID"].tolist()
        ]

    chosen_clusters = st.multiselect(
        "Customer segments (RFM clusters; blank = all)",
        options=list(membership.keys()),
        default=[],
        key=f"segments_{form_key}",
        help=(
            "Segments are recomputed on demand from the current snapshot. "
            "Selecting one or more clusters restricts the analyses to the "
            "pseudonymized customers those clusters contain."
        ),
    )
    selected: list[str] = []
    for cluster_label in chosen_clusters:
        selected.extend(membership.get(cluster_label, []))
    return selected


def _run_app() -> None:  # pragma: no cover - Streamlit entry point
    """Render the M1 UI."""

    st.set_page_config(
        page_title="Interactive Sales Analytics Dashboard",
        page_icon="📊",
        layout="wide",
    )
    st.title("Interactive Sales Analytics Dashboard")
    st.caption("MSIT 5910 Capstone - Rev. Drew Brown.")

    settings = _resolve_settings()

    try:
        snapshot_id = _ensure_snapshot(settings, source=_resolve_source(settings))
    except FileNotFoundError:
        st.error(
            "No snapshot found and no raw data at "
            f"``{_resolve_source(settings)}``. Run the pipeline once "
            "before launching the dashboard."
        )
        st.stop()
        return  # unreachable, keeps type-checkers happy

    store = _resolve_credential_store(settings)
    authenticator = stauth.Authenticate(
        credentials=build_authenticator_credentials(store),
        cookie_name=_COOKIE_NAME,
        cookie_key=_COOKIE_KEY,
        cookie_expiry_days=_COOKIE_EXPIRY_DAYS,
    )
    role, username = _authenticate(authenticator, store)

    tabs = tabs_for_role(role)
    tab_widgets = st.tabs([t.label for t in tabs])
    for tab_widget, tab_def in zip(tab_widgets, tabs):
        with tab_widget:
            _render_tab(
                tab_def,
                role=role,
                username=username,
                settings=settings,
                snapshot_id=snapshot_id,
            )


# Streamlit runs this module top-to-bottom on every rerun; guard with a
# name check so importing app.py for unit tests does not trigger the UI.
if __name__ == "__main__" or st.runtime.exists():  # pragma: no cover
    _run_app()
