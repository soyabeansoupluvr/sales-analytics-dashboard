"""Tests for the M1 layout and M2 controller in src.app.

Controller helpers are pure Python and tested directly. The Streamlit path is
covered by one AppTest smoke test that submits the form end to end without
starting a browser.

The layout under test has four tabs: Revenue, Products, Customers, and Time
trends. Unauthenticated visitors use the viewer role and see the Customers tab
as visible but locked. Tests inject a credential store through session state so
they do not touch the real SQLite database.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.access import Actor
from src.app import (
    _TAB_CATALOG,
    AnalysisChoice,
    TabDefinition,
    analyses_for_tab,
    dispatch_analysis,
    filter_frame,
    resolve_actor,
    tab_permitted_for_role,
    tabs_for_role,
)
from src.config import Settings
from src.credentials import CredentialStore, UserRecord
from src.logs import AuditLog, Database
from src.storage import write_snapshot
from src.visualization import ExplainedChart

_KEY_HEX = "ab" * 32
_APP_PATH = str(Path(__file__).resolve().parents[1] / "src" / "app.py")


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


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
def cleaned_frame() -> pd.DataFrame:
    """A four-week cleaned + pseudonymized frame suitable for every metric."""

    return pd.DataFrame(
        {
            "InvoiceNo": [f"I{i:03d}" for i in range(20)],
            "StockCode": ["SKU1", "SKU2", "SKU3", "SKU1", "SKU2"] * 4,
            "Description": ["Widget", "Gadget", "Gizmo", "Widget", "Gadget"] * 4,
            "Quantity": [1, 2, 3, 4, 5] * 4,
            "InvoiceDate": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-08",
                    "2024-01-09",
                    "2024-01-10",
                    "2024-01-11",
                    "2024-01-15",
                    "2024-01-16",
                    "2024-01-17",
                    "2024-01-18",
                    "2024-01-22",
                    "2024-01-23",
                    "2024-01-24",
                    "2024-01-25",
                    "2024-01-29",
                    "2024-01-30",
                    "2024-01-31",
                    "2024-02-01",
                ]
            ),
            "UnitPrice": [1.5, 2.5, 3.0, 1.5, 2.5] * 4,
            "CustomerID": [f"ps_{i:02d}" for i in range(20)],
            "Country": ["UK", "UK", "US", "FR", "UK"] * 4,
            "IsReturn": [False] * 20,
            "IsAdjustment": [False] * 20,
            "Revenue": [1.5, 5.0, 9.0, 6.0, 12.5] * 4,
        }
    )


@pytest.fixture
def audit_log(settings: Settings) -> AuditLog:
    return AuditLog(Database(settings))


@dataclass
class _StubCredentialStore:
    """In-memory CredentialStore for AppTest smoke runs."""

    records: tuple[UserRecord, ...]

    def load_all(self) -> tuple[UserRecord, ...]:
        return self.records

    def find(self, username: str) -> UserRecord | None:
        for record in self.records:
            if record.username == username:
                return record
        return None


@pytest.fixture
def stub_store() -> CredentialStore:
    return _StubCredentialStore(
        records=(
            UserRecord(
                username="analyst",
                password_hash="$2b$12$placeholderhashplaceholder",
                role="analyst",
                display_name="Analyst",
            ),
        )
    )


# --------------------------------------------------------------------------- #
# resolve_actor
# --------------------------------------------------------------------------- #


class TestResolveActor:
    @pytest.mark.parametrize("role", ["viewer", "analyst", "admin"])
    def test_returns_actor_for_each_known_role(self, role: str) -> None:
        actor = resolve_actor(role)  # type: ignore[arg-type]
        assert isinstance(actor, Actor)
        assert actor.role == role

    def test_accepts_explicit_username(self) -> None:
        actor = resolve_actor("analyst", username="alice")
        assert actor.username == "alice"

    def test_default_username_is_role_shaped(self) -> None:
        actor = resolve_actor("viewer")
        assert actor.username == "viewer@local"

    def test_rejects_unknown_role(self) -> None:
        with pytest.raises(ValueError, match="unknown role"):
            resolve_actor("superuser")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Tab catalog: shape and role visibility
# --------------------------------------------------------------------------- #


class TestTabCatalog:
    def test_catalog_lists_exactly_four_tabs_in_documented_order(self) -> None:
        assert [t.label for t in _TAB_CATALOG] == [
            "Revenue",
            "Products",
            "Customers",
            "Time trends",
        ]

    def test_tab_keys_match_m8_view_names(self) -> None:
        assert [t.key for t in _TAB_CATALOG] == [
            "revenue",
            "products",
            "customers",
            "time",
        ]

    def test_customers_tab_requires_analyst(self) -> None:
        customers = next(t for t in _TAB_CATALOG if t.key == "customers")
        assert customers.minimum_role == "analyst"

    def test_non_customer_tabs_open_to_viewer(self) -> None:
        for tab in _TAB_CATALOG:
            if tab.key == "customers":
                continue
            assert tab.minimum_role == "viewer"

    def test_every_tab_hosts_at_least_one_analysis(self) -> None:
        for tab in _TAB_CATALOG:
            assert len(tab.analyses) >= 1


class TestTabsForRole:
    @pytest.mark.parametrize("role", ["viewer", "analyst", "admin"])
    def test_returns_all_four_tabs_regardless_of_role(self, role: str) -> None:
        result = tabs_for_role(role)  # type: ignore[arg-type]
        assert len(result) == 4
        assert result == _TAB_CATALOG

    def test_rejects_unknown_role(self) -> None:
        with pytest.raises(ValueError, match="unknown role"):
            tabs_for_role("superuser")  # type: ignore[arg-type]


class TestTabPermittedForRole:
    @pytest.mark.parametrize("role", ["viewer", "analyst", "admin"])
    def test_non_customer_tabs_permitted_for_every_role(self, role: str) -> None:
        for tab in _TAB_CATALOG:
            if tab.key == "customers":
                continue
            assert tab_permitted_for_role(tab, role) is True  # type: ignore[arg-type]

    def test_customers_tab_locked_for_viewer(self) -> None:
        customers = next(t for t in _TAB_CATALOG if t.key == "customers")
        assert tab_permitted_for_role(customers, "viewer") is False

    @pytest.mark.parametrize("role", ["analyst", "admin"])
    def test_customers_tab_open_for_analyst_and_admin(self, role: str) -> None:
        customers = next(t for t in _TAB_CATALOG if t.key == "customers")
        assert tab_permitted_for_role(customers, role) is True  # type: ignore[arg-type]


class TestAnalysesForTab:
    def test_viewer_on_locked_customer_tab_gets_empty_tuple(self) -> None:
        customers = next(t for t in _TAB_CATALOG if t.key == "customers")
        assert analyses_for_tab(customers, "viewer") == ()

    @pytest.mark.parametrize("role", ["analyst", "admin"])
    def test_analyst_and_admin_get_full_customer_analysis_list(self, role: str) -> None:
        customers = next(t for t in _TAB_CATALOG if t.key == "customers")
        result = analyses_for_tab(customers, role)  # type: ignore[arg-type]
        assert result == customers.analyses

    def test_viewer_gets_full_analysis_list_on_open_tabs(self) -> None:
        revenue = next(t for t in _TAB_CATALOG if t.key == "revenue")
        assert analyses_for_tab(revenue, "viewer") == revenue.analyses


class TestAnalysisChoice:
    def test_is_frozen(self) -> None:
        choice = AnalysisChoice("x", "X", "revenue", "viewer")
        with pytest.raises((AttributeError, TypeError)):
            choice.label = "Y"  # type: ignore[misc]

    def test_uses_slots(self) -> None:
        choice = AnalysisChoice("x", "X", "revenue", "viewer")
        with pytest.raises((AttributeError, TypeError)):
            choice.extra = "no"  # type: ignore[attr-defined]


class TestTabDefinition:
    def test_is_frozen(self) -> None:
        tab = TabDefinition(
            key="revenue",
            label="Revenue",
            analyses=(AnalysisChoice("x", "X", "revenue", "viewer"),),
            minimum_role="viewer",
        )
        with pytest.raises((AttributeError, TypeError)):
            tab.label = "no"  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# filter_frame - extended with stock_codes and segments
# --------------------------------------------------------------------------- #


class TestFilterFrame:
    def test_no_args_returns_copy(self, cleaned_frame: pd.DataFrame) -> None:
        result = filter_frame(cleaned_frame)
        assert result.equals(cleaned_frame)
        assert result is not cleaned_frame

    def test_date_range(self, cleaned_frame: pd.DataFrame) -> None:
        result = filter_frame(cleaned_frame, start=date(2024, 1, 8), end=date(2024, 1, 14))
        dates = pd.to_datetime(result["InvoiceDate"])
        assert dates.min() >= pd.Timestamp("2024-01-08")
        assert dates.max() <= pd.Timestamp("2024-01-14")

    def test_country_subset(self, cleaned_frame: pd.DataFrame) -> None:
        result = filter_frame(cleaned_frame, countries=["US"])
        assert set(result["Country"]) == {"US"}

    def test_combined_filters(self, cleaned_frame: pd.DataFrame) -> None:
        result = filter_frame(
            cleaned_frame,
            start=date(2024, 1, 1),
            end=date(2024, 1, 15),
            countries=["UK", "FR"],
        )
        assert set(result["Country"]).issubset({"UK", "FR"})
        dates = pd.to_datetime(result["InvoiceDate"])
        assert dates.max() <= pd.Timestamp("2024-01-15")

    def test_empty_countries_iterable_is_noop(self, cleaned_frame: pd.DataFrame) -> None:
        result = filter_frame(cleaned_frame, countries=[])
        assert len(result) == len(cleaned_frame)

    def test_rejects_non_dataframe(self) -> None:
        with pytest.raises(TypeError):
            filter_frame([1, 2, 3])  # type: ignore[arg-type]

    def test_missing_date_column_raises(self, cleaned_frame: pd.DataFrame) -> None:
        without = cleaned_frame.drop(columns=["InvoiceDate"])
        with pytest.raises(ValueError, match="InvoiceDate"):
            filter_frame(without, start=date(2024, 1, 1))

    def test_missing_country_column_raises(self, cleaned_frame: pd.DataFrame) -> None:
        without = cleaned_frame.drop(columns=["Country"])
        with pytest.raises(ValueError, match="Country"):
            filter_frame(without, countries=["UK"])

    def test_returns_a_copy_not_a_view(self, cleaned_frame: pd.DataFrame) -> None:
        out = filter_frame(cleaned_frame)
        out.loc[out.index[0], "Country"] = "MUTATED"
        assert "MUTATED" not in set(cleaned_frame["Country"])

    # --- new filters ---------------------------------------------------------

    def test_stock_codes_subset(self, cleaned_frame: pd.DataFrame) -> None:
        result = filter_frame(cleaned_frame, stock_codes=["SKU1"])
        assert set(result["StockCode"]) == {"SKU1"}

    def test_stock_codes_multiple(self, cleaned_frame: pd.DataFrame) -> None:
        result = filter_frame(cleaned_frame, stock_codes=["SKU1", "SKU3"])
        assert set(result["StockCode"]) == {"SKU1", "SKU3"}

    def test_empty_stock_codes_is_noop(self, cleaned_frame: pd.DataFrame) -> None:
        result = filter_frame(cleaned_frame, stock_codes=[])
        assert len(result) == len(cleaned_frame)

    def test_missing_stockcode_column_raises(self, cleaned_frame: pd.DataFrame) -> None:
        without = cleaned_frame.drop(columns=["StockCode"])
        with pytest.raises(ValueError, match="StockCode"):
            filter_frame(without, stock_codes=["SKU1"])

    def test_segments_subset(self, cleaned_frame: pd.DataFrame) -> None:
        chosen = ["ps_00", "ps_01", "ps_02"]
        result = filter_frame(cleaned_frame, segments=chosen)
        assert set(result["CustomerID"]) == set(chosen)

    def test_empty_segments_is_noop(self, cleaned_frame: pd.DataFrame) -> None:
        result = filter_frame(cleaned_frame, segments=[])
        assert len(result) == len(cleaned_frame)

    def test_missing_customerid_column_raises(self, cleaned_frame: pd.DataFrame) -> None:
        without = cleaned_frame.drop(columns=["CustomerID"])
        with pytest.raises(ValueError, match="CustomerID"):
            filter_frame(without, segments=["ps_00"])

    def test_all_filters_combined(self, cleaned_frame: pd.DataFrame) -> None:
        result = filter_frame(
            cleaned_frame,
            start=date(2024, 1, 1),
            end=date(2024, 1, 31),
            countries=["UK"],
            stock_codes=["SKU1", "SKU2"],
            segments=[f"ps_{i:02d}" for i in range(10)],
        )
        assert set(result["Country"]) <= {"UK"}
        assert set(result["StockCode"]) <= {"SKU1", "SKU2"}
        assert set(result["CustomerID"]) <= {f"ps_{i:02d}" for i in range(10)}


# --------------------------------------------------------------------------- #
# dispatch_analysis
# --------------------------------------------------------------------------- #


class TestDispatchAnalysis:
    def test_rejects_unknown_id(self, cleaned_frame: pd.DataFrame, settings: Settings) -> None:
        with pytest.raises(ValueError, match="unknown analysis id"):
            dispatch_analysis("not_a_real_metric", cleaned_frame, settings=settings)

    @pytest.mark.parametrize(
        "analysis_id",
        [
            "revenue_summary_kpis",
            "revenue_by_month",
            "revenue_by_weekday",
            "top_products",
            "country_revenue",
            "repeat_rate",
            "customer_segments",
        ],
    )
    def test_every_analysis_id_dispatches(
        self,
        cleaned_frame: pd.DataFrame,
        settings: Settings,
        analysis_id: str,
    ) -> None:
        chart = dispatch_analysis(analysis_id, cleaned_frame, settings=settings)
        assert isinstance(chart, ExplainedChart)

    def test_customer_segments_uses_settings(
        self, cleaned_frame: pd.DataFrame, settings: Settings
    ) -> None:
        chart = dispatch_analysis("customer_segments", cleaned_frame, settings=settings)
        assert isinstance(chart, ExplainedChart)


# --------------------------------------------------------------------------- #
# AppTest smoke - verifies the Streamlit form wires up
# --------------------------------------------------------------------------- #


def _seed_snapshot(settings: Settings, frame: pd.DataFrame, audit_log: AuditLog) -> str:
    snapshot_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    write_snapshot(frame, settings, snapshot_id=snapshot_id, audit_log=audit_log)
    return snapshot_id


def test_apptest_renders_tabs_and_submits(
    settings: Settings,
    cleaned_frame: pd.DataFrame,
    audit_log: AuditLog,
    stub_store: CredentialStore,
) -> None:
    """Smoke-test the Streamlit app through one submitted form.

    The test injects temporary settings and credentials, seeds a snapshot,
    renders the app with AppTest, and submits the first available form.
    """

    _seed_snapshot(settings, cleaned_frame, audit_log)

    app = AppTest.from_file(_APP_PATH, default_timeout=15)
    app.session_state["__test_settings_override__"] = settings
    app.session_state["__test_credential_store_override__"] = stub_store
    app.run()
    assert not app.exception, [str(e) for e in app.exception]

    submit_buttons = [b for b in app.button if b.label.startswith("Show me")]
    assert submit_buttons, "expected at least one submit button to be rendered"

    submit_buttons[0].click().run()
    assert not app.exception, [str(e) for e in app.exception]
