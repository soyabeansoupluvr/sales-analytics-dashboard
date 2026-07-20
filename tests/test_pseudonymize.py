"""Tests for M6 pseudonymization.

Covers deterministic hashes, key-sensitive output, 64-character lowercase
SHA-256 digests, CustomerID replacement, alternate output columns, null
passthrough, numeric identifiers, contract errors, and Settings integration.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.config import ConfigError, Settings
from src.pseudonymize import (
    PseudonymizationError,
    pseudonymize_column,
    pseudonymize_customers,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

# 64 hex chars = 256 bits.
_KEY_HEX = "ab" * 32
_KEY_BYTES = bytes.fromhex(_KEY_HEX)
_KEY_HEX_ALT = "cd" * 32  # different key for key-sensitivity tests
_KEY_BYTES_ALT = bytes.fromhex(_KEY_HEX_ALT)


@pytest.fixture
def customers_frame() -> pd.DataFrame:
    """Small canonical frame with mixed CustomerID types."""

    return pd.DataFrame(
        {
            "CustomerID": ["12345", "67890", "12345", "99999"],
            "Quantity": [1, 2, 3, 4],
            "UnitPrice": [1.0, 2.0, 3.0, 4.0],
        }
    )


@pytest.fixture
def settings_with_key() -> Settings:
    return Settings(pseudonym_key=_KEY_HEX)


# --------------------------------------------------------------------------- #
# Determinism & key sensitivity
# --------------------------------------------------------------------------- #


def test_pseudonymize_is_deterministic_under_same_key(customers_frame: pd.DataFrame) -> None:
    first = pseudonymize_column(customers_frame, key=_KEY_BYTES)
    second = pseudonymize_column(customers_frame, key=_KEY_BYTES)
    pd.testing.assert_series_equal(first["CustomerID"], second["CustomerID"])


def test_pseudonymize_preserves_equality_for_repeated_ids(customers_frame: pd.DataFrame) -> None:
    """Two rows with the same original CustomerID must produce the same pseudonym.

    This is the property RFM/grouping downstream relies on.
    """

    out = pseudonymize_column(customers_frame, key=_KEY_BYTES)
    assert out.loc[0, "CustomerID"] == out.loc[2, "CustomerID"]  # both "12345"
    assert out.loc[0, "CustomerID"] != out.loc[1, "CustomerID"]  # "12345" vs "67890"


def test_pseudonymize_is_key_sensitive(customers_frame: pd.DataFrame) -> None:
    under_key_a = pseudonymize_column(customers_frame, key=_KEY_BYTES)
    under_key_b = pseudonymize_column(customers_frame, key=_KEY_BYTES_ALT)
    # Every row should differ across keys.
    assert (under_key_a["CustomerID"] != under_key_b["CustomerID"]).all()


def test_pseudonymize_output_is_64_char_lowercase_hex(customers_frame: pd.DataFrame) -> None:
    out = pseudonymize_column(customers_frame, key=_KEY_BYTES)
    for value in out["CustomerID"]:
        assert isinstance(value, str)
        assert len(value) == 64
        assert value == value.lower()
        int(value, 16)  # will raise if any non-hex char slipped in


# --------------------------------------------------------------------------- #
# In-place semantics & alternate outputs
# --------------------------------------------------------------------------- #


def test_pseudonymize_does_not_mutate_input(customers_frame: pd.DataFrame) -> None:
    original = customers_frame.copy(deep=True)
    _ = pseudonymize_column(customers_frame, key=_KEY_BYTES)
    pd.testing.assert_frame_equal(customers_frame, original)


def test_pseudonymize_default_replaces_customer_id_in_place(
    customers_frame: pd.DataFrame,
) -> None:
    out = pseudonymize_column(customers_frame, key=_KEY_BYTES)
    # Same schema (no new column, no dropped column).
    assert list(out.columns) == ["CustomerID", "Quantity", "UnitPrice"]
    # But the CustomerID values have been replaced with pseudonyms.
    assert (out["CustomerID"] != customers_frame["CustomerID"]).all()


def test_pseudonymize_supports_alternate_output_column(customers_frame: pd.DataFrame) -> None:
    out = pseudonymize_column(
        customers_frame,
        column="CustomerID",
        key=_KEY_BYTES,
        output_column="CustomerPseudonym",
    )
    assert "CustomerPseudonym" in out.columns
    assert "CustomerID" in out.columns  # drop_source defaults to False
    assert (out["CustomerID"] == customers_frame["CustomerID"]).all()


def test_pseudonymize_drop_source_removes_original_column(customers_frame: pd.DataFrame) -> None:
    out = pseudonymize_column(
        customers_frame,
        column="CustomerID",
        key=_KEY_BYTES,
        output_column="CustomerPseudonym",
        drop_source=True,
    )
    assert "CustomerID" not in out.columns
    assert "CustomerPseudonym" in out.columns


def test_pseudonymize_drop_source_is_noop_when_output_equals_column(
    customers_frame: pd.DataFrame,
) -> None:
    """drop_source with output_column=column must not delete the target."""

    out = pseudonymize_column(
        customers_frame,
        column="CustomerID",
        key=_KEY_BYTES,
        output_column="CustomerID",
        drop_source=True,
    )
    assert "CustomerID" in out.columns


# --------------------------------------------------------------------------- #
# Null passthrough & numeric coercion
# --------------------------------------------------------------------------- #


def test_pseudonymize_passes_null_values_through_unchanged() -> None:
    frame = pd.DataFrame({"CustomerID": ["A", None, "B", np.nan]})
    out = pseudonymize_column(frame, key=_KEY_BYTES)

    assert isinstance(out.loc[0, "CustomerID"], str)
    assert pd.isna(out.loc[1, "CustomerID"])
    assert isinstance(out.loc[2, "CustomerID"], str)
    assert pd.isna(out.loc[3, "CustomerID"])


def test_pseudonymize_handles_numeric_customer_ids() -> None:
    """Int or float CustomerIDs (that survived M5 without str coercion) still work."""

    frame = pd.DataFrame({"CustomerID": [12345, 67890, 12345]})
    out = pseudonymize_column(frame, key=_KEY_BYTES)
    assert out.loc[0, "CustomerID"] == out.loc[2, "CustomerID"]
    assert out.loc[0, "CustomerID"] != out.loc[1, "CustomerID"]

    # And matches the string form of the same integer -- the primitive coerces
    # via str(), so "12345" and 12345 collapse to the same pseudonym.
    string_frame = pd.DataFrame({"CustomerID": ["12345"]})
    string_out = pseudonymize_column(string_frame, key=_KEY_BYTES)
    assert string_out.loc[0, "CustomerID"] == out.loc[0, "CustomerID"]


# --------------------------------------------------------------------------- #
# Contract violations
# --------------------------------------------------------------------------- #


def test_pseudonymize_missing_column_raises_pseudonymization_error() -> None:
    frame = pd.DataFrame({"NotCustomer": [1, 2, 3]})
    with pytest.raises(PseudonymizationError, match="CustomerID"):
        pseudonymize_column(frame, key=_KEY_BYTES)


def test_pseudonymize_missing_named_column_reports_that_name() -> None:
    frame = pd.DataFrame({"CustomerID": [1]})
    with pytest.raises(PseudonymizationError, match="Email"):
        pseudonymize_column(frame, column="Email", key=_KEY_BYTES)


@pytest.mark.parametrize("bad_key", [b"", bytearray()])
def test_pseudonymize_empty_key_raises_value_error(bad_key: bytes) -> None:
    frame = pd.DataFrame({"CustomerID": ["A"]})
    with pytest.raises(ValueError, match="non-empty"):
        pseudonymize_column(frame, key=bad_key)


def test_pseudonymize_non_bytes_key_raises_value_error() -> None:
    frame = pd.DataFrame({"CustomerID": ["A"]})
    with pytest.raises(ValueError, match="non-empty"):
        pseudonymize_column(frame, key=_KEY_HEX)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Orchestrator: pseudonymize_customers
# --------------------------------------------------------------------------- #


def test_pseudonymize_customers_uses_settings_key(
    customers_frame: pd.DataFrame, settings_with_key: Settings
) -> None:
    from_settings = pseudonymize_customers(customers_frame, settings_with_key)
    directly = pseudonymize_column(customers_frame, key=_KEY_BYTES)
    pd.testing.assert_series_equal(from_settings["CustomerID"], directly["CustomerID"])


def test_pseudonymize_customers_preserves_schema(
    customers_frame: pd.DataFrame, settings_with_key: Settings
) -> None:
    out = pseudonymize_customers(customers_frame, settings_with_key)
    assert list(out.columns) == list(customers_frame.columns)
    assert len(out) == len(customers_frame)


def test_pseudonymize_customers_requires_configured_key(customers_frame: pd.DataFrame) -> None:
    """An empty PSEUDONYM_KEY in Settings must refuse pseudonymization."""

    empty_settings = Settings()  # default pseudonym_key = ""
    with pytest.raises(ConfigError, match="PSEUDONYM_KEY"):
        pseudonymize_customers(customers_frame, empty_settings)


def test_pseudonymize_customers_rejects_frame_missing_customer_id(
    settings_with_key: Settings,
) -> None:
    frame = pd.DataFrame({"OrderID": [1, 2, 3]})
    with pytest.raises(PseudonymizationError, match="CustomerID"):
        pseudonymize_customers(frame, settings_with_key)


def test_pseudonymize_customers_never_leaks_raw_id_into_output(
    customers_frame: pd.DataFrame, settings_with_key: Settings
) -> None:
    """No pseudonymized value in the output frame may equal any raw input value.

    Guards against a regression where the orchestrator forgets to apply the
    HMAC and passes the frame through unchanged.
    """

    raw_values = {str(v) for v in customers_frame["CustomerID"] if not _is_nan(v)}
    out = pseudonymize_customers(customers_frame, settings_with_key)
    out_values = {str(v) for v in out["CustomerID"]}
    assert raw_values.isdisjoint(out_values)


def _is_nan(value: object) -> bool:
    return isinstance(value, float) and math.isnan(value)
