"""Tests for M11 configuration and secrets loading.

Covers default Settings values, environment loading, validation rules,
pseudonym-key gating, dataclass immutability, get_settings caching, and parity
between Settings fields and .env.example keys.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from src.config import ConfigError, Settings, _from_env, get_settings

import src.config as config_module


# --------------------------------------------------------------------------- #
# Environment isolation
# --------------------------------------------------------------------------- #

# All Settings-relevant env vars. Cleared before each test so the loader
# always starts from a known baseline (its own field defaults).
_SETTINGS_ENV_VARS = (
    "DATA_RAW_DIR",
    "DATA_PROCESSED_DIR",
    "SQLITE_URL",
    "PSEUDONYM_KEY",
    "SMALL_GROUP_SUPPRESSION_THRESHOLD",
    "RATE_LIMIT_RPM",
    "LOG_LEVEL",
    "LOG_FORMAT",
    "LOG_DIR",
)


@pytest.fixture(autouse=True)
def _clean_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove Settings env vars and clear the get_settings cache before each test."""

    for name in _SETTINGS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config_module, "load_dotenv", lambda *a, **kw: False)
    get_settings.cache_clear()


# --------------------------------------------------------------------------- #
# Defaults & construction
# --------------------------------------------------------------------------- #


def test_settings_defaults_match_documented_baseline() -> None:
    """Zero-argument Settings() produces the documented development defaults."""

    settings = Settings()

    assert settings.data_raw_dir == Path("data/raw")
    assert settings.data_processed_dir == Path("data/processed")
    assert settings.sqlite_url == "sqlite:///data/processed/dashboard.db"
    assert settings.pseudonym_key == ""
    assert settings.small_group_threshold == 5
    assert settings.rate_limit_rpm == 120
    assert settings.log_level == "INFO"
    assert settings.log_format == "json"
    assert settings.log_dir == Path("logs")


def test_settings_normalizes_string_paths_to_path_objects() -> None:
    """Path-typed fields accept str input and are normalized to Path."""

    settings = Settings(
        data_raw_dir="custom/raw",  # type: ignore[arg-type]
        data_processed_dir="custom/processed",  # type: ignore[arg-type]
        log_dir="custom/logs",  # type: ignore[arg-type]
    )

    assert isinstance(settings.data_raw_dir, Path)
    assert isinstance(settings.data_processed_dir, Path)
    assert isinstance(settings.log_dir, Path)
    assert settings.data_raw_dir == Path("custom/raw")


# --------------------------------------------------------------------------- #
# Environment loader
# --------------------------------------------------------------------------- #


def test_from_env_reads_all_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every documented env var flows through _from_env() into Settings."""

    key = "a" * 64  # 256-bit hex key
    monkeypatch.setenv("DATA_RAW_DIR", "env/raw")
    monkeypatch.setenv("DATA_PROCESSED_DIR", "env/processed")
    monkeypatch.setenv("SQLITE_URL", "sqlite:///env/dashboard.db")
    monkeypatch.setenv("PSEUDONYM_KEY", key)
    monkeypatch.setenv("SMALL_GROUP_SUPPRESSION_THRESHOLD", "10")
    monkeypatch.setenv("RATE_LIMIT_RPM", "300")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("LOG_FORMAT", "text")
    monkeypatch.setenv("LOG_DIR", "env/logs")

    settings = _from_env()

    assert settings.data_raw_dir == Path("env/raw")
    assert settings.data_processed_dir == Path("env/processed")
    assert settings.sqlite_url == "sqlite:///env/dashboard.db"
    assert settings.pseudonym_key == key
    assert settings.small_group_threshold == 10
    assert settings.rate_limit_rpm == 300
    assert settings.log_level == "DEBUG"
    assert settings.log_format == "text"
    assert settings.log_dir == Path("env/logs")


def test_from_env_uses_defaults_when_variables_absent() -> None:
    """Missing env vars fall back to the Settings field defaults."""

    settings = _from_env()

    assert settings.sqlite_url == "sqlite:///data/processed/dashboard.db"
    assert settings.small_group_threshold == 5
    assert settings.log_level == "INFO"


def test_from_env_rejects_non_integer_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-numeric int env vars raise ConfigError with the offending value."""

    monkeypatch.setenv("RATE_LIMIT_RPM", "not-a-number")

    with pytest.raises(ConfigError, match="RATE_LIMIT_RPM"):
        _from_env()


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "bad_url",
    [
        "",
        "   ",
        "postgres://localhost/db",
        "sqlite:///",  # empty database path
        "sqlite:///   ",  # whitespace-only path
    ],
)
def test_settings_rejects_invalid_sqlite_url(bad_url: str) -> None:
    with pytest.raises(ConfigError, match="SQLITE_URL"):
        Settings(sqlite_url=bad_url)


def test_settings_rejects_invalid_log_level() -> None:
    with pytest.raises(ConfigError, match="LOG_LEVEL"):
        Settings(log_level="VERBOSE")


def test_settings_rejects_invalid_log_format() -> None:
    with pytest.raises(ConfigError, match="LOG_FORMAT"):
        Settings(log_format="yaml")


@pytest.mark.parametrize("bad_value", [0, -1, -100])
def test_settings_rejects_non_positive_small_group_threshold(bad_value: int) -> None:
    with pytest.raises(ConfigError, match="small_group_threshold"):
        Settings(small_group_threshold=bad_value)


@pytest.mark.parametrize("bad_value", [0, -1])
def test_settings_rejects_non_positive_rate_limit(bad_value: int) -> None:
    with pytest.raises(ConfigError, match="rate_limit_rpm"):
        Settings(rate_limit_rpm=bad_value)


def test_settings_rejects_short_pseudonym_key() -> None:
    with pytest.raises(ConfigError, match="at least 32"):
        Settings(pseudonym_key="abc123")


def test_settings_rejects_non_hex_pseudonym_key() -> None:
    # 32 characters but contains non-hex 'z'.
    with pytest.raises(ConfigError, match="hex-encoded"):
        Settings(pseudonym_key="z" * 32)


def test_settings_accepts_valid_pseudonym_key() -> None:
    key = "0123456789abcdef" * 4  # 64 hex chars = 256 bits  # gitleaks:allow
    settings = Settings(pseudonym_key=key)
    assert settings.pseudonym_key == key


# --------------------------------------------------------------------------- #
# require_pseudonym_key
# --------------------------------------------------------------------------- #


def test_require_pseudonym_key_raises_when_empty() -> None:
    settings = Settings()  # default empty key
    with pytest.raises(ConfigError, match="PSEUDONYM_KEY"):
        settings.require_pseudonym_key()


def test_require_pseudonym_key_returns_value_when_set() -> None:
    key = "f" * 64
    settings = Settings(pseudonym_key=key)
    assert settings.require_pseudonym_key() == key


# --------------------------------------------------------------------------- #
# Immutability
# --------------------------------------------------------------------------- #


def test_settings_is_frozen() -> None:
    """Fields cannot be reassigned after construction."""

    settings = Settings()
    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.log_level = "DEBUG"  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# get_settings cache
# --------------------------------------------------------------------------- #


def test_get_settings_returns_cached_instance() -> None:
    """Repeated calls return the identical object (same id)."""

    first = get_settings()
    second = get_settings()
    assert first is second


def test_get_settings_cache_clear_reloads_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """After cache_clear(), get_settings() reflects updated env vars."""

    first = get_settings()
    assert first.log_level == "INFO"

    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    # Without cache_clear, we'd still see the old snapshot.
    get_settings.cache_clear()

    second = get_settings()
    assert second.log_level == "WARNING"
    assert first is not second


# --------------------------------------------------------------------------- #
# .env.example parity
# --------------------------------------------------------------------------- #


# Env vars in .env.example that Streamlit reads directly and Settings does not model.
_STREAMLIT_ONLY_KEYS = frozenset({"STREAMLIT_SERVER_PORT", "STREAMLIT_SERVER_ADDRESS"})

# Env-var name --> Settings field name.
_ENV_TO_FIELD = {
    "DATA_RAW_DIR": "data_raw_dir",
    "DATA_PROCESSED_DIR": "data_processed_dir",
    "SQLITE_URL": "sqlite_url",
    "PSEUDONYM_KEY": "pseudonym_key",
    "SMALL_GROUP_SUPPRESSION_THRESHOLD": "small_group_threshold",
    "RATE_LIMIT_RPM": "rate_limit_rpm",
    "RFM_HOLDOUT_FRACTION": "rfm_holdout_fraction",
    "RFM_STABILITY_THRESHOLD": "rfm_stability_threshold",
    "LOG_LEVEL": "log_level",
    "LOG_FORMAT": "log_format",
    "LOG_DIR": "log_dir",
}


def _parse_env_example_keys() -> set[str]:
    """Return the set of KEY names declared in .env.example."""

    env_example = Path(__file__).parent.parent / ".env.example"
    keys: set[str] = set()
    for raw_line in env_example.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key = line.split("=", 1)[0].strip()
        if key:
            keys.add(key)
    return keys


def test_env_example_and_settings_are_in_strict_parity() -> None:
    """Every non-Streamlit key in .env.example maps to a Settings field, and vice versa."""

    env_keys = _parse_env_example_keys() - _STREAMLIT_ONLY_KEYS
    field_names = {f.name for f in dataclasses.fields(Settings)}

    expected_fields = {_ENV_TO_FIELD[k] for k in env_keys}

    missing_in_settings = env_keys - set(_ENV_TO_FIELD)
    assert not missing_in_settings, (
        f".env.example contains keys not mapped to any Settings field: "
        f"{sorted(missing_in_settings)}"
    )

    missing_in_env_example = field_names - expected_fields
    assert not missing_in_env_example, (
        f"Settings has fields with no matching .env.example entry: "
        f"{sorted(missing_in_env_example)}"
    )

    extra_env_keys = expected_fields - field_names
    assert not extra_env_keys, (
        f".env.example documents fields that no longer exist on Settings: "
        f"{sorted(extra_env_keys)}"
    )
