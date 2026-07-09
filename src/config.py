"""M11 - Configuration and secrets loader.

Reads environment variables at startup and exposes an immutable Settings record
that the rest of the application consumes. Secrets never appear in the
repository: .env is git-ignored and .env.example carries the schema with
placeholder values only.

Design notes:

* Settings is a frozen dataclass with slots. Fields cannot be reassigned and
  invalid values raise ConfigError at construction rather than at first use.
* get_settings() is the single process-wide accessor. It caches its result so
  every caller sees the same immutable object; tests that need a different
  snapshot construct Settings(...) directly with the values they want.
* pseudonym_key is allowed to be empty at construction so unit tests and
  development shells do not need to carry a real HMAC key. Modules that
  actually consume the key call Settings.require_pseudonym_key(), which
  raises ConfigError if the value is missing.
* Validation is fail-fast: bad SQLITE_URL, invalid LOG_FORMAT, non-positive
  thresholds, or a malformed PSEUDONYM_KEY refuse to construct Settings.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# Contracts
# --------------------------------------------------------------------------- #

_SQLITE_URL_PREFIX: Final[str] = "sqlite:///"
_VALID_LOG_LEVELS: Final[frozenset[str]] = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)
_VALID_LOG_FORMATS: Final[frozenset[str]] = frozenset({"json", "text"})

# HMAC keys are hex-encoded. 32 hex characters = 128 bits, the floor we
# accept for a pseudonymization key. Production keys generated with
# secrets.token_hex(32) are 64 hex characters (256 bits).
_PSEUDONYM_KEY_MIN_HEX_LEN: Final[int] = 32
_HEX_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-fA-F]+$")


class ConfigError(ValueError):
    """Raised when Settings receives invalid configuration values."""


# --------------------------------------------------------------------------- #
# Settings record
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable process-wide configuration.

    Construct directly with keyword arguments in tests; use get_settings() in
    application code so all callers share a single cached snapshot.
    """

    data_raw_dir: Path = field(default=Path("data/raw"))
    data_processed_dir: Path = field(default=Path("data/processed"))
    sqlite_url: str = field(default="sqlite:///data/processed/dashboard.db")

    # Security & Ethics
    pseudonym_key: str = field(default="")
    small_group_threshold: int = field(default=5)
    rate_limit_rpm: int = field(default=120)

    # Logging
    log_level: str = field(default="INFO")
    log_format: str = field(default="json")
    log_dir: Path = field(default=Path("logs"))

    def __post_init__(self) -> None:
        # Normalize paths so callers always receive Path instances even when
        # a str was supplied by from_env().
        object.__setattr__(self, "data_raw_dir", Path(self.data_raw_dir))
        object.__setattr__(self, "data_processed_dir", Path(self.data_processed_dir))
        object.__setattr__(self, "log_dir", Path(self.log_dir))
        object.__setattr__(self, "log_level", str(self.log_level).upper())
        object.__setattr__(self, "log_format", str(self.log_format).lower())

        self._validate_sqlite_url(self.sqlite_url)
        self._validate_pseudonym_key(self.pseudonym_key)
        self._validate_positive_int("small_group_threshold", self.small_group_threshold)
        self._validate_positive_int("rate_limit_rpm", self.rate_limit_rpm)
        self._validate_log_level(self.log_level)
        self._validate_log_format(self.log_format)

    # ------------------------------------------------------------------ #
    # Consumer-facing helpers
    # ------------------------------------------------------------------ #

    def require_pseudonym_key(self) -> str:
        """Return the pseudonym key or raise if it is empty."""

        if not self.pseudonym_key:
            raise ConfigError(
                "PSEUDONYM_KEY is empty. Set it in .env to a hex string of at "
                f"least {_PSEUDONYM_KEY_MIN_HEX_LEN} characters "
                '(generate with: python -c "import secrets; '
                'print(secrets.token_hex(32))").'
            )
        return self.pseudonym_key

    # ------------------------------------------------------------------ #
    # Validators
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_sqlite_url(value: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ConfigError("SQLITE_URL must be a non-empty string")
        if not value.startswith(_SQLITE_URL_PREFIX):
            raise ConfigError(f"SQLITE_URL must start with '{_SQLITE_URL_PREFIX}'; got {value!r}")
        remainder = value[len(_SQLITE_URL_PREFIX) :].strip()  # noqa: E203
        if not remainder:
            raise ConfigError(
                f"SQLITE_URL is missing a database path after " f"'{_SQLITE_URL_PREFIX}'"
            )

    @staticmethod
    def _validate_pseudonym_key(value: str) -> None:
        if not isinstance(value, str):
            raise ConfigError("PSEUDONYM_KEY must be a string")
        if value == "":
            # Empty is permitted at construction; require_pseudonym_key()
            # enforces presence for consumers that need it.
            return
        if len(value) < _PSEUDONYM_KEY_MIN_HEX_LEN:
            raise ConfigError(
                "PSEUDONYM_KEY must be at least "
                f"{_PSEUDONYM_KEY_MIN_HEX_LEN} hex characters; got "
                f"{len(value)}"
            )
        if len(value) % 2 != 0:
            raise ConfigError("PSEUDONYM_KEY must contain an even number of hex characters")
        if not _HEX_RE.match(value):
            raise ConfigError(
                "PSEUDONYM_KEY must be hex-encoded (0-9, a-f); "
                "generate with secrets.token_hex(32)"
            )

    @staticmethod
    def _validate_positive_int(name: str, value: int) -> None:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ConfigError(f"{name} must be an int; got {type(value).__name__}")
        if value < 1:
            raise ConfigError(f"{name} must be >= 1; got {value}")

    @staticmethod
    def _validate_log_level(value: str) -> None:
        if not isinstance(value, str) or value not in _VALID_LOG_LEVELS:
            raise ConfigError(
                f"LOG_LEVEL must be one of {sorted(_VALID_LOG_LEVELS)}; got {value!r}"
            )

    @staticmethod
    def _validate_log_format(value: str) -> None:
        if not isinstance(value, str) or value not in _VALID_LOG_FORMATS:
            raise ConfigError(
                f"LOG_FORMAT must be one of {sorted(_VALID_LOG_FORMATS)}; got {value!r}"
            )


# --------------------------------------------------------------------------- #
# Environment loader
# --------------------------------------------------------------------------- #


def _from_env() -> Settings:
    """Build a Settings snapshot from os.environ using field defaults for gaps.

    Separated from get_settings() so tests can drive the loader without
    interacting with the module-level cache.
    """

    def _int(name: str, default: int) -> int:
        raw = os.getenv(name)
        if raw is None or raw == "":
            return default
        try:
            return int(raw)
        except ValueError as exc:
            raise ConfigError(f"{name} must be an integer; got {raw!r}") from exc

    return Settings(
        data_raw_dir=Path(os.getenv("DATA_RAW_DIR", "data/raw")),
        data_processed_dir=Path(os.getenv("DATA_PROCESSED_DIR", "data/processed")),
        sqlite_url=os.getenv("SQLITE_URL", "sqlite:///data/processed/dashboard.db"),
        pseudonym_key=os.getenv("PSEUDONYM_KEY", ""),
        small_group_threshold=_int("SMALL_GROUP_SUPPRESSION_THRESHOLD", 5),
        rate_limit_rpm=_int("RATE_LIMIT_RPM", 120),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_format=os.getenv("LOG_FORMAT", "json"),
        log_dir=Path(os.getenv("LOG_DIR", "logs")),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide Settings snapshot."""

    load_dotenv()
    return _from_env()


__all__ = ["ConfigError", "Settings", "get_settings"]
