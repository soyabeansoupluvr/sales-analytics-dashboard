"""M11 — Configuration and secrets loader.

Reads environment variables from ``.env`` at startup. Secrets (HMAC keys,
authenticator hashes) are never committed and never logged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    data_raw_dir: Path = Path(os.getenv("DATA_RAW_DIR", "data/raw"))
    data_processed_dir: Path = Path(os.getenv("DATA_PROCESSED_DIR", "data/processed"))
    sqlite_url: str = os.getenv("SQLITE_URL", "sqlite:///data/processed/dashboard.db")

    # Security & Ethics
    pseudonym_key: str = os.getenv("PSEUDONYM_KEY", "")
    small_group_threshold: int = int(os.getenv("SMALL_GROUP_SUPPRESSION_THRESHOLD", "5"))
    rate_limit_rpm: int = int(os.getenv("RATE_LIMIT_RPM", "120"))

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_format: str = os.getenv("LOG_FORMAT", "json")
    log_dir: Path = Path(os.getenv("LOG_DIR", "logs"))


def get_settings() -> Settings:
    """Return a frozen settings snapshot."""
    return Settings()
