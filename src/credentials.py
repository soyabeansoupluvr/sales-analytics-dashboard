"""Credential store for the dashboard login widget.

This module stays separate from src.access. src.access decides what an
authenticated Actor may see. This module identifies who is signing in by
mapping usernames to roles and bcrypt-hashed passwords for
streamlit-authenticator.

Rows live in the app_user table created by migration 002 and seeded by
migration 003. Those migrations run through the M14 SchemaMigrator, so the
first Database.connect call on a fresh checkout creates the demo accounts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from src.access import Role
from src.config import Settings
from src.logs import Database

# Runtime copy of Role's legal values for validating rows read from SQLite.
_VALID_ROLES: frozenset[str] = frozenset({"viewer", "analyst", "admin"})


class CredentialStoreError(Exception):
    """Raised when a credential row fails validation.

    Invalid roles are treated as store corruption, not coerced to a default.
    """


@dataclass(frozen=True, slots=True)
class UserRecord:
    """Login account loaded from app_user."""

    username: str
    password_hash: str
    role: Role
    display_name: str


class CredentialStore(Protocol):
    """Read-only credential lookup interface used by the app and test fakes."""

    def load_all(self) -> tuple[UserRecord, ...]:
        """Return every login-eligible account, in a stable order."""
        ...

    def find(self, username: str) -> UserRecord | None:
        """Return the record for ``username`` or None when absent."""
        ...


class SqliteCredentialStore:
    """Credential store backed by the app_user table.

    Database owns connection and migration setup, matching the M14 AuditLog wiring.
    """

    def __init__(self, database: Database) -> None:
        self._database = database

    def load_all(self) -> tuple[UserRecord, ...]:
        with self._database.connect() as connection:
            rows = connection.execute(
                "SELECT username, password_hash, role, display_name"
                " FROM app_user"
                " ORDER BY username"
            ).fetchall()
        return tuple(_row_to_record(row) for row in rows)

    def find(self, username: str) -> UserRecord | None:
        with self._database.connect() as connection:
            row = connection.execute(
                "SELECT username, password_hash, role, display_name"
                " FROM app_user"
                " WHERE username = ?",
                (username,),
            ).fetchone()
        if row is None:
            return None
        return _row_to_record(row)


def _row_to_record(row: Any) -> UserRecord:
    """Convert one SQLite row to a validated UserRecord."""

    username = str(row["username"])
    password_hash = str(row["password_hash"])
    role = str(row["role"])
    display_name = str(row["display_name"])
    if role not in _VALID_ROLES:
        raise CredentialStoreError(
            f"app_user row for {username!r} has invalid role {role!r};"
            f" expected one of {sorted(_VALID_ROLES)}"
        )
    return UserRecord(
        username=username,
        password_hash=password_hash,
        role=role,  # type: ignore[arg-type]
        display_name=display_name,
    )


def build_authenticator_credentials(store: CredentialStore) -> dict:
    """Return the credentials mapping streamlit-authenticator consumes.

    The store owns usernames, display names, and password hashes. Widget
    bookkeeping fields are initialized here for each account.
    """

    users = store.load_all()
    return {
        "usernames": {
            record.username: {
                "name": record.display_name,
                "password": record.password_hash,
                "email": f"{record.username}@local",
                "failed_login_attempts": 0,
                "logged_in": False,
            }
            for record in users
        }
    }


def role_for_username(store: CredentialStore, username: str) -> Role:
    """Return the role for username, raising ValueError when absent."""

    record = store.find(username)
    if record is None:
        raise ValueError(f"unknown username: {username!r}")
    return record.role


def default_store(settings: Settings) -> SqliteCredentialStore:
    """Return the default SQLite-backed credential store."""

    return SqliteCredentialStore(Database(settings))


__all__ = [
    "CredentialStore",
    "CredentialStoreError",
    "SqliteCredentialStore",
    "UserRecord",
    "build_authenticator_credentials",
    "default_store",
    "role_for_username",
]
