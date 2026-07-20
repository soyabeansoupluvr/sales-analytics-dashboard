"""Tests for src.credentials.

Covers the SQLite credential store, SchemaMigrator integration, credential
mapping for streamlit-authenticator, and username-to-role lookup. Each test
uses temporary Settings so it does not touch the shared dashboard database.
"""

from __future__ import annotations

from dataclasses import dataclass, FrozenInstanceError
from pathlib import Path

import pytest
from streamlit_authenticator.utilities.hasher import Hasher

import sqlite3
import re

from src.config import Settings
from src.credentials import (
    CredentialStore,
    CredentialStoreError,
    SqliteCredentialStore,
    UserRecord,
    build_authenticator_credentials,
    default_store,
    role_for_username,
)
from src.logs import Database


_BCRYPT_RE = re.compile(r"^\$2[aby]\$\d{2}\$[./A-Za-z0-9]{53}$")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _settings_for(tmp_path: Path, filename: str = "test.db") -> Settings:
    """Return Settings for a temporary SQLite database."""

    db_path = tmp_path / filename
    return Settings(sqlite_url=f"sqlite:///{db_path.as_posix()}")


def _seeded_store(tmp_path: Path) -> tuple[SqliteCredentialStore, Database]:
    """Return a credential store backed by a migrated test database.

    The first load triggers Database.connect, which applies migrations before
    reading seeded users.
    """

    settings = _settings_for(tmp_path)
    database = Database(settings)
    return SqliteCredentialStore(database), database


# --------------------------------------------------------------------------- #
# UserRecord value object
# --------------------------------------------------------------------------- #


class TestUserRecord:
    def test_is_frozen(self) -> None:
        record = UserRecord(
            username="analyst",
            password_hash="hash",
            role="analyst",
            display_name="Analyst",
        )
        with pytest.raises(FrozenInstanceError):
            record.username = "other"  # type: ignore[misc]

    def test_uses_slots(self) -> None:
        record = UserRecord(
            username="analyst",
            password_hash="hash",
            role="analyst",
            display_name="Analyst",
        )
        # Exact exception varies because the dataclass is both frozen and slotted.
        with pytest.raises((AttributeError, TypeError)):
            record.extra = "x"  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Migrations 002 and 003 apply and seed
# --------------------------------------------------------------------------- #


class TestMigrationsApply:
    def test_first_connect_creates_app_user_table(self, tmp_path: Path) -> None:
        _, database = _seeded_store(tmp_path)
        with database.connect() as connection:
            rows = connection.execute(
                "SELECT name FROM sqlite_master" " WHERE type = 'table' AND name = 'app_user'"
            ).fetchall()
        assert len(rows) == 1

    def test_schema_version_records_002_and_003(self, tmp_path: Path) -> None:
        _, database = _seeded_store(tmp_path)
        with database.connect() as connection:
            versions = {
                int(row[0]) for row in connection.execute("SELECT version FROM schema_version")
            }
        assert {1, 2, 3}.issubset(versions)

    def test_re_running_migrations_is_idempotent(self, tmp_path: Path) -> None:
        store, _database = _seeded_store(tmp_path)
        first = store.load_all()

        # Reopening the same file should not re-apply seed rows.
        settings = _settings_for(tmp_path)
        second_database = Database(settings)
        second_store = SqliteCredentialStore(second_database)
        second = second_store.load_all()
        assert first == second


# --------------------------------------------------------------------------- #
# Seed rows
# --------------------------------------------------------------------------- #


class TestSeededRows:
    def test_load_all_returns_three_users(self, tmp_path: Path) -> None:
        store, _ = _seeded_store(tmp_path)
        records = store.load_all()
        usernames = {r.username for r in records}
        assert usernames == {"analyst", "mgr", "adm"}

    def test_load_all_is_sorted_by_username(self, tmp_path: Path) -> None:
        store, _ = _seeded_store(tmp_path)
        records = store.load_all()
        assert [r.username for r in records] == sorted(r.username for r in records)

    def test_role_assignments(self, tmp_path: Path) -> None:
        store, _ = _seeded_store(tmp_path)
        by_name = {r.username: r for r in store.load_all()}
        assert by_name["analyst"].role == "analyst"
        assert by_name["mgr"].role == "analyst"
        assert by_name["adm"].role == "admin"

    def test_display_names(self, tmp_path: Path) -> None:
        store, _ = _seeded_store(tmp_path)
        by_name = {r.username: r for r in store.load_all()}
        assert by_name["analyst"].display_name == "Analyst"
        assert by_name["mgr"].display_name == "Manager"
        assert by_name["adm"].display_name == "Administrator"

    def test_seeded_passwords_are_bcrypt_hashes(self, tmp_path: Path) -> None:
        store, _ = _seeded_store(tmp_path)

        for record in store.load_all():
            assert _BCRYPT_RE.fullmatch(record.password_hash)
            assert record.password_hash != record.username
            assert record.password_hash != record.display_name

    def test_seeded_hashes_reject_wrong_password(self, tmp_path: Path) -> None:
        store, _ = _seeded_store(tmp_path)

        for record in store.load_all():
            assert Hasher.check_pw("wrong-password", record.password_hash) is False


# --------------------------------------------------------------------------- #
# SqliteCredentialStore.find
# --------------------------------------------------------------------------- #


class TestFind:
    def test_find_returns_record_for_seeded_username(self, tmp_path: Path) -> None:
        store, _ = _seeded_store(tmp_path)
        record = store.find("analyst")
        assert record is not None
        assert record.username == "analyst"
        assert record.role == "analyst"

    def test_find_returns_none_for_unknown_username(self, tmp_path: Path) -> None:
        store, _ = _seeded_store(tmp_path)
        assert store.find("nobody") is None

    def test_find_is_case_sensitive(self, tmp_path: Path) -> None:
        store, _ = _seeded_store(tmp_path)
        # SQLite TEXT PRIMARY KEY is case-sensitive by default.
        assert store.find("Analyst") is None


# --------------------------------------------------------------------------- #
# Fail-fast on invalid role rows
# --------------------------------------------------------------------------- #


class TestInvalidRoleFailFast:
    def test_check_constraint_rejects_bad_role(self, tmp_path: Path) -> None:
        """The schema CHECK constraint rejects unknown roles."""

        _, database = _seeded_store(tmp_path)
        with database.connect() as connection:
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(
                    "INSERT INTO app_user"
                    " (username, password_hash, role, display_name, created_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    ("rogue", "x", "superuser", "Rogue", "2026-01-01T00:00:00+00:00"),
                )

    def test_load_all_raises_on_role_that_slipped_the_check(self, tmp_path: Path) -> None:
        """The Python layer rejects invalid roles that already exist in storage."""

        _, database = _seeded_store(tmp_path)

        # Recreate the table without its CHECK constraint to simulate tampering or an
        # out-of-band write.
        with database.connect() as connection:
            connection.execute("DROP TABLE app_user")
            connection.execute(
                "CREATE TABLE app_user ("
                " username TEXT PRIMARY KEY,"
                " password_hash TEXT NOT NULL,"
                " role TEXT NOT NULL,"
                " display_name TEXT NOT NULL,"
                " created_at TEXT NOT NULL"
                ")"
            )
            connection.execute(
                "INSERT INTO app_user VALUES (?, ?, ?, ?, ?)",
                ("rogue", "hash", "superuser", "Rogue", "2026-01-01T00:00:00+00:00"),
            )

        store = SqliteCredentialStore(database)
        with pytest.raises(CredentialStoreError) as exc:
            store.load_all()
        assert "superuser" in str(exc.value)


# --------------------------------------------------------------------------- #
# build_authenticator_credentials
# --------------------------------------------------------------------------- #


class TestBuildAuthenticatorCredentials:
    def test_shape_matches_stauth_expectation(self, tmp_path: Path) -> None:
        store, _ = _seeded_store(tmp_path)
        creds = build_authenticator_credentials(store)
        assert set(creds.keys()) == {"usernames"}
        assert set(creds["usernames"].keys()) == {"analyst", "mgr", "adm"}

    def test_each_entry_has_the_five_required_fields(self, tmp_path: Path) -> None:
        store, _ = _seeded_store(tmp_path)
        creds = build_authenticator_credentials(store)
        for entry in creds["usernames"].values():
            assert set(entry.keys()) == {
                "name",
                "password",
                "email",
                "failed_login_attempts",
                "logged_in",
            }

    def test_entries_carry_hash_and_display_name(self, tmp_path: Path) -> None:
        store, _ = _seeded_store(tmp_path)
        creds = build_authenticator_credentials(store)
        analyst = creds["usernames"]["analyst"]
        assert analyst["name"] == "Analyst"
        assert _BCRYPT_RE.fullmatch(analyst["password"])
        assert analyst["email"] == "analyst@local"
        assert analyst["failed_login_attempts"] == 0
        assert analyst["logged_in"] is False

    def test_works_with_any_credentialstore_implementation(self) -> None:
        """The helper depends on the Protocol, not the SQLite class."""

        @dataclass
        class _FakeStore:
            records: tuple[UserRecord, ...]

            def load_all(self) -> tuple[UserRecord, ...]:
                return self.records

            def find(self, username: str) -> UserRecord | None:
                for r in self.records:
                    if r.username == username:
                        return r
                return None

        fake: CredentialStore = _FakeStore(
            (
                UserRecord(
                    username="alice",
                    password_hash="$2b$12$fakefakefakefakefakefu",
                    role="analyst",
                    display_name="Alice",
                ),
            )
        )
        creds = build_authenticator_credentials(fake)
        assert set(creds["usernames"].keys()) == {"alice"}


# --------------------------------------------------------------------------- #
# role_for_username
# --------------------------------------------------------------------------- #


class TestRoleForUsername:
    @pytest.mark.parametrize(
        "username, expected",
        [("analyst", "analyst"), ("mgr", "analyst"), ("adm", "admin")],
    )
    def test_returns_role_for_seeded_username(
        self, tmp_path: Path, username: str, expected: str
    ) -> None:
        store, _ = _seeded_store(tmp_path)
        assert role_for_username(store, username) == expected

    def test_raises_value_error_for_unknown_username(self, tmp_path: Path) -> None:
        store, _ = _seeded_store(tmp_path)
        with pytest.raises(ValueError) as exc:
            role_for_username(store, "nobody")
        assert "nobody" in str(exc.value)


# --------------------------------------------------------------------------- #
# default_store
# --------------------------------------------------------------------------- #


class TestDefaultStore:
    def test_returns_sqlite_credential_store_bound_to_settings(self, tmp_path: Path) -> None:
        settings = _settings_for(tmp_path)
        store = default_store(settings)

        assert isinstance(store, SqliteCredentialStore)
        assert {r.username for r in store.load_all()} == {"analyst", "mgr", "adm"}
