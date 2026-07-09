"""Tests for M14 database and migration plumbing.

These tests cover the SQLite connection factory and schema migrator separately
from application logging. Each test uses a private Settings instance pointed
at a temporary SQLite file, so CI never touches data/processed/dashboard.db.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src.config import Settings
from src.logs import (
    Database,
    DatabaseError,
    Migration,
    MigrationError,
    SchemaMigrator,
    _path_from_sqlite_url,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _settings_for(tmp_path: Path, filename: str = "test.db") -> Settings:
    """Return a Settings pointing at a per-test SQLite file."""

    db_path = tmp_path / filename
    return Settings(sqlite_url=f"sqlite:///{db_path.as_posix()}")


def _write_migration(directory: Path, name: str, sql: str) -> None:
    (directory / name).write_text(sql, encoding="utf-8")


# --------------------------------------------------------------------------- #
# _path_from_sqlite_url
# --------------------------------------------------------------------------- #


class TestPathFromSqliteUrl:
    def test_extracts_relative_path(self) -> None:
        assert _path_from_sqlite_url("sqlite:///data/processed/dashboard.db") == Path(
            "data/processed/dashboard.db"
        )

    def test_extracts_absolute_posix_path(self) -> None:
        assert _path_from_sqlite_url("sqlite:////tmp/foo.db") == Path("/tmp/foo.db")

    def test_strips_surrounding_whitespace(self) -> None:
        assert _path_from_sqlite_url("  sqlite:///x.db\n") == Path("x.db")

    @pytest.mark.parametrize(
        "url",
        [
            "",
            "sqlite://",
            "sqlite:///",
            "postgresql:///x.db",
            "not-a-url",
        ],
    )
    def test_rejects_malformed_urls(self, url: str) -> None:
        with pytest.raises(DatabaseError):
            _path_from_sqlite_url(url)


# --------------------------------------------------------------------------- #
# SchemaMigrator
# --------------------------------------------------------------------------- #


class TestSchemaMigrator:
    def test_discover_ignores_non_migration_files(self, tmp_path: Path) -> None:
        _write_migration(tmp_path, "001_init.sql", "CREATE TABLE a (id INTEGER);")
        _write_migration(tmp_path, "__init__.py", '""" package marker """')
        _write_migration(tmp_path, "notes.txt", "scratch")
        _write_migration(tmp_path, "backup.sql.bak", "-- ignored")
        _write_migration(tmp_path, "abc_bad.sql", "-- wrong prefix")

        migrations = SchemaMigrator(tmp_path).discover()

        assert [m.name for m in migrations] == ["001_init.sql"]

    def test_discover_returns_empty_when_directory_missing(self, tmp_path: Path) -> None:
        migrator = SchemaMigrator(tmp_path / "does-not-exist")
        assert migrator.discover() == ()

    def test_discover_sorts_by_version(self, tmp_path: Path) -> None:
        _write_migration(tmp_path, "003_c.sql", "CREATE TABLE c (id INTEGER);")
        _write_migration(tmp_path, "001_a.sql", "CREATE TABLE a (id INTEGER);")
        _write_migration(tmp_path, "002_b.sql", "CREATE TABLE b (id INTEGER);")

        versions = [m.version for m in SchemaMigrator(tmp_path).discover()]

        assert versions == [1, 2, 3]

    def test_duplicate_versions_raise(self, tmp_path: Path) -> None:
        _write_migration(tmp_path, "001_first.sql", "CREATE TABLE a (id INTEGER);")
        _write_migration(tmp_path, "001_second.sql", "CREATE TABLE b (id INTEGER);")

        with pytest.raises(MigrationError, match="Duplicate migration version"):
            SchemaMigrator(tmp_path).discover()

    def test_apply_records_and_is_idempotent(self, tmp_path: Path) -> None:
        _write_migration(tmp_path, "001_init.sql", "CREATE TABLE t (id INTEGER);")
        migrator = SchemaMigrator(tmp_path)

        connection = sqlite3.connect(":memory:")
        try:
            first = migrator.apply(connection)
            second = migrator.apply(connection)

            assert first == (1,)
            assert second == ()

            recorded = [
                row[0]
                for row in connection.execute("SELECT version FROM schema_version ORDER BY version")
            ]
            assert recorded == [1]
        finally:
            connection.close()

    def test_apply_wraps_sqlite_errors_in_migration_error(self, tmp_path: Path) -> None:
        _write_migration(tmp_path, "001_broken.sql", "CREATE TABLE oops (;")
        migrator = SchemaMigrator(tmp_path)
        connection = sqlite3.connect(":memory:")
        try:
            with pytest.raises(MigrationError, match="001_broken.sql"):
                migrator.apply(connection)
        finally:
            connection.close()

    def test_apply_runs_new_migrations_added_later(self, tmp_path: Path) -> None:
        _write_migration(tmp_path, "001_a.sql", "CREATE TABLE a (id INTEGER);")
        migrator = SchemaMigrator(tmp_path)

        connection = sqlite3.connect(":memory:")
        try:
            assert migrator.apply(connection) == (1,)

            _write_migration(tmp_path, "002_b.sql", "CREATE TABLE b (id INTEGER);")
            assert migrator.apply(connection) == (2,)

            tables = {
                row[0]
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            assert {"a", "b", "schema_version"}.issubset(tables)
        finally:
            connection.close()


# --------------------------------------------------------------------------- #
# Migration dataclass
# --------------------------------------------------------------------------- #


class TestMigration:
    def test_migration_is_frozen(self) -> None:
        migration = Migration(version=1, name="001_x.sql", sql="SELECT 1;")
        with pytest.raises(Exception):
            migration.version = 2  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Database
# --------------------------------------------------------------------------- #


class TestDatabase:
    def test_connect_creates_file_and_parent_directory(self, tmp_path: Path) -> None:
        settings = Settings(
            sqlite_url=f"sqlite:///{(tmp_path / 'nested' / 'sub' / 'x.db').as_posix()}"
        )
        db = Database(settings=settings)

        with db.connect() as connection:
            connection.execute("SELECT 1")

        assert db.path.exists()
        assert db.path.parent.is_dir()

    def test_connect_sets_expected_pragmas(self, tmp_path: Path) -> None:
        db = Database(settings=_settings_for(tmp_path))
        with db.connect() as connection:
            (fk_enabled,) = connection.execute("PRAGMA foreign_keys").fetchone()
            (journal_mode,) = connection.execute("PRAGMA journal_mode").fetchone()

        assert fk_enabled == 1
        assert journal_mode.lower() == "wal"

    def test_connect_applies_bundled_audit_migration(self, tmp_path: Path) -> None:
        db = Database(settings=_settings_for(tmp_path))
        with db.connect() as connection:
            tables = {
                row[0]
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            columns = {row[1] for row in connection.execute("PRAGMA table_info(audit_event)")}

        assert "audit_event" in tables
        assert "schema_version" in tables
        assert {
            "id",
            "timestamp",
            "actor",
            "action",
            "outcome",
            "resource",
            "details_json",
            "previous_hash",
            "event_hash",
        }.issubset(columns)

    def test_migrations_only_run_once_per_instance(self, tmp_path: Path) -> None:
        settings = _settings_for(tmp_path)
        db = Database(settings=settings)

        with db.connect() as connection:
            connection.execute("SELECT 1")
        with db.connect() as connection:
            rows = list(connection.execute("SELECT version FROM schema_version"))

        assert len(rows) == 1

    def test_migrations_survive_across_instances(self, tmp_path: Path) -> None:
        settings = _settings_for(tmp_path)

        first = Database(settings=settings)
        with first.connect() as connection:
            connection.execute("SELECT 1")

        second = Database(settings=settings)
        with second.connect() as connection:
            rows = list(connection.execute("SELECT version FROM schema_version"))

        assert len(rows) == 1

    def test_connect_context_manager_closes_connection(self, tmp_path: Path) -> None:
        db = Database(settings=_settings_for(tmp_path))
        with db.connect() as connection:
            pass

        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")

    def test_context_manager_rolls_back_on_exception(self, tmp_path: Path) -> None:
        db = Database(settings=_settings_for(tmp_path))

        with pytest.raises(RuntimeError):
            with db.connect() as connection:
                connection.execute(
                    "INSERT INTO schema_version (version, name, applied_at)" " VALUES (?, ?, ?)",
                    (999, "phantom.sql", "2026-01-01T00:00:00+00:00"),
                )
                raise RuntimeError("boom")

        # A fresh connection must not see the rolled-back row.
        with db.connect() as connection:
            rows = [row[0] for row in connection.execute("SELECT version FROM schema_version")]
        assert 999 not in rows

    def test_path_property_matches_settings_url(self, tmp_path: Path) -> None:
        settings = _settings_for(tmp_path, "custom.db")
        db = Database(settings=settings)
        assert db.path == tmp_path / "custom.db"

    def test_settings_url_override_targets_custom_path(self, tmp_path: Path) -> None:
        """Settings.sqlite_url controls the database file path.

        This checks that Database uses the configured SQLite URL directly, so an
        operator can change the database location without changing M14 code.
        """

        target = tmp_path / "custom_target.db"
        settings = Settings(sqlite_url=f"sqlite:///{target.as_posix()}")
        db = Database(settings=settings)

        with db.connect() as connection:
            connection.execute("SELECT 1")

        assert target.exists()

    def test_rejects_malformed_settings_url(self, tmp_path: Path) -> None:
        # Validation moved up the stack in M11: Settings now refuses malformed
        # SQLITE_URLs at construction, so Database never sees a bad value.
        from src.config import ConfigError

        with pytest.raises(ConfigError):
            Settings(sqlite_url="not-a-sqlite-url")
