-- 002_users.sql
--
-- Creates the credential store consumed by src.credentials.
-- Rows in this table drive the streamlit-authenticator login widget
-- rendered by M1/M2. The audit trail (audit_event) captures login
-- outcomes; this table captures only who is permitted to log in.
--
-- Column contract (documented here, enforced in Python):
--   * username       - Case-sensitive login identifier. Primary key.
--   * password_hash  - bcrypt digest produced by streamlit-authenticator's
--                       Hasher. Length is not fixed here so a future
--                       cost-factor bump does not require another
--                       migration.
--   * role           - One of viewer, analyst, admin. Enforced by the
--                       CHECK constraint below so a malformed row
--                       cannot silently grant the wrong posture.
--   * display_name   - Human-readable name surfaced by the login widget.
--   * created_at     - ISO-8601 UTC string, seconds precision. Seed rows
--                       use a fixed timestamp so the migration remains
--                       deterministic and re-running it is a no-op.

CREATE TABLE IF NOT EXISTS app_user (
    username TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('viewer', 'analyst', 'admin')),
    display_name TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_app_user_role
    ON app_user (role);
