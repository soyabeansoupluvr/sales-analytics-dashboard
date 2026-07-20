-- 001_audit.sql
--
-- Creates the audit event table used by M14.
-- Events are append-only in application code and linked by hash values.
--
-- Column contract (documented here, enforced in Python):
--   * timestamp       - ISO-8601 UTC string, seconds precision.
--   * actor           - Free-text principal identifier (username or "system").
--   * action          - Value drawn from src.logs.AuditAction.
--   * outcome         - Value drawn from src.logs.AuditOutcome.
--   * resource        - Optional resource identifier.
--   * details_json    - Canonical JSON of the event's structured payload.
--   * previous_hash   - Hex-encoded SHA-256 of the prior row's event_hash,
--                        or NULL for the genesis row.
--   * event_hash      - Hex-encoded SHA-256 of this row's canonical fields.

CREATE TABLE IF NOT EXISTS audit_event (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    outcome TEXT NOT NULL,
    resource TEXT,
    details_json TEXT NOT NULL,
    previous_hash TEXT,
    event_hash TEXT NOT NULL UNIQUE
);

-- Common lookups: recent events and events by actor.

CREATE INDEX IF NOT EXISTS ix_audit_event_timestamp
    ON audit_event (timestamp);

CREATE INDEX IF NOT EXISTS ix_audit_event_actor_timestamp
    ON audit_event (actor, timestamp);
