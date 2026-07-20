-- 003_seed_demo_users.sql
--
-- Seeds the three demo accounts documented in the capstone plan:
--   analyst / 1111  -> analyst role
--   mgr     / 2222  -> analyst role (the architecture diagram treats
--                       manager and analyst as sharing the same access
--                       posture)
--   adm     / 3457  -> admin role
--
-- Passwords are bcrypt-hashed (streamlit-authenticator's Hasher, cost
-- factor 12). The seed is intentionally split from 002_users.sql so a
-- future non-classroom deployment can apply 002 while skipping 003.
--
-- INSERT OR IGNORE keeps the migration idempotent: re-running the
-- migration against a database whose usernames already exist changes
-- nothing. The created_at values are fixed strings for the same reason.
--
-- SECURITY NOTE: these hashes correspond to the classroom-plan
-- passwords committed alongside them; a real deployment would replace
-- these rows before the first user login.

INSERT OR IGNORE INTO app_user (username, password_hash, role, display_name, created_at)
VALUES (
    'analyst',
    '$2b$12$Q/yl3XV4.1rvR5XVOSQOveBf9TVgVOqU.nZMpCGFhrCDzc/inAhGq',
    'analyst',
    'Analyst',
    '2026-07-19T00:00:00+00:00'
);

INSERT OR IGNORE INTO app_user (username, password_hash, role, display_name, created_at)
VALUES (
    'mgr',
    '$2b$12$73BZsGgqEZ4be4fyJS4QnOCFxD689eeloFjSsLXWtbr4B8XjTjQPa',
    'analyst',
    'Manager',
    '2026-07-19T00:00:00+00:00'
);

INSERT OR IGNORE INTO app_user (username, password_hash, role, display_name, created_at)
VALUES (
    'adm',
    '$2b$12$OQ1INYCh3dBGTRvUaooRXOa4ZM7aDrWP2GcdJ5Pmmstb0mrTWh1A6',
    'admin',
    'Administrator',
    '2026-07-19T00:00:00+00:00'
);
