-- 003_seed_demo_users.sql
--
-- Seeds the three demo accounts referenced by the capstone plan:
--   analyst -> analyst role
--   mgr     -> analyst role
--   adm     -> admin role
--
-- Passwords are bcrypt-hashed (streamlit-authenticator's Hasher, cost
-- factor 12). The plaintext values are shared with graders out-of-band
-- and are deliberately not recorded in the repository. The seed is
-- intentionally split from 002_users.sql so a future non-classroom
-- deployment can apply 002 while skipping 003.
--
-- INSERT OR IGNORE keeps the migration idempotent: re-running the
-- migration against a database whose usernames already exist changes
-- nothing. The created_at values are fixed strings for the same reason.

INSERT OR IGNORE INTO app_user (username, password_hash, role, display_name, created_at)
VALUES (
    'analyst',
    '$2b$12$VJPiC88nK8nK589gTkFwNusPp4w45IAuKm.YCQda4FUQuzVA0aEaG',
    'analyst',
    'Analyst',
    '2026-07-19T00:00:00+00:00'
);

INSERT OR IGNORE INTO app_user (username, password_hash, role, display_name, created_at)
VALUES (
    'mgr',
    '$2b$12$b8uxE46UrZXjqPUcFGnSRu2Z/48a2Sv1rrHz.DnneRE8ORjn8ZzIO',
    'analyst',
    'Manager',
    '2026-07-19T00:00:00+00:00'
);

INSERT OR IGNORE INTO app_user (username, password_hash, role, display_name, created_at)
VALUES (
    'adm',
    '$2b$12$fU1/0ZcGpc.jnOGU7bMxNeUTv8pdud6uWedRdKT4QD/UQ5hKRzUHW',
    'admin',
    'Administrator',
    '2026-07-19T00:00:00+00:00'
);
