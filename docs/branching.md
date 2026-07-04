# Git Branching Strategy

**Project:** Interactive Sales Analytics Dashboard for Small Businesses
**Course:** MSIT 5290 Capstone, University of the People
**Author:** Rev. Drew Brown

This project follows a **simplified Git Flow** appropriate for a single-developer
capstone: two protected long-lived branches (`main`, `develop`) plus short-lived
topic branches (`feature/*`, `bugfix/*`, `hotfix/*`).

## Long-lived branches

| Branch    | Purpose                                                | Protection                         |
|-----------|--------------------------------------------------------|------------------------------------|
| `main`    | Production-ready; every commit is a tagged release     | No direct pushes; PR + review only |
| `develop` | Integration branch for tested features                 | No direct pushes; PR + CI required |

## Short-lived branches

| Prefix     | Source     | Merge target       | Naming example              |
|------------|------------|--------------------|-----------------------------|
| `feature/` | `develop`  | `develop`          | `feature/rfm-segmentation`  |
| `bugfix/`  | `develop`  | `develop`          | `bugfix/cleaning-nan-ids`   |
| `hotfix/`  | `main`     | `main` + `develop` | `hotfix/audit-log-write`    |
| `docs/`    | `develop`  | `develop`          | `docs/branching-policy`     |

## Merge policy

1. Every pull request must pass GitHub Actions CI (pytest, flake8, coverage
   ≥ 60%, secret scan).
2. Squash-merge is the default. This keeps `develop` history linear and easy
   to bisect for defects.
3. `hotfix/*` branches are merged with a merge commit (not squashed) so the
   fix appears identically in `main` and `develop`.
4. Release tags on `main` follow semantic versioning: `v0.1.0`, `v0.2.0`, etc.

## Weekly cadence

- **Weeks 3–5:** feature branches for M9 ingestion, M10 cleaning, M6
  pseudonymization, M7 protected store, M3/M4 analytics, M1/M2 Streamlit UI.
- **Week 6:** verification queries, small-group suppression fixes on
  `bugfix/*` branches.
- **Week 7:** feature freeze; only `bugfix/*` and `docs/*` accepted.
- **Week 8:** cut `v1.0.0` on `main`; final report and presentation delivered.

## Commit message convention

Follows **type(scope): subject** (Conventional Commits style):

```
feat(analytics): add K-Means clustering with silhouette scoring
fix(cleaning): handle NaN CustomerID rows before RFM aggregation
sec(access): enforce small-group suppression at cohort size < 5
docs(readme): add Streamlit quickstart instructions
test(pseudonymize): cover HMAC key rotation edge cases
chore(ci): pin flake8 version and enable coverage upload
```

Common `scope` values map to the modules: `ingestion`, `cleaning`,
`pseudonymize`, `storage`, `access`, `analytics`, `visualization`, `ui`,
`config`, `logs`, `ci`, `docs`.
