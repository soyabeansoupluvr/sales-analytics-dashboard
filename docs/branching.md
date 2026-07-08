# Branching model and branch protection

The repository follows a trunk-with-feature-branches model that matches the
Git workflow described in Document 1.

## Long-lived branches

| Branch    | Purpose                                                              |
| --------- | -------------------------------------------------------------------- |
| `main`    | Release-ready code. Every commit is expected to pass CI.             |
| `develop` | Integration branch. Feature branches merge here first for CI checks. |

## Short-lived branches

Feature branches follow the pattern `feature/m<module-number>-<slug>`, for
example `feature/m9-ingestion`, `feature/m10-cleaning`, or
`feature/m13-ci-pipeline`. Bug-fix branches follow the pattern
`fix/<slug>`.

Every short-lived branch is opened from `develop`, keeps a linear history
where practical, and closes through a pull request rather than a direct
push.

## Branch protection rules

The following rules are configured in **Settings -> Branches** for both
`main` and `develop`:

1. **Require a pull request before merging.** Direct pushes to `main` and
   `develop` are disabled for every contributor, including the repository
   owner. Every change lands through a reviewable pull request.
2. **Require status checks to pass before merging.** The `test` job in
   `.github/workflows/ci.yml` is marked as a required status check. A pull
   request cannot be merged while any of the following fail:
   - `flake8` lint (max line length 100).
   - `black --check` formatting.
   - `gitleaks` secret scan.
   - `pytest` with the `--cov-fail-under=60` coverage floor.
3. **Require branches to be up to date before merging.** Pull requests
   must be rebased on the latest target branch so the CI result reflects
   the code that will actually land.
4. **Require conversation resolution before merging.** Every review
   comment must be resolved before merge.
5. **Restrict force pushes and deletions.** `main` and `develop` cannot be
   force-pushed or deleted, which preserves audit history.

## Local workflow

```bash
# Start work on module M13.
git switch develop
git pull --ff-only
git switch -c feature/m13-ci-pipeline

# ... edit, commit ...

# Before pushing, run the same checks CI will run.
flake8 src/ tests/ --max-line-length=100 --statistics
black --check src/ tests/
pytest --cov=src --cov-fail-under=60

git push -u origin feature/m13-ci-pipeline
# Open a pull request into develop; CI runs automatically.
```

The commands above mirror the CI workflow exactly, so a green local run is
a strong predictor of a green pull request.
