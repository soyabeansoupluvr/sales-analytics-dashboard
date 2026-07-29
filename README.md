# Interactive Sales Analytics Dashboard for Small Businesses

A Python/Streamlit decision-support prototype that ingests retail transaction
data from the [UCI Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail),
pseudonymizes customer identifiers, computes RFM-based customer segments, and
presents role-scoped, ethics- and security-by-design KPI dashboards for
small-business owners, retail analysts, and administrators.

**MSIT 5290: Capstone Project — University of the People**
Author: Rev. Drew Brown
Instructor: Dr. Sirisha Pavuluri
Term: July 2026

---

## Architecture at a Glance

The application is a layered modular monolith with four vertical layers and a
cross-cutting Security & Ethics band:

| Layer                  | Modules                                              |
|------------------------|------------------------------------------------------|
| Presentation           | M1 Streamlit UI & Layout, M2 Presentation Controller |
| Analytics Service      | M3 Metrics & KPI, M4 Segmentation (RFM), M5 Visualization & Explain |
| Protected Data         | M6 Pseudonymization, M7 Protected Store, M8 Access Control & Audit |
| Ingestion & Validation | M9 Ingestion & Validation, M10 Cleaning / ETL, M11 Config & Secrets |
| DevOps & Observability | M12 Git & GitHub, M13 GitHub Actions CI, M14 Logging & Observability |

Cross-cutting concerns (authentication, RBAC, encryption, audit logging, bias
sensitivity, small-group suppression) are enforced by every layer and align to
[OWASP Top 10 (2025)](https://owasp.org/Top10/2025/), [NIST SSDF v1.1](https://doi.org/10.6028/NIST.SP.800-218),
and [GDPR](https://eur-lex.europa.eu/eli/reg/2016/679/oj) Privacy-by-Design.

See `design/architecture.png` for the full diagram and `design/architecture.drawio`
for module-by-module detail.

---

## Project Structure

```
sales-analytics-dashboard/
├── src/
│   ├── __init__.py
│   ├── app.py            # M1 + M2 — Streamlit entry point, four-tab layout, auth wiring
│   ├── analytics.py      # M3 + M4 — KPIs and RFM K-Means segmentation with stability holdout
│   ├── visualization.py  # M5  — Labeled charts with stability captions
│   ├── ingestion.py      # M9  — CSV/XLSX ingestion + validation
│   ├── cleaning.py       # M10 — Cancellations, returns, derived revenue
│   ├── pseudonymize.py   # M6  — HMAC-SHA256 CustomerID pseudonymization
│   ├── storage.py        # M7  — SQLite protected store with role-scoped read_view
│   ├── access.py         # M8  — RBAC and session throttle
│   ├── credentials.py    # M2  — Credential loader for streamlit-authenticator
│   ├── config.py         # M11 — Environment + secrets loader with strict env-parity
│   ├── logs.py           # M14 — Structured JSON logging + hash-chained audit journal
│   └── schema/
│       ├── __init__.py
│       ├── 001_audit.sql        # Audit journal schema
│       ├── 002_users.sql        # app_user table
│       └── 003_seed_demo_users.sql  # Seed rows for the demo accounts
├── data/
│   ├── raw/              # Input file (gitignored, .gitkeep tracked)
│   └── processed/        # SQLite protected store (gitignored, .gitkeep tracked)
├── docs/
│   ├── input_schema.md   # Input file and column contract
│   └── branching.md      # Branching model and PR policy
├── design/
│   ├── architecture.drawio
│   └── architecture.png
├── notebooks/            # Reserved for exploratory notebooks; intentionally empty
├── tests/
│   ├── conftest.py
│   ├── test_access.py
│   ├── test_analytics.py
│   ├── test_app.py
│   ├── test_audit.py
│   ├── test_cleaning.py
│   ├── test_config.py
│   ├── test_credentials.py
│   ├── test_ingestion.py
│   ├── test_logs_db.py
│   ├── test_pseudonymize.py
│   ├── test_rfm.py
│   ├── test_smoke.py
│   ├── test_storage.py
│   └── test_visualization.py
├── .github/workflows/ci.yml
├── .env.example
├── .gitignore
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

## Quickstart

```bash
# 1. Clone
git clone https://github.com/soyabeansoupluvr/sales-analytics-dashboard.git
cd sales-analytics-dashboard

# 2. Virtual environment (Python 3.12 recommended)
python3.12 -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows

# 3. Install
pip install -r requirements.txt          # runtime only
pip install -r requirements-dev.txt      # add-on for tests, lint, and formatter

# 4. Configuration and secrets
cp .env.example .env
# Edit .env: set PSEUDONYM_KEY (32 bytes hex) at minimum.
# Generate with: python -c "import secrets; print(secrets.token_hex(32))"

# 5. Provide the input file
# Drop a conforming .csv or .xlsx into data/raw/. The classroom target is
# "Online Retail.xlsx" from the UCI archive. See docs/input_schema.md for
# the full contract.

# 6. Run the dashboard
streamlit run src/app.py
# Defaults to http://localhost:8501
```

## Input Data

The application accepts a `.csv` or `.xlsx` conforming to the UCI Online Retail
schema. Column names, types, size limits, and per-row validation rules are
documented in [`docs/input_schema.md`](docs/input_schema.md). Drop the file
into `data/raw/`; the app resolves it by scanning for a supported extension.

## Testing

```bash
python -m pytest                          # full suite
python -m pytest --cov=src                # with coverage
python -m flake8 --max-line-length=100 src tests
```

The CI pipeline (`.github/workflows/ci.yml`) runs the same commands on every
push and pull request with a `--cov-fail-under=60` gate.

## Branching Strategy (Summary)

Simplified Git Flow with two long-lived branches and short-lived topic branches:

| Branch      | Purpose                                    | Protection                     |
|-------------|--------------------------------------------|--------------------------------|
| `main`      | Production-ready; signed release tags      | Protected — PR + review only   |
| `develop`   | Integration branch for tested features     | Protected — PR + CI required   |
| `feature/*` | New features (e.g. `feature/rfm-scoring`)  | Merged into `develop`          |
| `bugfix/*`  | Non-urgent bug fixes                       | Merged into `develop`          |
| `hotfix/*`  | Emergency production fixes                 | Merged into `main` + `develop` |

Full policy in [`docs/branching.md`](docs/branching.md).

## Commit Convention

Follows `type(scope): subject` (Conventional Commits style):

```
feat(analytics): add K-Means clustering with silhouette scoring
fix(cleaning): handle NaN CustomerID rows before RFM aggregation
sec(access): enforce small-group suppression at cohort size < 5
docs(architecture): update layer diagram
test(pseudonymize): cover HMAC key rotation edge cases
```

## Contributing

1. Branch from `develop`: `git checkout -b feature/short-description develop`
2. Commit with descriptive messages
3. Push and open a Pull Request into `develop`
4. Ensure CI passes (pytest, flake8, coverage ≥ 60%, secret scan)
5. Request review; squash-merge on approval

## License

MIT — see [`LICENSE`](LICENSE).

## Author

Rev. Drew Brown — MSIT 5290 Capstone, University of the People, July 2026.
