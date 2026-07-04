# sales-analytics-dashboard
Interactive Sales Analytics Dashboard for Small Businesses

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

The application is a **layered modular monolith** with four vertical layers and a
cross-cutting **Security & Ethics** band:

| Layer               | Modules                                              |
|---------------------|------------------------------------------------------|
| Presentation        | M1 Streamlit UI & Layout, M2 Presentation Controller |
| Analytics Service   | M3 Metrics & KPI, M4 Segmentation (RFM), M5 Visualization & Explain |
| Protected Data      | M6 Pseudonymization, M7 Protected Store, M8 Access Control & Audit |
| Ingestion & Validation | M9 Ingestion & Validation, M10 Cleaning / ETL, M11 Config & Secrets |
| DevOps & Observability | M12 Git & GitHub, M13 GitHub Actions CI, M14 Logging & Observability |

Cross-cutting concerns (AuthN/Z, RBAC, encryption, audit logging, bias
sensitivity, small-group suppression) are enforced by every layer and align to
[OWASP Top 10 (2025)](https://owasp.org/Top10/2025/), [NIST SSDF v1.1](https://doi.org/10.6028/NIST.SP.800-218),
and [GDPR](https://eur-lex.europa.eu/eli/reg/2016/679/oj) Privacy-by-Design.

See `design/architecture.png` for the full diagram and `docs/architecture.md`
for module-by-module detail.

---

## Project Structure

```
sales-analytics-dashboard/
├── src/
│   ├── app.py            # M1 + M2 — Streamlit entry point
│   ├── ingestion.py      # M9  — CSV/XLSX ingestion + validation
│   ├── cleaning.py       # M10 — Cancellations, returns, derived revenue
│   ├── pseudonymize.py   # M6  — HMAC-keyed CustomerID pseudonymization
│   ├── storage.py        # M7  — Parquet + SQLite protected store
│   ├── access.py         # M8  — RBAC, session throttle, audit journal
│   ├── analytics.py      # M3 + M4 — KPIs and RFM K-Means segmentation
│   ├── visualization.py  # M5  — Labeled charts with formulas / exclusions
│   ├── config.py         # M11 — Environment + secrets loader
│   └── logs.py           # M14 — Structured JSON logging
├── data/
│   ├── raw/              # UCI Online Retail (gitignored)
│   └── processed/        # Encrypted parquet snapshots (gitignored)
├── docs/
│   ├── SRS.md
│   ├── architecture.md
│   └── branching.md
├── design/
│   ├── architecture.drawio
│   └── architecture.png
├── notebooks/
│   └── 01_exploratory_analysis.ipynb
├── tests/
│   ├── test_ingestion.py
│   ├── test_cleaning.py
│   ├── test_pseudonymize.py
│   └── test_analytics.py
├── .github/workflows/ci.yml
├── .env.example
├── .gitignore
├── requirements.txt
├── LICENSE
└── README.md
```

## Quickstart

```bash
# 1. Clone
git clone https://github.com/<your-username>/sales-analytics-dashboard.git
cd sales-analytics-dashboard

# 2. Virtual environment
python3 -m venv venv
source venv/bin/activate      # Linux / macOS
# venv\Scripts\activate       # Windows

# 3. Install
pip install -r requirements.txt

# 4. Configuration and secrets
cp .env.example .env
# Edit .env: set PSEUDONYM_KEY (32 bytes hex) and admin credentials

# 5. Fetch dataset (see docs/data_setup.md)
#    Place Online Retail.xlsx in data/raw/

# 6. Run the dashboard
streamlit run src/app.py
# Defaults to http://localhost:8501
```

## Branching Strategy (Summary)

Simplified Git Flow with two long-lived branches and short-lived topic branches:

| Branch        | Purpose                                    | Protection                   |
|---------------|--------------------------------------------|------------------------------|
| `main`        | Production-ready; signed release tags      | Protected — PR + review only |
| `develop`     | Integration branch for tested features     | Protected — PR + CI required |
| `feature/*`   | New features (`feature/rfm-scoring`)       | Merged into `develop`        |
| `bugfix/*`    | Non-urgent bug fixes                       | Merged into `develop`        |
| `hotfix/*`    | Emergency production fixes                 | Merged into `main` + `develop` |

Full policy in [`docs/branching.md`](docs/branching.md).

## Commit Convention

Follows **type(scope): subject** (Conventional Commits style):

```
feat(analytics): add K-Means clustering with silhouette scoring
fix(cleaning): handle NaN CustomerID rows before RFM aggregation
sec(access): enforce small-group suppression at cohort size < 5
docs(architecture): update layer diagram to match Unit 3 submission
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
<<<<<<< HEAD
=======

>>>>>>> 22fba0a9b92e95ce92f69c250e1d7eca90ceb648
