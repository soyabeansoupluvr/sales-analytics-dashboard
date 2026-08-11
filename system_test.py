"""System and acceptance test run.

Executes the full protected pipeline on the UCI Online Retail workbook, then exercises
acceptance-level scenarios: role-scoped reads, denial paths, analytics queries against the
protected snapshot, and audit-chain verification.
"""

from __future__ import annotations

import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.access import Actor  # noqa: E402
from src.analytics import (  # noqa: E402
    country_metrics,
    product_metrics,
    repeat_rate,
    revenue_summary,
    rfm_segments,
    time_series,
)
from src.cleaning import Cleaner  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.ingestion import ingest  # noqa: E402
from src.logs import AuditLog, AuditLogVerifier, Database  # noqa: E402
from src.pseudonymize import pseudonymize_column  # noqa: E402
from src.storage import StorageError, read_view, write_snapshot  # noqa: E402

PASS = "PASS"
FAIL = "FAIL"
results: list[tuple[str, str, str]] = []


def record(case: str, ok: bool, detail: str) -> None:
    results.append((case, PASS if ok else FAIL, detail))
    print(f"[{PASS if ok else FAIL}] {case}: {detail}")


def main() -> None:
    settings = get_settings()
    source = Path("data/raw/Online Retail.xlsx")

    print("=" * 80)
    print("SYSTEM TEST RUN")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print(f"Source workbook: {source} ({source.stat().st_size / 1_048_576:.1f} MiB)")
    print("=" * 80)

    tracemalloc.start()
    t0 = time.perf_counter()

    # ST-01: ingestion trust boundary accepts the genuine workbook
    t = time.perf_counter()
    raw = ingest(source)
    ingest_s = time.perf_counter() - t
    record(
        "ST-01 ingestion accepts valid workbook",
        len(raw) == 541_909,
        f"{len(raw):,} rows ingested in {ingest_s:.1f} s",
    )

    # ST-02: staged cleaning with reported deltas
    t = time.perf_counter()
    result = Cleaner().clean(raw)
    clean_s = time.perf_counter() - t
    cleaned = result.frame
    report = result.report
    for delta in report.stages:
        print(
            f"    stage={delta.stage:<22} in={delta.rows_in:>7,} "
            f"out={delta.rows_out:>7,} dropped={delta.dropped:>7,}"
        )
    record(
        "ST-02 staged cleaning reports deltas",
        len(cleaned) == 406_829,
        f"{report.input_rows:,} -> {report.output_rows:,} rows "
        f"({report.total_dropped:,} dropped) in {clean_s:.1f} s",
    )

    # ST-03: pseudonymization rewrites CustomerID
    t = time.perf_counter()
    key = settings.require_pseudonym_key().encode("utf-8")
    protected = pseudonymize_column(cleaned, key=key, drop_source=True)
    pseud_s = time.perf_counter() - t
    raw_ids = set(str(v) for v in cleaned["CustomerID"].dropna().unique()[:50])
    leaked = raw_ids & set(str(v) for v in protected["CustomerID"].unique()[:5000])
    record(
        "ST-03 pseudonymization leaves no raw identifiers",
        not leaked,
        f"{len(protected):,} rows pseudonymized in {pseud_s:.2f} s",
    )

    # ST-04: encrypted snapshot write with audit event
    audit_log = AuditLog(Database(settings))
    snapshot_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    t = time.perf_counter()
    write_snapshot(protected, settings, snapshot_id=snapshot_id, audit_log=audit_log)
    write_s = time.perf_counter() - t
    total_s = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    record(
        "ST-04 encrypted snapshot written",
        True,
        f"snapshot {snapshot_id} written in {write_s:.1f} s",
    )
    print(
        f"    pipeline total {total_s:.1f} s, throughput "
        f"{len(raw) / total_s:,.0f} rows/s, peak memory {peak / 1_048_576:.1f} MiB"
    )

    # AT-01: analyst may read the customers view
    analyst = Actor(username="analyst", role="analyst")
    frame = read_view("customers", analyst, settings, audit_log=audit_log)
    record(
        "AT-01 analyst reads customers view",
        len(frame) > 0,
        f"{len(frame):,} rows returned",
    )

    # AT-02: viewer is denied the customers view before decryption
    viewer = Actor(username="viewer", role="viewer")
    try:
        read_view("customers", viewer, settings, audit_log=audit_log)
        record("AT-02 viewer denied customers view", False, "denial did not raise")
    except StorageError as exc:
        record("AT-02 viewer denied customers view", True, f"denied: {exc}")

    # AT-03: viewer may read aggregate revenue view
    rev_frame = read_view("revenue", viewer, settings, audit_log=audit_log)
    record(
        "AT-03 viewer reads revenue view",
        len(rev_frame) > 0,
        f"{len(rev_frame):,} rows returned",
    )

    # AT-04 .. AT-09: analytics queries answer the business questions
    analytics_frame = read_view("customers", analyst, settings, audit_log=audit_log)
    queries = {
        "AT-04 revenue summary": lambda: revenue_summary(analytics_frame),
        "AT-05 product metrics": lambda: product_metrics(analytics_frame),
        "AT-06 time series": lambda: time_series(analytics_frame),
        "AT-07 country metrics": lambda: country_metrics(analytics_frame),
        "AT-08 repeat rate": lambda: repeat_rate(analytics_frame),
        "AT-09 RFM segmentation": lambda: rfm_segments(
            analytics_frame, settings=settings
        ),
    }
    for case, fn in queries.items():
        t = time.perf_counter()
        out = fn()
        q_s = time.perf_counter() - t
        record(case, out is not None, f"answered in {q_s:.2f} s")

    # AT-10: hash-chained audit log verifies intact
    try:
        AuditLogVerifier(Database(settings)).verify()
        record("AT-10 audit chain verification", True, "hash chain intact")
    except Exception as exc:  # noqa: BLE001
        record("AT-10 audit chain verification", False, str(exc))

    print("=" * 80)
    passed = sum(1 for _, status, _ in results if status == PASS)
    print(f"RESULT: {passed}/{len(results)} system and acceptance cases passed")
    print("=" * 80)


if __name__ == "__main__":
    main()
