# Input Schema Requirements

This document defines the file and column contract that the ingestion module
(`src/ingestion.py`) enforces on any dataset placed in `data/raw/`. Files that
violate the contract are rejected before any downstream module sees them.

The contract is the [UCI Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail)
schema. Any other retail transaction extract that conforms to the schema below
will ingest and clean identically; the module does not privilege the UCI file
by name.

## Accepted file formats

| Extension | Loader path                          | Notes                                             |
|-----------|--------------------------------------|---------------------------------------------------|
| `.csv`    | `pandas.read_csv` with `dtype=str`   | UTF-8, UTF-8 with BOM, and Latin-1 are attempted in order. |
| `.xlsx`   | `pandas.read_excel` with `dtype=str` | First sheet is used. Legacy `.xls` is rejected.   |

Files with any other extension are rejected with a clear error. The application
resolves the source file by scanning `data/raw/` for a supported extension —
the file may be named anything.

## File-level limits

The ingestion module refuses files that exceed these limits so a malformed or
hostile input cannot exhaust memory or CPU.

| Limit                            | Value      | Rationale                                |
|----------------------------------|------------|------------------------------------------|
| Maximum file size on disk        | 64 MiB     | Well above the ~44 MB UCI workbook.      |
| Maximum rows                     | 2,000,000  | Well above the ~541,909 UCI transactions.|
| Maximum members in an XLSX zip   | 10,000     | Guards against zip-bomb style archives.  |
| Maximum uncompressed XLSX bytes  | 512 MiB    | Guards against zip-bomb style archives.  |
| Maximum XLSX compression ratio   | 250:1      | Guards against zip-bomb style archives.  |

An XLSX file must additionally contain `[Content_Types].xml`, `_rels/.rels`,
and `xl/workbook.xml` at the expected paths. A missing member is treated as a
malformed workbook.

## Required columns

The header row must contain exactly the following eight column names, in any
order, with no duplicates and no additional columns:

| Column        | Type on disk | Semantic type              | Notes                                                       |
|---------------|--------------|----------------------------|-------------------------------------------------------------|
| `InvoiceNo`   | string       | Invoice identifier         | Six digits for normal sales; `C` + digits for cancellations; `A` + digits for adjustments. |
| `StockCode`   | string       | Product identifier         | Alphanumeric. A short set of well-known non-product codes (POST, DOT, M, BANK CHARGES, AMAZONFEE, CRUK, PADS, S, D) is tagged and excluded from product analytics. |
| `Description` | string       | Product description        | May be empty for administrative rows.                       |
| `Quantity`    | integer      | Units transacted           | May be negative for cancellations, returns, and adjustments.|
| `InvoiceDate` | datetime     | Transaction timestamp      | Preferred format `MM/DD/YYYY HH:MM`. Other layouts are retried with pandas' mixed-format parser. |
| `UnitPrice`   | decimal      | Unit price in GBP          | Non-negative for sales. Negative or zero rows are flagged.  |
| `CustomerID`  | string       | Customer identifier        | Digits only. An Excel-produced trailing `.0` is accepted and stripped. Empty values remove the row from customer analytics; the row is retained for revenue totals only if the invoice is not a cancellation. |
| `Country`     | string       | Country of the customer    | Free text; not enumerated.                                  |

Column names are matched case-sensitively and whitespace-sensitively. A missing
column, an unexpected column, or a duplicated column raises `SchemaMismatchError`
before any row is read.

## Value-level rules

Every rule below either raises a fail-fast structural error or records a
`ValidationIssue` in the returned `DataQualityReport`. Structural errors block
ingestion entirely; value-level issues are surfaced to the operator and, where
possible, dropped from the accepted frame rather than crashing the pipeline.

### Formula-injection safety

Any cell whose text begins with `=`, `+`, `-`, `@`, tab, carriage return, line
feed, or the full-width equivalents `＝`, `＋`, `－`, `＠` is treated as a
formula-injection attempt and rejected at the row level. Pure signed numeric
literals such as `-1` and `+2.5` are allowed.

### Type coercion

| Column        | Coercion                                       | On failure                                                       |
|---------------|------------------------------------------------|------------------------------------------------------------------|
| `Quantity`    | `pandas.to_numeric` then cast to `Int64`.      | Non-numeric values become NA and the row is dropped.             |
| `UnitPrice`   | `pandas.to_numeric` then cast to `Float64`.    | Non-numeric values become NA and the row is dropped.             |
| `InvoiceDate` | `pandas.to_datetime` with the UCI format, then a mixed-format retry. | Unparseable dates become NaT and the row is dropped. |
| `CustomerID`  | Digits-only pattern `^[0-9]+(?:\.0+)?$` with the trailing `.0` stripped. | Non-conforming values become NA and the row is excluded from customer-level analytics. |

### Cancellations and returns

* An invoice matching `^C\d+$` is tagged `IsCancellation=True`.
* An invoice matching `^A\d+$` is tagged `IsAdjustment=True`.
* Cancellations with a corresponding original sale are paired to that sale for
  net-revenue calculation. Unpaired cancellations are retained but reported.

### Derived columns

The cleaning stage adds these columns to the accepted frame. They are not part
of the input contract but are documented here so downstream consumers know what
to expect from `src/cleaning.py`.

| Column           | Type    | Meaning                                                                 |
|------------------|---------|-------------------------------------------------------------------------|
| `IsCancellation` | boolean | True for `C`-prefixed invoices.                                         |
| `IsAdjustment`   | boolean | True for `A`-prefixed invoices.                                         |
| `IsNonProduct`   | boolean | True for the enumerated administrative stock codes above.               |
| `Revenue`        | float   | `Quantity * UnitPrice`.                                                 |

## Failure modes and error surfaces

| Situation                                  | Raised as                                              | Where                       |
|--------------------------------------------|--------------------------------------------------------|-----------------------------|
| No file in `data/raw/` with a supported extension | `FileNotFoundError`                              | `src/app.py::_resolve_source` |
| Unsupported extension                      | `UnsupportedFormatError`                               | `src/ingestion.py`           |
| File exceeds any limit above               | `FileTooLargeError` or `MalformedArchiveError`         | `src/ingestion.py`           |
| Header does not match the required schema  | `SchemaMismatchError`                                  | `src/ingestion.py`           |
| Individual bad rows                        | Dropped and reported in `DataQualityReport.issues`     | `src/ingestion.py`           |
| Structural cleaning failure                | `CleaningError`                                        | `src/cleaning.py`            |
| Individual bad values during cleaning      | Recorded in `CleaningReport.stage_deltas`              | `src/cleaning.py`            |

## Placing a file

Drop a `.csv` or `.xlsx` conforming to the contract above into `data/raw/`.
The classroom target file is `Online Retail.xlsx` from the UCI archive linked
above. Any conforming extract may be substituted.

```
data/
  raw/
    Online Retail.xlsx        # or your own conforming extract
  processed/
    dashboard.db              # created on first run, holds SQLite protected store
```

Both `data/raw/` and `data/processed/` are gitignored. Only the `.gitkeep`
markers are tracked.

## Verification

The ingestion and cleaning contracts above are covered by:

* `tests/test_ingestion.py` — schema enforcement, file-level limits, formula
  injection, and encoding fallback.
* `tests/test_cleaning.py` — cancellation and adjustment tagging, non-product
  filtering, type coercion, and revenue derivation.

Running `pytest tests/test_ingestion.py tests/test_cleaning.py` exercises every
rule documented above.
