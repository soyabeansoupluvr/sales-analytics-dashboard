"""M10 - Cleaning / ETL.

Normalizes the string-based data produced by M9 Ingestion, removes rows
without a usable customer identifier, identifies cancellations, adjustments,
and returns, matches eligible return rows to earlier original transactions,
calculates Revenue, and marks non-product rows. The cleaned data is passed
to M6 Pseudonymization.

Design notes:

* Each transformation stage has a single responsibility: TypeCoercer,
  NullFilter, CancellationTagger, RevenueDeriver, and NonProductFilter.
* The default stages do not modify the caller's DataFrame and do not access
  the filesystem or network.
* Report data is stored in frozen dataclasses. StageDelta.details uses an
  immutable mapping. CleaningResult prevents its fields from being reassigned,
  but the contained pandas DataFrame remains mutable.
* Cleaner coordinates the stages. Stages are supplied through the constructor,
  allowing them to be replaced with test doubles when needed.
* Structural problems, such as missing or duplicate required columns, raise
  CleaningError. Data problems, such as malformed values, missing customer
  identifiers, invalid dates, and unmatched returns, are recorded in the
  CleaningReport instead of raising exceptions.
"""
from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Mapping, Protocol, Any, cast

if TYPE_CHECKING:
    import pandas as pd

from src.logs import get_logger


# --------------------------------------------------------------------------- #
# Contracts
# --------------------------------------------------------------------------- #

_EXPECTED_COLUMNS: Final[tuple[str, ...]] = (
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
)

_NON_PRODUCT_CODES: Final[frozenset[str]] = frozenset(
    {
        "POST",          # Postage
        "DOT",           # Dotcom postage
        "M",             # Manual adjustment
        "BANK CHARGES",  # Bank charges
        "AMAZONFEE",     # Amazon fee
        "CRUK",          # Charity item
        "PADS",          # Packaging materials
        "S",             # Sample
        "D",             # Discount
    }
)

# UCI cancellation and adjustment invoices use a letter followed by digits.
# Matching the full invoice number avoids tagging unrelated values.
_CANCELLATION_RE: Final[re.Pattern[str]] = re.compile(r"^C\d+$")
_ADJUSTMENT_RE: Final[re.Pattern[str]] = re.compile(r"^A\d+$")

_DATE_DAYFIRST: Final[bool] = False

# Most UCI dates use this format. Other layouts are retried with
# pandas' mixed-format parser.
_UCI_DATE_FORMAT: Final[str] = "%m/%d/%Y %H:%M"

# Customer identifiers are categorical values. Digits are accepted directly,
# and an Excel-style trailing decimal portion containing only zeroes is
# removed. Scientific notation, signs, and fractional identifiers are refused.
_CUSTOMER_ID_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<digits>[0-9]+)(?:\.0+)?$"
)

_LOG = get_logger(__name__)


class CleaningError(ValueError):
    """Raised for structural violations that block cleaning entirely."""


@dataclass(frozen=True, slots=True)
class StageDelta:
    """Row-count effect and counters for one cleaning stage."""

    stage: str
    rows_in: int
    rows_out: int
    details: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Copy first so later mutations to the caller's dictionary cannot
        # change an already-created report object.
        immutable_details = MappingProxyType(dict(self.details))
        object.__setattr__(self, "details", immutable_details)

    @property
    def dropped(self) -> int:
        return max(0, self.rows_in - self.rows_out)


@dataclass(frozen=True, slots=True)
class CleaningReport:
    """Aggregate diagnostics from a cleaning run."""

    input_rows: int
    output_rows: int
    stages: tuple[StageDelta, ...] = field(default_factory=tuple)

    @property
    def total_dropped(self) -> int:
        return max(0, self.input_rows - self.output_rows)

    def get(self, stage: str) -> StageDelta | None:
        for delta in self.stages:
            if delta.stage == stage:
                return delta
        return None


@dataclass(frozen=True, slots=True)
class CleaningResult:
    """Cleaning output and report.

    The dataclass fields cannot be reassigned. The contained pandas DataFrame
    remains mutable because pandas does not provide a frozen DataFrame type.
    """

    frame: pd.DataFrame
    report: CleaningReport


class _Stage(Protocol):
    """Structural interface for a cleaning stage."""

    name: str

    def apply(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, StageDelta]: ...


# --------------------------------------------------------------------------- #
# Stage 1: Type coercion
# --------------------------------------------------------------------------- #


class TypeCoercer:
    """Coerce M9 string values into stable analytics types.

    Malformed individual values become missing values and are counted. The
    stage does not raise merely because one cell cannot be converted.
    """

    name = "type_coercion"

    def apply(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, StageDelta]:
        rows_in = int(len(frame))
        out = frame.copy()

        quantity, quantity_missing, quantity_invalid = self._coerce_integer(
            out["Quantity"]
        )
        out["Quantity"] = quantity

        unit_price, price_missing, price_invalid = self._coerce_float(
            out["UnitPrice"]
        )
        out["UnitPrice"] = unit_price

        customer_id, customer_missing, customer_invalid = (
            self._coerce_customer_id(out["CustomerID"])
        )
        out["CustomerID"] = customer_id

        invoice_date, date_missing, date_invalid = self._coerce_date(
            out["InvoiceDate"]
        )
        out["InvoiceDate"] = invoice_date

        for column in ("InvoiceNo", "StockCode", "Description", "Country"):
            out[column] = self._normalize_text(out[column])

        details = {
            "quantity_missing": quantity_missing,
            "quantity_uncoerced": quantity_invalid,
            "unit_price_missing": price_missing,
            "unit_price_uncoerced": price_invalid,
            "customer_id_missing": customer_missing,
            "customer_id_uncoerced": customer_invalid,
            "invoice_date_missing": date_missing,
            "invoice_date_uncoerced": date_invalid,
        }
        return out, StageDelta(self.name, rows_in, int(len(out)), details)

    @staticmethod
    def _normalize_text(series: pd.Series) -> pd.Series:
        import pandas as pd

        text = series.astype("string").str.strip()
        return text.mask(text == "", pd.NA)

    @classmethod
    def _coerce_integer(
        cls, series: pd.Series
    ) -> tuple[pd.Series, int, int]:
        import pandas as pd

        text = series.astype("string").str.strip()
        missing = series.isna() | text.isna() | (text == "")

        parsed = pd.to_numeric(text.mask(missing), errors="coerce")
        finite = parsed.map(cls._is_finite_number)
        whole = parsed.notna() & finite & (parsed == parsed.round())

        int64_min = -(2**63)
        int64_max = (2**63) - 1
        in_range = parsed.ge(int64_min) & parsed.le(int64_max)
        valid = whole & in_range

        invalid = (~missing) & (~valid)
        coerced = parsed.where(valid).astype("Int64")
        return coerced, int(missing.sum()), int(invalid.sum())

    @classmethod
    def _coerce_float(
        cls, series: pd.Series
    ) -> tuple[pd.Series, int, int]:
        import pandas as pd

        text = series.astype("string").str.strip()
        missing = series.isna() | text.isna() | (text == "")

        parsed = pd.to_numeric(text.mask(missing), errors="coerce")
        finite = parsed.map(cls._is_finite_number)
        valid = parsed.notna() & finite
        invalid = (~missing) & (~valid)

        coerced = parsed.where(valid).astype("Float64")
        return coerced, int(missing.sum()), int(invalid.sum())

    @staticmethod
    def _coerce_customer_id(
        series: pd.Series,
    ) -> tuple[pd.Series, int, int]:
        import pandas as pd

        text = series.astype("string").str.strip()
        missing = series.isna() | text.isna() | (text == "")

        extracted = text.str.extract(_CUSTOMER_ID_RE, expand=True)["digits"]
        valid = (~missing) & extracted.notna()
        invalid = (~missing) & (~valid)

        coerced = extracted.where(valid).astype("string")
        return coerced, int(missing.sum()), int(invalid.sum())

    @staticmethod
    def _coerce_date(series: pd.Series) -> tuple[pd.Series, int, int]:
        import pandas as pd

        text = series.astype("string").str.strip()
        missing = series.isna() | text.isna() | (text == "")
        candidates = text.mask(missing)

        # Parse the standard UCI format first, then retry unmatched values.
        parsed = pd.to_datetime(
            candidates,
            format=_UCI_DATE_FORMAT,
            errors="coerce",
        )
        fallback_needed = (~missing) & parsed.isna()
        if bool(fallback_needed.any()):
            # Retry nonstandard dates without pandas' format-inference warning.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                fallback = pd.to_datetime(
                    candidates.where(fallback_needed),
                    errors="coerce",
                    dayfirst=_DATE_DAYFIRST,
                    format="mixed",
                )
            parsed = parsed.where(~fallback_needed, fallback)

        invalid = (~missing) & parsed.isna()
        return parsed, int(missing.sum()), int(invalid.sum())

    @staticmethod
    def _is_finite_number(value: object) -> bool:
        import pandas as pd

        if value is None or value is pd.NA:
            return False
        try:
            return math.isfinite(float(cast(Any, value)))
        except (TypeError, ValueError, OverflowError):
            return False


# --------------------------------------------------------------------------- #
# Stage 2: Drop rows without a usable CustomerID
# --------------------------------------------------------------------------- #


class NullFilter:
    """Drop rows that cannot be associated with a customer key."""

    name = "null_customer_drop"

    def apply(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, StageDelta]:
        rows_in = int(len(frame))
        mask = frame["CustomerID"].notna()
        out = frame.loc[mask].copy()
        rows_out = int(len(out))

        return out, StageDelta(
            self.name,
            rows_in,
            rows_out,
            {"customer_id_rows_dropped": rows_in - rows_out},
        )


# --------------------------------------------------------------------------- #
# Stage 3: Cancellation, adjustment, and return tagging and pairing
# --------------------------------------------------------------------------- #


class CancellationTagger:
    """Tags cancellations, adjustments, and returns, then pairs eligible returns.

    Eligible returns are matched to the most recent earlier transaction with the
    same customer, stock code, quantity, and unit price. Each original transaction
    can be matched only once. Returns and original transactions that cannot
    participate in pairing are counted in the stage report.
    """

    name = "cancellation_tagging"

    def apply(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, StageDelta]:
        import pandas as pd

        rows_in = int(len(frame))

        # Reset the index so row lookups remain unambiguous.
        out = frame.reset_index(drop=True).copy()

        invoice = out["InvoiceNo"].astype("string").fillna("")
        normalized_invoice = invoice.str.strip().str.upper()

        out["IsCancellation"] = normalized_invoice.str.match(
            _CANCELLATION_RE, na=False
        )
        out["IsAdjustment"] = normalized_invoice.str.match(
            _ADJUSTMENT_RE, na=False
        )

        negative_quantity = out["Quantity"].lt(0).fillna(False)
        out["IsReturn"] = negative_quantity & (~out["IsAdjustment"])

        paired_invoice = pd.Series(pd.NA, index=out.index, dtype="string")
        unmatched_originals: dict[tuple[str, str, int, float], list[str]] = {}

        # Preserve source order when timestamps are equal.
        sorted_indexes = out.sort_values("InvoiceDate", kind="stable").index

        ineligible_returns = 0
        ineligible_originals = 0
        for row_index in sorted_indexes:
            is_cancellation = bool(out.at[row_index, "IsCancellation"])
            is_adjustment = bool(out.at[row_index, "IsAdjustment"])
            is_return = bool(out.at[row_index, "IsReturn"])
            date_missing = bool(pd.isna(out.at[row_index, "InvoiceDate"]))

            # Transactions without a valid date cannot be paired chronologically.
            # Count returns and potential originals separately.
            if date_missing:
                if is_return:
                    ineligible_returns += 1
                elif not is_cancellation and not is_adjustment:
                    ineligible_originals += 1
                continue

            pair_key = self._build_pair_key(out, row_index)

            if is_return:
                if pair_key is None:
                    ineligible_returns += 1
                    continue

                candidates = unmatched_originals.get(pair_key)
                if candidates:
                    paired_invoice.at[row_index] = candidates.pop()
                    if not candidates:
                        del unmatched_originals[pair_key]
                continue

            quantity = out.at[row_index, "Quantity"]
            is_positive_quantity = (
                not pd.isna(quantity) and int(cast(Any, quantity)) > 0
            )

            if (
                pair_key is not None
                and is_positive_quantity
                and not is_cancellation
                and not is_adjustment
            ):
                original_invoice = out.at[row_index, "InvoiceNo"]
                if isinstance(original_invoice, str) and original_invoice:
                    candidates = unmatched_originals.setdefault(pair_key, [])
                    candidates.append(original_invoice)
            elif (
                pair_key is None
                and not is_cancellation
                and not is_adjustment
                and is_positive_quantity
            ):
                # Count positive transactions that lack the data required for pairing.
                ineligible_originals += 1

        out["PairedInvoiceNo"] = paired_invoice

        cancellation_mask = out["IsCancellation"]
        return_mask = out["IsReturn"]
        paired_mask = out["PairedInvoiceNo"].notna()

        cancellation_count = int(cancellation_mask.sum())
        adjustment_count = int(out["IsAdjustment"].sum())
        return_count = int(return_mask.sum())
        paired_cancellations = int((cancellation_mask & paired_mask).sum())
        paired_returns = int((return_mask & paired_mask).sum())
        positive_cancellations = int(
            (cancellation_mask & (~negative_quantity)).sum()
        )

        details = {
            "cancellations": cancellation_count,
            "adjustments": adjustment_count,
            "returns": return_count,
            "paired_cancellations": paired_cancellations,
            "unpaired_cancellations": cancellation_count - paired_cancellations,
            "paired_returns": paired_returns,
            "unpaired_returns": return_count - paired_returns,
            "pairing_ineligible_returns": ineligible_returns,
            "pairing_ineligible_originals": ineligible_originals,
            "cancellations_without_negative_quantity": positive_cancellations,
        }
        return out, StageDelta(self.name, rows_in, int(len(out)), details)

    @staticmethod
    def _build_pair_key(
        frame: pd.DataFrame, row_index: int
    ) -> tuple[str, str, int, float] | None:
        import pandas as pd

        customer_id = frame.at[row_index, "CustomerID"]
        stock_code = frame.at[row_index, "StockCode"]
        quantity = frame.at[row_index, "Quantity"]
        unit_price = frame.at[row_index, "UnitPrice"]
        invoice_no = frame.at[row_index, "InvoiceNo"]

        if (
            pd.isna(customer_id)
            or pd.isna(stock_code)
            or pd.isna(quantity)
            or pd.isna(unit_price)
            or pd.isna(invoice_no)
        ):
            return None

        try:
            quantity_value = int(cast(Any, quantity))
            unit_price_value = float(cast(Any, unit_price))
        except (TypeError, ValueError, OverflowError):
            return None

        if quantity_value == 0 or not math.isfinite(unit_price_value):
            return None

        customer_text = str(customer_id).strip()
        stock_text = str(stock_code).strip()
        invoice_text = str(invoice_no).strip()
        if not customer_text or not stock_text or not invoice_text:
            return None

        return (
            customer_text,
            stock_text,
            abs(quantity_value),
            round(unit_price_value, 4),
        )


# --------------------------------------------------------------------------- #
# Stage 4: Revenue derivation
# --------------------------------------------------------------------------- #


class RevenueDeriver:
    """Calculates Revenue as Quantity times UnitPrice using nullable floats."""

    name = "revenue_derivation"

    def apply(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, StageDelta]:
        rows_in = int(len(frame))
        out = frame.copy()

        quantity = out["Quantity"].astype("Float64")
        unit_price = out["UnitPrice"].astype("Float64")
        out["Revenue"] = quantity * unit_price

        negative_revenue = out["Revenue"].lt(0).fillna(False)
        zero_revenue = out["Revenue"].eq(0).fillna(False)

        return out, StageDelta(
            self.name,
            rows_in,
            int(len(out)),
            {
                "negative_revenue_rows": int(negative_revenue.sum()),
                "zero_revenue_rows": int(zero_revenue.sum()),
            },
        )


# --------------------------------------------------------------------------- #
# Stage 5: Non-product tagging
# --------------------------------------------------------------------------- #


class NonProductFilter:
    """Adds an IsNonProduct flag for use by downstream filters.

    Rows are marked rather than removed so RFM calculations can still include
    postage, fees, and manual adjustments.
    """

    name = "non_product_tagging"

    def __init__(self, codes: frozenset[str] = _NON_PRODUCT_CODES) -> None:
        self._codes = codes

    def apply(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, StageDelta]:
        rows_in = int(len(frame))
        out = frame.copy()
        codes = out["StockCode"].astype("string").fillna("")
        out["IsNonProduct"] = codes.isin(self._codes)
        non_product_count = int(out["IsNonProduct"].sum())

        return out, StageDelta(
            self.name,
            rows_in,
            int(len(out)),
            {"non_product_rows": non_product_count},
        )


# --------------------------------------------------------------------------- #
# Composition root
# --------------------------------------------------------------------------- #


class Cleaner:
    """Compose cleaning stages into an end-to-end transform."""

    _stages: tuple[_Stage, ...]

    def __init__(self, stages: tuple[_Stage, ...] | None = None) -> None:
        if stages is None:
            self._stages = self._default_stages()
        else:
            # An empty tuple intentionally means that no stages should run.
            self._stages = tuple(stages)

    @staticmethod
    def _default_stages() -> tuple[_Stage, ...]:
        return (
            TypeCoercer(),
            NullFilter(),
            CancellationTagger(),
            RevenueDeriver(),
            NonProductFilter(),
        )

    def clean(self, frame: pd.DataFrame) -> CleaningResult:
        self._require_columns(frame)

        rows_in = int(len(frame))
        current = frame.copy()
        deltas: list[StageDelta] = []

        for stage in self._stages:
            current, delta = stage.apply(current)
            deltas.append(delta)
            _LOG.info(
                "clean stage=%s in=%d out=%d details=%s",
                delta.stage,
                delta.rows_in,
                delta.rows_out,
                dict(delta.details),
            )

        report = CleaningReport(
            input_rows=rows_in,
            output_rows=int(len(current)),
            stages=tuple(deltas),
        )
        _LOG.info(
            "clean done input=%d output=%d dropped=%d",
            report.input_rows,
            report.output_rows,
            report.total_dropped,
        )
        return CleaningResult(frame=current, report=report)

    @staticmethod
    def _require_columns(frame: pd.DataFrame) -> None:
        import pandas as pd

        if not isinstance(frame, pd.DataFrame):
            raise CleaningError(
                "Cleaning requires a pandas DataFrame as its input"
            )

        actual = tuple(frame.columns)
        duplicate_names = tuple(
            dict.fromkeys(
                str(column)
                for column in frame.columns[frame.columns.duplicated()].tolist()
            )
        )
        if duplicate_names:
            raise CleaningError(
                "Cleaning received duplicate column names: "
                f"{duplicate_names}. Got: {actual}"
            )

        missing = tuple(column for column in _EXPECTED_COLUMNS if column not in actual)
        if missing:
            raise CleaningError(
                "Cleaning received a frame missing required columns: "
                f"{missing}. Got: {actual}"
            )


# --------------------------------------------------------------------------- #
# Module-level convenience API
# --------------------------------------------------------------------------- #


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    """Return only the cleaned DataFrame."""

    return Cleaner().clean(frame).frame


__all__ = [
    "StageDelta",
    "CleaningReport",
    "CleaningResult",
    "CleaningError",
    "TypeCoercer",
    "NullFilter",
    "CancellationTagger",
    "RevenueDeriver",
    "NonProductFilter",
    "Cleaner",
    "clean",
]
