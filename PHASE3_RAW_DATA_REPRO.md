# Phase 3 Raw-Data Reproduction With Standalone Code

## Purpose

This document contains the standalone processing code needed to go from the raw challenge tables to the Phase 3 Level 1 predictions.

Instead, it embeds the prediction-affecting logic directly:

- raw input validation
- transaction normalization
- eclass entity construction
- pre-boundary filtering
- warm-start candidate generation
- cohort assignment
- eligibility gating
- scoring
- per-buyer ranking
- ranked parquet writing
- submission CSV writing

This is the path that reproduced the successful submission with:

- Total Score: `€771,991.01`
- Savings: `€935,411.01`
- Fees: `€163,420.00`
- Hits: `11,739`

Reference date used in the successful run:

- `2025-12-31`

## Inputs

The raw input directory must contain:

- `plis_training.csv`
- `customer_test.csv`
- `features_per_sku.csv`
- `nace_codes.csv`

## Outputs

The standalone script below writes:

- normalized transactions parquet
- eclass entity parquet
- ranked evidence parquet
- portal-ready submission CSV

For the successful `2025-12-31` reproduction, the expected Phase 3 counts were:

- ranked rows: `16,342`
- buyers with predictions: `47`
- submission rows: `16,342`

## Usage

Save the following code as a Python script, for example `phase3_level1_standalone.py`, then run:

```bash
uv run python phase3_level1_standalone.py \
  --raw-root <raw_data_root> \
  --output-root <artifacts_root> \
  --reference-date 2025-12-31
```

## Standalone Code

```python
from __future__ import annotations

import argparse
import calendar
import math
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_ETA = 2
DEFAULT_TAU = 100.0
DEFAULT_LOOKBACK_MONTHS = 18
DEFAULT_SAVINGS_RATE = 0.10
SPARSE_HISTORY_ETA_MULTIPLIER = 3
SPARSE_HISTORY_TAU_MULTIPLIER = 2.0

COLD_START_COHORT = "cold_start"
SPARSE_HISTORY_COHORT = "sparse_history"
RICH_HISTORY_COHORT = "rich_history"
SPARSE_HISTORY_MAX_MONTHS = 3
ECLASS_ID_PREFIX = "eclass::"

PLIS_REQUIRED_COLUMNS = [
    "orderdate",
    "legal_entity_id",
    "set_id",
    "sku",
    "eclass",
    "manufacturer",
    "quantityvalue",
    "vk_per_item",
    "estimated_number_employees",
    "nace_code",
    "secondary_nace_code",
]
CUSTOMER_TEST_REQUIRED_COLUMNS = [
    "legal_entity_id",
    "estimated_number_employees",
    "nace_code",
    "secondary_nace_code",
    "task",
]
FEATURES_REQUIRED_COLUMNS = ["safe_synonym", "sku", "key", "fvalue", "fvalue_set"]
NACE_REQUIRED_COLUMNS = [
    "nace_code",
    "n_nace_description",
    "toplevel_section",
    "toplevel_section_description",
    "nace_2digits",
    "nace_2digits_description",
    "nace_3digits",
    "nace_3digits_description",
]


@dataclass(frozen=True, slots=True)
class CandidateRow:
    reference_date: date
    legal_entity_id: str
    eclass_id: str
    eclass: str
    history_cohort: str
    n_orders_in_lookback: int
    n_active_months_in_lookback: int
    lookback_spend: float
    total_orders: int
    total_spend: float
    avg_spend_per_order: float
    first_order_date: date
    last_order_date: date
    history_span_days: int
    score: float
    rank: int
    passes_eta_gate: bool
    passes_tau_gate: bool
    is_eligible: bool

    def to_row(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RankedRow:
    reference_date: date
    legal_entity_id: str
    eclass_id: str
    eclass: str
    history_cohort: str
    score: float
    rank: int
    n_orders_in_lookback: int
    n_active_months_in_lookback: int
    days_since_last_order: int
    recurrence_weight: float
    avg_spend_per_order: float
    lookback_spend: float
    passes_sparse_gate: bool

    def to_row(self) -> dict:
        return asdict(self)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone Phase 3 Level 1 warm-start reproduction from raw CSVs."
    )
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reference-date", type=date.fromisoformat, default=date(2025, 12, 31))
    parser.add_argument("--lookback-months", type=int, default=DEFAULT_LOOKBACK_MONTHS)
    parser.add_argument("--eta", type=int, default=DEFAULT_ETA)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--savings-rate", type=float, default=DEFAULT_SAVINGS_RATE)
    return parser


def clean_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    collapsed = " ".join(str(value).split())
    return collapsed or None


def normalize_eclass(value: object) -> str | None:
    cleaned = clean_text(value)
    if cleaned is None:
        return None
    compact = cleaned.replace(" ", "")
    return compact if compact.isdigit() else None


def history_cohort(active_months_before_cutoff: int) -> str:
    if active_months_before_cutoff <= 0:
        return COLD_START_COHORT
    if active_months_before_cutoff <= SPARSE_HISTORY_MAX_MONTHS:
        return SPARSE_HISTORY_COHORT
    return RICH_HISTORY_COHORT


def make_eclass_id(eclass: str) -> str:
    return f"{ECLASS_ID_PREFIX}{eclass}"


def subtract_months(d: date, months: int) -> date:
    year = d.year
    month = d.month - months
    while month <= 0:
        month += 12
        year -= 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def validate_columns(path: Path, required_columns: list[str]) -> None:
    frame = pd.read_csv(path, sep="\t", nrows=0, encoding="utf-8-sig")
    actual = list(frame.columns)
    missing = [col for col in required_columns if col not in actual]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")


def validate_raw_inputs(raw_root: Path) -> None:
    validate_columns(raw_root / "plis_training.csv", PLIS_REQUIRED_COLUMNS)
    validate_columns(raw_root / "customer_test.csv", CUSTOMER_TEST_REQUIRED_COLUMNS)
    validate_columns(raw_root / "features_per_sku.csv", FEATURES_REQUIRED_COLUMNS)
    validate_columns(raw_root / "nace_codes.csv", NACE_REQUIRED_COLUMNS)


def load_plis_training(raw_root: Path) -> pd.DataFrame:
    path = raw_root / "plis_training.csv"
    frame = pd.read_csv(
        path,
        sep="\t",
        encoding="utf-8-sig",
        dtype={
            "legal_entity_id": "string",
            "set_id": "string",
            "sku": "string",
            "eclass": "string",
            "manufacturer": "string",
            "nace_code": "string",
            "secondary_nace_code": "string",
        },
        keep_default_na=False,
    )
    frame["order_date"] = pd.to_datetime(frame["orderdate"], errors="raise").dt.date
    frame["eclass_raw"] = frame["eclass"].map(clean_text)
    frame["eclass"] = frame["eclass"].map(normalize_eclass)
    frame["has_canonical_eclass"] = frame["eclass"].notna()
    frame["quantity_value"] = pd.to_numeric(frame["quantityvalue"], errors="coerce")
    frame["unit_price"] = pd.to_numeric(frame["vk_per_item"], errors="coerce")
    frame["legal_entity_id"] = frame["legal_entity_id"].astype("string")
    return frame[
        [
            "order_date",
            "legal_entity_id",
            "set_id",
            "sku",
            "eclass_raw",
            "eclass",
            "has_canonical_eclass",
            "quantity_value",
            "unit_price",
        ]
    ].copy()


def load_customer_test_buyers(raw_root: Path) -> list[str]:
    path = raw_root / "customer_test.csv"
    frame = pd.read_csv(
        path,
        sep="\t",
        encoding="utf-8-sig",
        usecols=["legal_entity_id"],
        dtype={"legal_entity_id": "string"},
        keep_default_na=False,
    )
    buyers = [
        str(value).strip()
        for value in frame["legal_entity_id"].tolist()
        if str(value).strip()
    ]
    return sorted(set(buyers))


def build_eclass_entities(transactions: pd.DataFrame) -> pd.DataFrame:
    entity_frame = (
        transactions.loc[transactions["eclass"].notna(), ["eclass"]]
        .value_counts()
        .reset_index(name="transaction_count")
        .sort_values("eclass", kind="mergesort")
        .reset_index(drop=True)
    )
    entity_frame["eclass_id"] = entity_frame["eclass"].map(make_eclass_id)
    return entity_frame[["eclass_id", "eclass", "transaction_count"]]


def load_pre_boundary_transactions(
    transactions: pd.DataFrame,
    reference_date: date,
) -> pd.DataFrame:
    df = transactions.loc[transactions["has_canonical_eclass"] == True].copy()  # noqa: E712
    df = df.loc[df["order_date"] <= reference_date].copy()
    df["quantity_value"] = pd.to_numeric(df["quantity_value"], errors="coerce").fillna(0.0)
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce").fillna(0.0)
    df["spend"] = df["quantity_value"] * df["unit_price"]
    df = df.loc[df["spend"] > 0].copy()
    df["year_month"] = pd.to_datetime(df["order_date"]).dt.to_period("M")
    return df


def build_warm_start_candidates(
    *,
    transactions: pd.DataFrame,
    eclass_id_map: dict[str, str],
    reference_date: date,
    target_buyers: list[str],
    lookback_months: int,
    eta: int,
    tau: float,
    savings_rate: float,
) -> list[CandidateRow]:
    lookback_start = subtract_months(reference_date, lookback_months)
    candidates: list[CandidateRow] = []
    all_buyers = set(target_buyers)

    for buyer_id in sorted(all_buyers):
        buyer_txns = transactions.loc[transactions["legal_entity_id"] == buyer_id].copy()
        full_history_months = int(buyer_txns["year_month"].nunique()) if not buyer_txns.empty else 0
        cohort = history_cohort(full_history_months)
        if buyer_txns.empty:
            continue

        by_eclass = buyer_txns.groupby("eclass")
        raw_candidates: list[dict] = []

        for eclass, group in by_eclass:
            eclass_str = str(eclass)
            eclass_id = eclass_id_map.get(eclass_str, make_eclass_id(eclass_str))
            total_orders = int(len(group))
            total_spend = float(group["spend"].sum())
            avg_spend_per_order = total_spend / total_orders if total_orders else 0.0
            first_order_date = min(group["order_date"])
            last_order_date = max(group["order_date"])
            history_span_days = (last_order_date - first_order_date).days

            lookback_group = group.loc[group["order_date"] >= lookback_start]
            n_orders_in_lookback = int(len(lookback_group))
            n_active_months_in_lookback = int(lookback_group["year_month"].nunique())
            lookback_spend = float(lookback_group["spend"].sum())

            passes_eta_gate = n_orders_in_lookback >= eta
            passes_tau_gate = lookback_spend >= tau
            is_eligible = passes_eta_gate and passes_tau_gate

            days_since_last = (reference_date - last_order_date).days
            recurrence_weight = n_orders_in_lookback * math.exp(-(1.0 / 365.0) * days_since_last)
            score = recurrence_weight * avg_spend_per_order * savings_rate

            raw_candidates.append(
                {
                    "eclass": eclass_str,
                    "eclass_id": eclass_id,
                    "n_orders_in_lookback": n_orders_in_lookback,
                    "n_active_months_in_lookback": n_active_months_in_lookback,
                    "lookback_spend": lookback_spend,
                    "total_orders": total_orders,
                    "total_spend": total_spend,
                    "avg_spend_per_order": avg_spend_per_order,
                    "first_order_date": first_order_date,
                    "last_order_date": last_order_date,
                    "history_span_days": history_span_days,
                    "score": score,
                    "passes_eta_gate": passes_eta_gate,
                    "passes_tau_gate": passes_tau_gate,
                    "is_eligible": is_eligible,
                }
            )

        eligible_sorted = sorted(
            [row for row in raw_candidates if row["is_eligible"]],
            key=lambda row: (-row["score"], row["eclass"]),
        )
        rank_map = {row["eclass"]: index + 1 for index, row in enumerate(eligible_sorted)}

        for row in raw_candidates:
            candidates.append(
                CandidateRow(
                    reference_date=reference_date,
                    legal_entity_id=buyer_id,
                    eclass_id=row["eclass_id"],
                    eclass=row["eclass"],
                    history_cohort=cohort,
                    n_orders_in_lookback=row["n_orders_in_lookback"],
                    n_active_months_in_lookback=row["n_active_months_in_lookback"],
                    lookback_spend=row["lookback_spend"],
                    total_orders=row["total_orders"],
                    total_spend=row["total_spend"],
                    avg_spend_per_order=row["avg_spend_per_order"],
                    first_order_date=row["first_order_date"],
                    last_order_date=row["last_order_date"],
                    history_span_days=row["history_span_days"],
                    score=row["score"],
                    rank=rank_map.get(row["eclass"], 0),
                    passes_eta_gate=row["passes_eta_gate"],
                    passes_tau_gate=row["passes_tau_gate"],
                    is_eligible=row["is_eligible"],
                )
            )

    return candidates


def compute_score(
    *,
    n_orders_in_lookback: int,
    days_since_last: int,
    avg_spend_per_order: float,
    savings_rate: float,
) -> tuple[float, float]:
    recurrence_weight = n_orders_in_lookback * math.exp(-(1.0 / 365.0) * days_since_last)
    score = recurrence_weight * avg_spend_per_order * savings_rate
    return score, recurrence_weight


def passes_emission_gate(candidate: CandidateRow, *, eta: int, tau: float) -> bool:
    if not candidate.is_eligible:
        return False
    if candidate.history_cohort == SPARSE_HISTORY_COHORT:
        if candidate.n_orders_in_lookback < eta * SPARSE_HISTORY_ETA_MULTIPLIER:
            return False
        if candidate.lookback_spend < tau * SPARSE_HISTORY_TAU_MULTIPLIER:
            return False
    return True


def passes_sparse_gate(candidate: CandidateRow, *, eta: int, tau: float) -> bool:
    if candidate.history_cohort != SPARSE_HISTORY_COHORT:
        return True
    return (
        candidate.n_orders_in_lookback >= eta * SPARSE_HISTORY_ETA_MULTIPLIER
        and candidate.lookback_spend >= tau * SPARSE_HISTORY_TAU_MULTIPLIER
    )


def rank_candidates(
    candidates: list[CandidateRow],
    *,
    eta: int,
    tau: float,
    savings_rate: float,
) -> list[RankedRow]:
    emittable = [candidate for candidate in candidates if passes_emission_gate(candidate, eta=eta, tau=tau)]
    by_buyer: dict[str, list[CandidateRow]] = {}
    for candidate in emittable:
        by_buyer.setdefault(candidate.legal_entity_id, []).append(candidate)

    ranked_rows: list[RankedRow] = []
    for buyer_id in sorted(by_buyer):
        buyer_candidates = sorted(
            by_buyer[buyer_id],
            key=lambda candidate: (-candidate.score, candidate.eclass),
        )
        for rank_index, candidate in enumerate(buyer_candidates, start=1):
            days_since_last = (candidate.reference_date - candidate.last_order_date).days
            _, recurrence_weight = compute_score(
                n_orders_in_lookback=candidate.n_orders_in_lookback,
                days_since_last=days_since_last,
                avg_spend_per_order=candidate.avg_spend_per_order,
                savings_rate=savings_rate,
            )
            ranked_rows.append(
                RankedRow(
                    reference_date=candidate.reference_date,
                    legal_entity_id=buyer_id,
                    eclass_id=candidate.eclass_id,
                    eclass=candidate.eclass,
                    history_cohort=candidate.history_cohort,
                    score=candidate.score,
                    rank=rank_index,
                    n_orders_in_lookback=candidate.n_orders_in_lookback,
                    n_active_months_in_lookback=candidate.n_active_months_in_lookback,
                    days_since_last_order=days_since_last,
                    recurrence_weight=recurrence_weight,
                    avg_spend_per_order=candidate.avg_spend_per_order,
                    lookback_spend=candidate.lookback_spend,
                    passes_sparse_gate=passes_sparse_gate(candidate, eta=eta, tau=tau),
                )
            )

    return ranked_rows


def write_ranked_parquet(ranked_rows: list[RankedRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row.to_row() for row in ranked_rows])
    if frame.empty:
        frame = pd.DataFrame(
            columns=[
                "reference_date",
                "legal_entity_id",
                "eclass_id",
                "eclass",
                "history_cohort",
                "score",
                "rank",
                "n_orders_in_lookback",
                "n_active_months_in_lookback",
                "days_since_last_order",
                "recurrence_weight",
                "avg_spend_per_order",
                "lookback_spend",
                "passes_sparse_gate",
            ]
        )
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(table, output_path, compression="zstd", use_dictionary=False)


def write_submission_csv(
    ranked_rows: list[RankedRow],
    *,
    output_path: Path,
    target_buyers: list[str],
) -> None:
    buyer_set = set(target_buyers)
    frame = pd.DataFrame(
        [
            {
                "legal_entity_id": row.legal_entity_id,
                "cluster": row.eclass_id.removeprefix(ECLASS_ID_PREFIX),
            }
            for row in ranked_rows
            if row.legal_entity_id in buyer_set
        ]
    )
    if frame.empty:
        frame = pd.DataFrame(columns=["legal_entity_id", "cluster"])
    else:
        frame["legal_entity_id"] = frame["legal_entity_id"].astype("string")
        frame["cluster"] = frame["cluster"].astype("string")
        frame = frame.sort_values(["legal_entity_id", "cluster"], kind="mergesort").reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def main() -> None:
    args = build_parser().parse_args()
    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()
    reference_date = args.reference_date

    validate_raw_inputs(raw_root)
    normalized_transactions = load_plis_training(raw_root)
    target_buyers = load_customer_test_buyers(raw_root)

    phase1_normalized_root = output_root / "phase1" / "normalized"
    phase1_entities_root = output_root / "phase1" / "entities"
    phase3_ranked_root = output_root / "phase3" / "ranked_evidence"
    phase3_submission_root = output_root / "phase3" / "submission"

    phase1_normalized_root.mkdir(parents=True, exist_ok=True)
    phase1_entities_root.mkdir(parents=True, exist_ok=True)

    transactions_path = phase1_normalized_root / "transactions.parquet"
    eclass_path = phase1_entities_root / "eclass.parquet"
    normalized_transactions.to_parquet(transactions_path, index=False)

    eclass_entities = build_eclass_entities(normalized_transactions)
    eclass_entities.to_parquet(eclass_path, index=False)
    eclass_id_map = dict(zip(eclass_entities["eclass"], eclass_entities["eclass_id"]))

    pre_boundary_transactions = load_pre_boundary_transactions(normalized_transactions, reference_date)
    candidates = build_warm_start_candidates(
        transactions=pre_boundary_transactions,
        eclass_id_map=eclass_id_map,
        reference_date=reference_date,
        target_buyers=target_buyers,
        lookback_months=args.lookback_months,
        eta=args.eta,
        tau=args.tau,
        savings_rate=args.savings_rate,
    )
    ranked_rows = rank_candidates(
        candidates,
        eta=args.eta,
        tau=args.tau,
        savings_rate=args.savings_rate,
    )

    ranked_path = phase3_ranked_root / f"ranked_{reference_date.isoformat()}.parquet"
    submission_path = phase3_submission_root / f"submission_{reference_date.isoformat()}.csv"
    write_ranked_parquet(ranked_rows, ranked_path)
    write_submission_csv(ranked_rows, output_path=submission_path, target_buyers=target_buyers)

    print("normalized_transactions_rows", len(normalized_transactions))
    print("eclass_rows", len(eclass_entities))
    print("ranked_rows", len(ranked_rows))
    print("buyers_with_predictions", len({row.legal_entity_id for row in ranked_rows}))
    print("ranked_path", ranked_path)
    print("submission_path", submission_path)


if __name__ == "__main__":
    main()
```

## What The Code Does

The embedded script performs these steps directly:

1. Validates that the raw CSV files and required columns exist.
2. Normalizes the transaction table and canonicalizes eclass values.
3. Builds the Level 1 eclass entity table.
4. Filters history to rows on or before the reference date.
5. Drops rows with non-canonical eclass values.
6. Drops zero-spend rows.
7. Builds buyer-by-eclass warm-start candidates from historical evidence only.
8. Applies the base order gate and spend gate.
9. Applies stricter sparse-history emission rules.
10. Computes the warm-start score from recurrence, recency, and monetary value.
11. Ranks the emitted predictions per buyer.
12. Writes the ranked evidence parquet and final `legal_entity_id,cluster` submission CSV.

## Final Note

This document now contains the processing code itself rather than instructions to import and run the project pipeline entrypoints.
