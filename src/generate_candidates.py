"""
Candidate generation for Level 1 (E-Class) core demand.

Reads split plis_training and customer metadata; restricts to warm buyers
(task == predict future or testing). Applies lookback window and singleton filter.
Output: one row per (legal_entity_id, eclass) with base columns needed for
feature engineering (no derived features). Orderdates stored as list of "YYYY-MM" strings.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _read_plis(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    for col in (
        "legal_entity_id",
        "orderdate",
        "eclass",
        "quantityvalue",
        "vk_per_item",
    ):
        if col not in df.columns:
            raise ValueError(f"plis must contain '{col}'. Got: {list(df.columns)}")
    return df


def _read_customer(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    for col in ("legal_entity_id", "task"):
        if col not in df.columns:
            raise ValueError(f"customer must contain '{col}'. Got: {list(df.columns)}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plis", required=True, help="Path to plis_training (split) TSV.")
    parser.add_argument("--customer", required=True, help="Path to customer metadata TSV.")
    parser.add_argument("--output", required=True, help="Path to output candidates raw parquet.")
    parser.add_argument(
        "--train-end",
        required=True,
        dest="train_end",
        help="Last date of train period (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--lookback-months",
        type=int,
        default=18,
        dest="lookback_months",
        help="Months of lookback for candidate recency (default: 18).",
    )
    parser.add_argument(
        "--min-spend-singleton",
        type=float,
        default=50.0,
        dest="min_spend_singleton",
        help="Drop singleton eclass with total spend below this (default: 50).",
    )
    args = parser.parse_args()

    plis_path = Path(args.plis)
    customer_path = Path(args.customer)
    out_path = Path(args.output)
    train_end = pd.Timestamp(args.train_end)
    lookback_months = args.lookback_months
    min_spend_singleton = args.min_spend_singleton

    plis = _read_plis(plis_path)
    plis["orderdate"] = pd.to_datetime(plis["orderdate"], format="%Y-%m-%d")
    plis["legal_entity_id"] = plis["legal_entity_id"].astype(str)
    plis["eclass"] = plis["eclass"].astype(str).str.strip().replace("nan", "")
    plis = plis[plis["eclass"] != ""]

    q = pd.to_numeric(plis["quantityvalue"], errors="coerce").fillna(0)
    v = pd.to_numeric(plis["vk_per_item"], errors="coerce").fillna(0)
    plis["_spend"] = q * v

    customer = _read_customer(customer_path)
    customer["legal_entity_id"] = customer["legal_entity_id"].astype(str)
    task_norm = customer["task"].str.strip().str.lower()
    warm_ids = set(
        customer.loc[
            task_norm.isin(["predict future", "testing"]),
            "legal_entity_id",
        ].unique().tolist()
    )

    plis_train = plis[plis["orderdate"] <= train_end].copy()
    plis_train = plis_train[plis_train["legal_entity_id"].isin(warm_ids)]

    t_min = train_end - pd.DateOffset(months=lookback_months)

    agg = (
        plis_train.groupby(["legal_entity_id", "eclass"])
        .agg(
            n_orders=("_spend", "count"),
            s_total=("_spend", "sum"),
            orderdate_min=("orderdate", "min"),
            orderdate_max=("orderdate", "max"),
            orderdates=("orderdate", lambda x: x.dt.to_period("M").unique().tolist()),
        )
        .reset_index()
    )

    agg["t_last"] = agg["orderdate_max"]
    candidates = agg[agg["t_last"] >= t_min].copy()

    drop_singleton = (candidates["n_orders"] == 1) & (
        candidates["s_total"] < min_spend_singleton
    )
    candidates = candidates[~drop_singleton].copy()

    # Store orderdates as list of "YYYY-MM" for parquet-safe serialization
    candidates["orderdates_str"] = candidates["orderdates"].apply(
        lambda periods: [f"{p.year:04d}-{p.month:02d}" for p in periods]
    )
    candidates = candidates.drop(columns=["orderdates"], errors="ignore")

    out_cols = [
        "legal_entity_id",
        "eclass",
        "n_orders",
        "s_total",
        "orderdate_min",
        "orderdate_max",
        "t_last",
        "orderdates_str",
    ]
    candidates = candidates[[c for c in out_cols if c in candidates.columns]]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_parquet(out_path, index=False)
    print(f"Wrote {len(candidates)} candidate rows to {out_path}")


if __name__ == "__main__":
    main()
