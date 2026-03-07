"""
Candidate generation for Level 1 (E-Class) core demand.

Reads split plis_training and customer metadata; restricts to hot buyers
(task == predict future or testing). Builds two candidate sets:
- Seen: every (legal_entity_id, eclass) from hot entities' historical PLIs.
- Trending: cartesian product of hot entities x eclasses from trending_classes.csv.

Unions both sets and enriches with train-period aggregates (defaults for trending-only pairs).
Output: one row per (legal_entity_id, eclass) with base columns for feature
engineering. Orderdates stored as list of "YYYY-MM" strings.
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


def _read_trending_classes(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    if "eclass" not in df.columns:
        raise ValueError(f"trending_classes must contain 'eclass'. Got: {list(df.columns)}")
    eclass = df["eclass"].astype(str).str.strip().str.replace("nan", "", regex=False)
    eclass = eclass[eclass != ""].unique()
    return pd.Series(eclass, name="eclass")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plis", required=True, help="Path to plis_training (split) TSV.")
    parser.add_argument("--customer", required=True, help="Path to customer metadata TSV.")
    parser.add_argument(
        "--trending-classes",
        required=True,
        dest="trending_classes",
        help="Path to trending eclasses CSV (column: eclass).",
    )
    parser.add_argument("--output", required=True, help="Path to output candidates raw parquet.")
    parser.add_argument(
        "--train-end",
        required=True,
        dest="train_end",
        help="Last date of train period (YYYY-MM-DD).",
    )
    args = parser.parse_args()

    plis_path = Path(args.plis)
    customer_path = Path(args.customer)
    trending_path = Path(args.trending_classes)
    out_path = Path(args.output)
    train_end = pd.Timestamp(args.train_end)

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
    hot_ids = set(
        customer.loc[
            task_norm.isin(["predict future", "testing"]),
            "legal_entity_id",
        ].unique().tolist()
    )

    trending_eclasses = _read_trending_classes(trending_path)

    plis_train = plis[plis["orderdate"] <= train_end].copy()
    plis_train = plis_train[plis_train["legal_entity_id"].isin(hot_ids)]

    # Set A (seen): every (legal_entity_id, eclass) hot entity ever bought in train period
    seen = (
        plis_train[["legal_entity_id", "eclass"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    n_seen = len(seen)

    # Set B (trending cross): hot entity x trending eclass
    hot_df = pd.DataFrame({"legal_entity_id": list(hot_ids)})
    trend_df = trending_eclasses.to_frame()
    trending_cross = hot_df.assign(_k=1).merge(trend_df.assign(_k=1), on="_k").drop(columns=["_k"])
    n_trending_cross = len(trending_cross)

    # Union keys (deduplicate: seen pairs may also appear in trending cross)
    union_keys = (
        pd.concat([seen, trending_cross[["legal_entity_id", "eclass"]]], ignore_index=True)
        .drop_duplicates(subset=["legal_entity_id", "eclass"])
        .reset_index(drop=True)
    )

    # Full train-period aggregates for feature engineering
    agg = (
        plis_train.groupby(["legal_entity_id", "eclass"])
        .agg(
            n_orders=("_spend", "count"),
            historical_purchase_value_total=("_spend", "sum"),
            orderdate_min=("orderdate", "min"),
            orderdate_max=("orderdate", "max"),
            orderdates=("orderdate", lambda x: x.dt.to_period("M").unique().tolist()),
        )
        .reset_index()
    )
    agg["orderdates_str"] = agg["orderdates"].apply(
        lambda periods: [f"{p.year:04d}-{p.month:02d}" for p in periods]
    )
    agg = agg.drop(columns=["orderdates"], errors="ignore")

    # Left join union keys to aggregates; fill missing (trending-only pairs) with defaults
    candidates = union_keys.merge(
        agg,
        on=["legal_entity_id", "eclass"],
        how="left",
    )
    candidates["n_orders"] = candidates["n_orders"].fillna(0).astype(int)
    candidates["historical_purchase_value_total"] = candidates["historical_purchase_value_total"].fillna(0.0)
    candidates["orderdate_min"] = candidates["orderdate_min"].fillna(pd.NaT)
    candidates["orderdate_max"] = candidates["orderdate_max"].fillna(pd.NaT)
    candidates["orderdates_str"] = candidates["orderdates_str"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    out_cols = [
        "legal_entity_id",
        "eclass",
        "n_orders",
        "historical_purchase_value_total",
        "orderdate_min",
        "orderdate_max",
        "orderdates_str",
    ]
    candidates = candidates[[c for c in out_cols if c in candidates.columns]]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_parquet(out_path, index=False)

    print(f"Candidate set sizes: seen (hot x history) = {n_seen}, trending (hot x trending eclasses) = {n_trending_cross}")
    print(f"Wrote {len(candidates)} candidate rows to {out_path}")
