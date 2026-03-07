"""
Candidate generation for Level 1 or Level 2 core demand.

Level 1: key = (legal_entity_id, eclass). Seen = hot x history; trending = hot x trending eclasses.
Level 2: key = (legal_entity_id, eclass, manufacturer). Seen only from plis (manufacturer required).

Reads split plis_training and customer metadata; restricts to hot buyers
(task == predict future or testing).
Output: one row per key with key columns only (no aggregates). Aggregates are
computed in engineer_features_raw from PLIs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _read_plis(path: Path, require_manufacturer: bool = False) -> pd.DataFrame:
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
    if require_manufacturer and "manufacturer" not in df.columns:
        raise ValueError("plis must contain 'manufacturer' for level 2.")
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
    parser.add_argument(
        "--level",
        required=True,
        choices=("1", "2"),
        help="Level 1 = (legal_entity_id, eclass); Level 2 = (legal_entity_id, eclass, manufacturer).",
    )
    args = parser.parse_args()

    plis_path = Path(args.plis)
    customer_path = Path(args.customer)
    trending_path = Path(args.trending_classes)
    out_path = Path(args.output)
    train_end = pd.Timestamp(args.train_end)
    level = int(args.level)

    plis = _read_plis(plis_path, require_manufacturer=(level == 2))
    plis["orderdate"] = pd.to_datetime(plis["orderdate"], format="%Y-%m-%d")
    plis["legal_entity_id"] = plis["legal_entity_id"].astype(str)
    plis["eclass"] = plis["eclass"].astype(str).str.strip().replace("nan", "")
    plis = plis[plis["eclass"] != ""]
    if level == 2:
        plis["manufacturer"] = plis["manufacturer"].astype(str).str.strip().replace("nan", "")
        plis = plis[plis["manufacturer"] != ""]

    customer = _read_customer(customer_path)
    customer["legal_entity_id"] = customer["legal_entity_id"].astype(str)
    task_norm = customer["task"].str.strip().str.lower()
    hot_ids = set(
        customer.loc[
            task_norm.isin(["predict future", "testing"]),
            "legal_entity_id",
        ].unique().tolist()
    )

    plis_train = plis[plis["orderdate"] <= train_end].copy()
    plis_train = plis_train[plis_train["legal_entity_id"].isin(hot_ids)]

    if level == 1:
        # Set A (seen): every (legal_entity_id, eclass) hot entity ever bought in train period
        seen = (
            plis_train[["legal_entity_id", "eclass"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        n_seen = len(seen)
        trending_eclasses = _read_trending_classes(trending_path)
        hot_df = pd.DataFrame({"legal_entity_id": list(hot_ids)})
        trend_df = trending_eclasses.to_frame()
        trending_cross = hot_df.assign(_k=1).merge(trend_df.assign(_k=1), on="_k").drop(columns=["_k"])
        n_trending_cross = len(trending_cross)
        union_keys = (
            pd.concat([seen, trending_cross[["legal_entity_id", "eclass"]]], ignore_index=True)
            .drop_duplicates(subset=["legal_entity_id", "eclass"])
            .reset_index(drop=True)
        )
        group_cols = ["legal_entity_id", "eclass"]
    else:
        # Level 2: seen only, key = (legal_entity_id, eclass, manufacturer)
        union_keys = (
            plis_train[["legal_entity_id", "eclass", "manufacturer"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        n_seen = len(union_keys)
        n_trending_cross = 0
        group_cols = ["legal_entity_id", "eclass", "manufacturer"]

    out_cols = ["legal_entity_id", "eclass"]
    if level == 2:
        out_cols.append("manufacturer")
    candidates = union_keys[[c for c in out_cols if c in union_keys.columns]].copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_parquet(out_path, index=False)

    print(f"Candidate set sizes: seen = {n_seen}, trending = {n_trending_cross}")
    print(f"Wrote {len(candidates)} candidate rows to {out_path}")


if __name__ == "__main__":
    main()
