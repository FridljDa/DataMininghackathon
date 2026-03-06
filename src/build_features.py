"""
Candidate generation and feature engineering for Level 1 (E-Class) core demand.

Reads split plis_training and customer metadata; restricts to warm buyers (task == predict future).
Builds candidates with lookback window and singleton filter, then computes all features
from modelling.md §4-5. Output: one row per (legal_entity_id, eclass) with feature columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _read_plis(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    for col in (
        "legal_entity_id",
        "orderdate",
        "eclass",
        "quantityvalue",
        "vk_per_item",
        "nace_code",
        "estimated_number_employees",
    ):
        if col not in df.columns:
            raise ValueError(f"plis must contain '{col}'. Got: {list(df.columns)}")
    return df


def _read_customer(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    for col in ("legal_entity_id", "task", "nace_code", "estimated_number_employees"):
        if col not in df.columns:
            raise ValueError(f"customer must contain '{col}'. Got: {list(df.columns)}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plis", required=True, help="Path to plis_training (split) TSV.")
    parser.add_argument("--customer", required=True, help="Path to customer metadata TSV.")
    parser.add_argument("--output", required=True, help="Path to output candidates.parquet.")
    parser.add_argument(
        "--train-end",
        required=True,
        dest="train_end",
        help="Last date of train period (YYYY-MM-DD). Features use only data on or before this.",
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
        customer.loc[task_norm == "predict future", "legal_entity_id"].unique().tolist()
    )

    plis_train = plis[plis["orderdate"] <= train_end].copy()
    plis_train = plis_train[plis_train["legal_entity_id"].isin(warm_ids)]

    t_min = train_end - pd.DateOffset(months=lookback_months)

    # Aggregate per (legal_entity_id, eclass) in train period
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

    # Candidate filter: last purchase >= T - L
    agg["t_last"] = agg["orderdate_max"]
    candidates = agg[agg["t_last"] >= t_min].copy()

    # Singleton filter: n_orders == 1 and s_total < tau_s -> drop
    drop_singleton = (candidates["n_orders"] == 1) & (
        candidates["s_total"] < min_spend_singleton
    )
    candidates = candidates[~drop_singleton].copy()

    # --- Feature engineering ---
    # m_active: distinct months with at least one purchase
    candidates["m_active"] = candidates["orderdates"].apply(len)
    # m_observed: months between first and last purchase (buyer-level would need plis; use span of (b,e) as proxy)
    candidates["_span_months"] = (
        (candidates["orderdate_max"] - candidates["orderdate_min"]).dt.days / 30.44
    ).clip(lower=1)
    candidates["m_observed"] = candidates["_span_months"].round().astype(int).clip(lower=1)
    candidates["rho_freq"] = candidates["m_active"] / candidates["m_observed"]

    # Recency: months since last purchase (relative to train_end)
    candidates["delta_recency"] = (
        (train_end - candidates["t_last"]).dt.days / 30.44
    ).round().astype(int).clip(lower=0)

    # Inter-purchase gaps (months) for regularity
    def _gaps(periods: list) -> tuple[float, float]:
        if not periods or len(periods) < 2:
            return np.nan, np.nan
        p = sorted(set(periods))
        gaps = [
            (p[i + 1].year - p[i].year) * 12 + (p[i + 1].month - p[i].month)
            for i in range(len(p) - 1)
        ]
        if not gaps:
            return np.nan, np.nan
        return float(np.mean(gaps)), float(np.std(gaps)) if len(gaps) > 1 else 0.0

    candidates["_gap_mean"], candidates["_gap_std"] = zip(
        *candidates["orderdates"].map(_gaps)
    )
    candidates["sigma_gap"] = candidates["_gap_std"]
    candidates["mu_gap"] = candidates["_gap_mean"]
    candidates["CV_gap"] = np.where(
        candidates["mu_gap"] > 0,
        candidates["sigma_gap"] / candidates["mu_gap"],
        np.nan,
    )

    # Economic: s_total already; sqrt; median per line from plis
    candidates["s_total_sqrt"] = np.sqrt(candidates["s_total"])
    line_median = (
        plis_train.groupby(["legal_entity_id", "eclass"])["_spend"]
        .median()
        .reset_index()
        .rename(columns={"_spend": "s_median_line"})
    )
    candidates = candidates.merge(
        line_median, on=["legal_entity_id", "eclass"], how="left"
    )

    buyer_total = (
        plis_train.groupby("legal_entity_id")["_spend"]
        .sum()
        .reset_index()
        .rename(columns={"_spend": "s_total_buyer"})
    )
    candidates = candidates.merge(buyer_total, on="legal_entity_id", how="left")
    candidates["w_e_b"] = np.where(
        candidates["s_total_buyer"] > 0,
        candidates["s_total"] / candidates["s_total_buyer"],
        0.0,
    )

    # Trend: m_active last 3mo vs prior 6mo/2
    def _month_diff(end: pd.Period, p: pd.Period) -> int:
        return (end.year - p.year) * 12 + (end.month - p.month)

    def _trend(periods: list, train_end_ts: pd.Timestamp) -> float:
        if not periods:
            return np.nan
        end = train_end_ts.to_period("M")
        last_3 = sum(1 for p in periods if _month_diff(end, p) <= 3)
        prior_6 = sum(1 for p in periods if 3 < _month_diff(end, p) <= 9)
        return float(last_3 - prior_6 / 2.0)

    candidates["delta_trend"] = candidates["orderdates"].map(
        lambda x: _trend(x, train_end)
    )

    # Buyer context: merge customer
    cust_sub = customer[["legal_entity_id", "estimated_number_employees", "nace_code"]].drop_duplicates(subset="legal_entity_id")
    candidates = candidates.merge(cust_sub, on="legal_entity_id", how="left")
    emp = pd.to_numeric(candidates["estimated_number_employees"], errors="coerce").fillna(0)
    candidates["log_employees"] = np.log1p(emp)
    candidates["nace_2"] = candidates["nace_code"].astype(str).str[:2].replace("nan", "")

    # Drop temporary columns and orderdates (not serialized nicely)
    out_cols = [
        "legal_entity_id",
        "eclass",
        "n_orders",
        "m_active",
        "m_observed",
        "rho_freq",
        "delta_recency",
        "sigma_gap",
        "mu_gap",
        "CV_gap",
        "s_total",
        "s_total_sqrt",
        "s_median_line",
        "w_e_b",
        "delta_trend",
        "log_employees",
        "nace_2",
        "nace_code",
    ]
    candidates = candidates[[c for c in out_cols if c in candidates.columns]]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_parquet(out_path, index=False)
    print(f"Wrote {len(candidates)} candidate rows to {out_path}")


if __name__ == "__main__":
    main()
