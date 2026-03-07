"""
Feature engineering for Level 1 (E-Class) candidates.

Reads raw candidates parquet (from generate_candidates), plis and customer metadata.
Computes all modelling features: m_active, rho_freq, delta_recency, gap stats,
economic features, trend, buyer context. Output: full feature matrix parquet.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _read_plis(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    for col in ("legal_entity_id", "orderdate", "eclass", "quantityvalue", "vk_per_item"):
        if col not in df.columns:
            raise ValueError(f"plis must contain '{col}'. Got: {list(df.columns)}")
    return df


def _read_customer(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    for col in ("legal_entity_id", "nace_code", "estimated_number_employees"):
        if col not in df.columns:
            raise ValueError(f"customer must contain '{col}'. Got: {list(df.columns)}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-raw", required=True, dest="candidates_raw", help="Path to raw candidates parquet.")
    parser.add_argument("--plis", required=True, help="Path to plis_training (split) TSV.")
    parser.add_argument("--customer", required=True, help="Path to customer metadata TSV.")
    parser.add_argument("--output", required=True, help="Path to output features parquet.")
    parser.add_argument(
        "--train-end",
        required=True,
        dest="train_end",
        help="Last date of train period (YYYY-MM-DD).",
    )
    args = parser.parse_args()

    raw_path = Path(args.candidates_raw)
    plis_path = Path(args.plis)
    customer_path = Path(args.customer)
    out_path = Path(args.output)
    train_end = pd.Timestamp(args.train_end)

    candidates = pd.read_parquet(raw_path)
    for col in ("legal_entity_id", "eclass", "n_orders", "historical_purchase_value_total", "orderdate_min", "orderdate_max", "orderdates_str"):
        if col not in candidates.columns:
            raise ValueError(f"candidates_raw must contain '{col}'. Got: {list(candidates.columns)}")

    # Restore orderdates as list of periods for feature computation (parquet may give list or ndarray)
    def _str_to_periods(ss) -> list:
        if ss is None:
            return []
        if isinstance(ss, np.ndarray):
            ss = ss.tolist()
        if not isinstance(ss, list) or len(ss) == 0:
            return []
        return [pd.Period(s, freq="M") for s in ss if isinstance(s, str)]

    candidates["orderdates"] = candidates["orderdates_str"].apply(_str_to_periods)

    # m_active, m_observed, rho_freq
    candidates["m_active"] = candidates["orderdates"].apply(len)
    candidates["_span_months"] = (
        (candidates["orderdate_max"] - candidates["orderdate_min"]).dt.days / 30.44
    ).clip(lower=1)
    candidates["m_observed"] = candidates["_span_months"].round().astype(int).clip(lower=1)
    candidates["rho_freq"] = candidates["m_active"] / candidates["m_observed"]

    # delta_recency
    candidates["t_last"] = candidates["orderdate_max"]
    candidates["delta_recency"] = (
        (train_end - candidates["t_last"]).dt.days / 30.44
    ).round().astype(int).clip(lower=0)

    # Inter-purchase gaps
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

    gap_results = list(zip(*candidates["orderdates"].map(_gaps)))
    candidates["_gap_mean"] = gap_results[0]
    candidates["_gap_std"] = gap_results[1]
    candidates["sigma_gap"] = candidates["_gap_std"]
    candidates["mu_gap"] = candidates["_gap_mean"]
    candidates["CV_gap"] = np.where(
        candidates["mu_gap"] > 0,
        candidates["sigma_gap"] / candidates["mu_gap"],
        np.nan,
    )

    # Economic
    candidates["historical_purchase_value_sqrt"] = np.sqrt(candidates["historical_purchase_value_total"])

    plis = _read_plis(plis_path)
    plis["orderdate"] = pd.to_datetime(plis["orderdate"], format="%Y-%m-%d")
    plis["legal_entity_id"] = plis["legal_entity_id"].astype(str)
    plis["eclass"] = plis["eclass"].astype(str).str.strip().replace("nan", "")
    q = pd.to_numeric(plis["quantityvalue"], errors="coerce").fillna(0)
    v = pd.to_numeric(plis["vk_per_item"], errors="coerce").fillna(0)
    plis["_spend"] = q * v

    line_median = (
        plis.groupby(["legal_entity_id", "eclass"])["_spend"]
        .median()
        .reset_index()
        .rename(columns={"_spend": "s_median_line"})
    )
    candidates = candidates.merge(
        line_median, on=["legal_entity_id", "eclass"], how="left"
    )

    buyer_total = (
        plis.groupby("legal_entity_id")["_spend"]
        .sum()
        .reset_index()
        .rename(columns={"_spend": "s_total_buyer"})
    )
    candidates = candidates.merge(buyer_total, on="legal_entity_id", how="left")
    candidates["w_e_b"] = np.where(
        candidates["s_total_buyer"] > 0,
        candidates["historical_purchase_value_total"] / candidates["s_total_buyer"],
        0.0,
    )

    # Trend
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

    # Calendar and momentum (aggregated per candidate)
    end_period = train_end.to_period("M")

    def _last_order_month_cos_sin(periods: list) -> tuple[float, float]:
        if not periods:
            return np.nan, np.nan
        last = max(periods)
        month_1based = last.month
        x = 2 * np.pi * (month_1based - 1) / 12
        return float(np.cos(x)), float(np.sin(x))

    def _last_order_quarter_cos_sin(periods: list) -> tuple[float, float]:
        if not periods:
            return np.nan, np.nan
        last = max(periods)
        quarter = (last.month - 1) // 3 + 1
        x = 2 * np.pi * (quarter - 1) / 4
        return float(np.cos(x)), float(np.sin(x))

    def _q4_share(periods: list) -> float:
        if not periods:
            return np.nan
        q4_count = sum(1 for p in periods if p.month >= 10)
        return float(q4_count) / len(periods)

    def _recent_counts(periods: list, end: pd.Period) -> tuple[float, float]:
        if not periods:
            return 0.0, 0.0
        n_3 = sum(1 for p in periods if _month_diff(end, p) <= 2)
        n_6 = sum(1 for p in periods if _month_diff(end, p) <= 5)
        return float(n_3), float(n_6)

    def _active_month_share_12m(periods: list, end: pd.Period) -> float:
        if not periods:
            return 0.0
        n_12 = sum(1 for p in periods if _month_diff(end, p) <= 11)
        return float(n_12) / 12.0

    last_month_results = list(zip(*candidates["orderdates"].map(_last_order_month_cos_sin)))
    candidates["last_order_month_cos"] = last_month_results[0]
    candidates["last_order_month_sin"] = last_month_results[1]

    last_quarter_results = list(zip(*candidates["orderdates"].map(_last_order_quarter_cos_sin)))
    candidates["last_order_quarter_cos"] = last_quarter_results[0]
    candidates["last_order_quarter_sin"] = last_quarter_results[1]

    candidates["q4_share"] = candidates["orderdates"].map(_q4_share)

    recent_results = list(zip(*candidates["orderdates"].map(lambda x: _recent_counts(x, end_period))))
    candidates["recent_3m_count"] = recent_results[0]
    candidates["recent_6m_count"] = recent_results[1]
    candidates["recent_3_over_6"] = np.where(
        candidates["recent_6m_count"] >= 1,
        candidates["recent_3m_count"] / candidates["recent_6m_count"],
        np.nan,
    )
    candidates["active_month_share_12m"] = candidates["orderdates"].map(
        lambda x: _active_month_share_12m(x, end_period)
    )

    # Explicit year features (relative to train_end)
    def _first_last_year(periods: list) -> tuple[float, float]:
        if not periods:
            return np.nan, np.nan
        first = min(periods).year
        last = max(periods).year
        return float(first), float(last)

    year_results = list(zip(*candidates["orderdates"].map(_first_last_year)))
    candidates["first_order_year"] = year_results[0]
    candidates["last_order_year"] = year_results[1]
    train_year = end_period.year
    candidates["years_since_last_order"] = np.where(
        candidates["last_order_year"].notna(),
        train_year - candidates["last_order_year"],
        np.nan,
    )
    candidates["active_year_span"] = np.where(
        candidates["first_order_year"].notna() & candidates["last_order_year"].notna(),
        (candidates["last_order_year"] - candidates["first_order_year"]).astype(int) + 1,
        np.nan,
    )

    candidates["recency_to_gap_ratio"] = np.where(
        (candidates["mu_gap"] > 0) & candidates["mu_gap"].notna(),
        candidates["delta_recency"] / candidates["mu_gap"],
        np.nan,
    )
    candidates["delta_vs_expected_gap"] = np.where(
        candidates["mu_gap"].notna(),
        candidates["delta_recency"] - candidates["mu_gap"],
        np.nan,
    )
    candidates["is_overdue"] = np.where(
        (candidates["mu_gap"] > 0) & candidates["mu_gap"].notna(),
        (candidates["delta_recency"] > candidates["mu_gap"]).astype(float),
        np.nan,
    )

    plis["order_month"] = plis["orderdate"].dt.to_period("M")
    plis["month_diff"] = (
        (end_period.year - plis["order_month"].dt.year) * 12
        + (end_period.month - plis["order_month"].dt.month)
    )
    plis["_spend_recent_3m"] = np.where(plis["month_diff"] <= 2, plis["_spend"], 0.0)
    plis["_spend_recent_6m"] = np.where(plis["month_diff"] <= 5, plis["_spend"], 0.0)
    spend_recent = (
        plis.groupby(["legal_entity_id", "eclass"], as_index=False)
        .agg(
            spend_recent_3m=("_spend_recent_3m", "sum"),
            spend_recent_6m=("_spend_recent_6m", "sum"),
            _order_value_mean=("_spend", "mean"),
            _order_value_std=("_spend", lambda x: x.std(ddof=0)),
        )
    )
    spend_recent["order_value_cv"] = np.where(
        spend_recent["_order_value_mean"] > 0,
        spend_recent["_order_value_std"] / spend_recent["_order_value_mean"],
        np.nan,
    )
    spend_recent = spend_recent.drop(
        columns=["_order_value_mean", "_order_value_std"], errors="ignore"
    )
    spend_recent["spend_recent_ratio"] = np.where(
        spend_recent["spend_recent_6m"] > 0,
        spend_recent["spend_recent_3m"] / spend_recent["spend_recent_6m"],
        np.nan,
    )
    candidates = candidates.merge(
        spend_recent,
        on=["legal_entity_id", "eclass"],
        how="left",
    )

    # Buyer context
    customer = _read_customer(customer_path)
    customer["legal_entity_id"] = customer["legal_entity_id"].astype(str)
    cust_sub = customer[["legal_entity_id", "estimated_number_employees", "nace_code"]].drop_duplicates(
        subset="legal_entity_id"
    )
    candidates = candidates.merge(cust_sub, on="legal_entity_id", how="left")
    emp = pd.to_numeric(candidates["estimated_number_employees"], errors="coerce").fillna(0)
    candidates["log_employees"] = np.log1p(emp)
    candidates["nace_2"] = candidates["nace_code"].astype(str).str[:2].replace("nan", "")

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
        "historical_purchase_value_total",
        "historical_purchase_value_sqrt",
        "s_median_line",
        "w_e_b",
        "delta_trend",
        "last_order_month_cos",
        "last_order_month_sin",
        "last_order_quarter_cos",
        "last_order_quarter_sin",
        "q4_share",
        "recent_3m_count",
        "recent_6m_count",
        "recent_3_over_6",
        "active_month_share_12m",
        "first_order_year",
        "last_order_year",
        "years_since_last_order",
        "active_year_span",
        "recency_to_gap_ratio",
        "delta_vs_expected_gap",
        "is_overdue",
        "spend_recent_3m",
        "spend_recent_6m",
        "spend_recent_ratio",
        "order_value_cv",
        "log_employees",
        "nace_2",
        "nace_code",
    ]
    candidates = candidates[[c for c in out_cols if c in candidates.columns]]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_parquet(out_path, index=False)
    print(f"Wrote {len(candidates)} feature rows to {out_path}")


if __name__ == "__main__":
    main()
