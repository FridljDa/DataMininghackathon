"""
Derived feature engineering: raw features + plis/customer/nace -> full feature matrix.

Reads data/08_features_raw/{mode}/features_raw.parquet and adds all derived
features (m_active, rho_freq, delta_recency, gap stats, economic, trend,
buyer context, etc.). Output: data/09_features_derived/{mode}/features_all.parquet.
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


def _read_nace_codes(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    df["nace_code"] = df["nace_code"].astype(str).str.strip()
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-raw", required=True, dest="features_raw", help="Path to features_raw parquet.")
    parser.add_argument("--plis", required=True, help="Path to plis_training (split) TSV.")
    parser.add_argument("--customer", required=True, help="Path to customer metadata TSV.")
    parser.add_argument("--nace-codes", required=False, dest="nace_codes", help="Path to nace_codes.csv reference.")
    parser.add_argument("--output", required=True, help="Path to output features_all parquet.")
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
        help="Lookback window in months for Phase 3 features (default: 18).",
    )
    args = parser.parse_args()

    raw_path = Path(args.features_raw)
    plis_path = Path(args.plis)
    customer_path = Path(args.customer)
    out_path = Path(args.output)
    train_end = pd.Timestamp(args.train_end)
    lookback_months = args.lookback_months

    df = pd.read_parquet(raw_path)
    for col in ("legal_entity_id", "eclass", "n_orders", "historical_purchase_value_total", "orderdate_min", "orderdate_max", "orderdates_str"):
        if col not in df.columns:
            raise ValueError(f"features_raw must contain '{col}'. Got: {list(df.columns)}")

    def _str_to_periods(ss) -> list:
        if ss is None:
            return []
        if isinstance(ss, np.ndarray):
            ss = ss.tolist()
        if not isinstance(ss, list) or len(ss) == 0:
            return []
        return [pd.Period(s, freq="M") for s in ss if isinstance(s, str)]

    df["orderdates"] = df["orderdates_str"].apply(_str_to_periods)

    # m_active, m_observed, rho_freq
    df["m_active"] = df["orderdates"].apply(len)
    df["_span_months"] = (
        (df["orderdate_max"] - df["orderdate_min"]).dt.days / 30.44
    ).clip(lower=1)
    df["m_observed"] = df["_span_months"].round().astype(int).clip(lower=1)
    df["rho_freq"] = df["m_active"] / df["m_observed"]

    # delta_recency
    df["t_last"] = df["orderdate_max"]
    df["delta_recency"] = (
        (train_end - df["t_last"]).dt.days / 30.44
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

    gap_results = list(zip(*df["orderdates"].map(_gaps)))
    df["_gap_mean"] = gap_results[0]
    df["_gap_std"] = gap_results[1]
    df["sigma_gap"] = df["_gap_std"]
    df["mu_gap"] = df["_gap_mean"]
    df["CV_gap"] = np.where(
        df["mu_gap"] > 0,
        df["sigma_gap"] / df["mu_gap"],
        np.nan,
    )

    # Economic
    df["historical_purchase_value_sqrt"] = np.sqrt(df["historical_purchase_value_total"])

    plis = _read_plis(plis_path)
    plis["orderdate"] = pd.to_datetime(plis["orderdate"], format="%Y-%m-%d")
    plis["legal_entity_id"] = plis["legal_entity_id"].astype(str)
    plis["eclass"] = plis["eclass"].astype(str).str.strip().replace("nan", "")
    q = pd.to_numeric(plis["quantityvalue"], errors="coerce").fillna(0)
    v = pd.to_numeric(plis["vk_per_item"], errors="coerce").fillna(0)
    plis["_spend"] = q * v

    lookback_start = train_end - pd.DateOffset(months=lookback_months)
    plis_lookback = plis[
        (plis["orderdate"] >= lookback_start) & (plis["orderdate"] <= train_end)
    ]
    lookback_agg = (
        plis_lookback.groupby(["legal_entity_id", "eclass"])
        .agg(
            n_orders_in_lookback=("_spend", "count"),
            lookback_spend=("_spend", "sum"),
        )
        .reset_index()
    )
    df = df.merge(lookback_agg, on=["legal_entity_id", "eclass"], how="left")
    df["avg_spend_per_order"] = (
        df["lookback_spend"]
        / df["n_orders_in_lookback"].clip(lower=1)
    )
    df["days_since_last"] = (
        train_end - df["orderdate_max"]
    ).dt.days

    plis_train = plis[plis["orderdate"] <= train_end]
    buyer_active_months = (
        plis_train.groupby("legal_entity_id")["orderdate"]
        .apply(lambda x: x.dt.to_period("M").nunique())
        .reset_index(name="buyer_active_months_total")
    )
    buyer_first_seen = (
        plis_train.groupby("legal_entity_id")["orderdate"]
        .min()
        .reset_index(name="buyer_first_seen")
    )
    df = df.merge(buyer_active_months, on="legal_entity_id", how="left")
    df = df.merge(buyer_first_seen, on="legal_entity_id", how="left")
    df["history_cohort"] = df["buyer_active_months_total"].fillna(0).apply(
        lambda m: "sparse_history"
        if 0 < m <= 3
        else ("cold_start" if m == 0 else "rich_history")
    )

    df["buyer_tenure_months"] = (
        (train_end - df["buyer_first_seen"]).dt.days / 30.44
    ).clip(lower=1)
    df["pair_tenure_months"] = (
        (train_end - df["orderdate_min"]).dt.days / 30.44
    ).clip(lower=1)
    effective_start = np.maximum(
        lookback_start,
        df["buyer_first_seen"].fillna(lookback_start),
    )
    df["effective_lookback_months"] = (
        (train_end - effective_start).dt.days / 30.44
    ).clip(lower=1)

    df["avg_monthly_orders_buyer_tenure"] = (
        df["n_orders"] / df["buyer_tenure_months"]
    )
    df["avg_monthly_orders_pair_tenure"] = (
        df["n_orders"] / df["pair_tenure_months"]
    )
    df["avg_monthly_spend_buyer_tenure"] = (
        df["historical_purchase_value_total"] / df["buyer_tenure_months"]
    )
    df["avg_monthly_spend_pair_tenure"] = (
        df["historical_purchase_value_total"] / df["pair_tenure_months"]
    )
    df["avg_monthly_orders_in_lookback"] = (
        df["n_orders_in_lookback"].fillna(0) / df["effective_lookback_months"]
    )
    df["avg_monthly_spend_in_lookback"] = (
        df["lookback_spend"].fillna(0) / df["effective_lookback_months"]
    )

    line_median = (
        plis.groupby(["legal_entity_id", "eclass"])["_spend"]
        .median()
        .reset_index()
        .rename(columns={"_spend": "s_median_line"})
    )
    df = df.merge(line_median, on=["legal_entity_id", "eclass"], how="left")

    buyer_total = (
        plis.groupby("legal_entity_id")["_spend"]
        .sum()
        .reset_index()
        .rename(columns={"_spend": "s_total_buyer"})
    )
    df = df.merge(buyer_total, on="legal_entity_id", how="left")
    df["w_e_b"] = np.where(
        df["s_total_buyer"] > 0,
        df["historical_purchase_value_total"] / df["s_total_buyer"],
        0.0,
    )

    def _month_diff(end: pd.Period, p: pd.Period) -> int:
        return (end.year - p.year) * 12 + (end.month - p.month)

    def _trend(periods: list, train_end_ts: pd.Timestamp) -> float:
        if not periods:
            return np.nan
        end = train_end_ts.to_period("M")
        last_3 = sum(1 for p in periods if _month_diff(end, p) <= 3)
        prior_6 = sum(1 for p in periods if 3 < _month_diff(end, p) <= 9)
        return float(last_3 - prior_6 / 2.0)

    df["delta_trend"] = df["orderdates"].map(lambda x: _trend(x, train_end))

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

    last_month_results = list(zip(*df["orderdates"].map(_last_order_month_cos_sin)))
    df["last_order_month_cos"] = last_month_results[0]
    df["last_order_month_sin"] = last_month_results[1]

    last_quarter_results = list(zip(*df["orderdates"].map(_last_order_quarter_cos_sin)))
    df["last_order_quarter_cos"] = last_quarter_results[0]
    df["last_order_quarter_sin"] = last_quarter_results[1]

    df["q4_share"] = df["orderdates"].map(_q4_share)

    recent_results = list(zip(*df["orderdates"].map(lambda x: _recent_counts(x, end_period))))
    df["recent_3m_count"] = recent_results[0]
    df["recent_6m_count"] = recent_results[1]
    df["recent_3_over_6"] = np.where(
        df["recent_6m_count"] >= 1,
        df["recent_3m_count"] / df["recent_6m_count"],
        np.nan,
    )
    df["active_month_share_12m"] = df["orderdates"].map(
        lambda x: _active_month_share_12m(x, end_period)
    )

    def _first_last_year(periods: list) -> tuple[float, float]:
        if not periods:
            return np.nan, np.nan
        first = min(periods).year
        last = max(periods).year
        return float(first), float(last)

    year_results = list(zip(*df["orderdates"].map(_first_last_year)))
    df["first_order_year"] = year_results[0]
    df["last_order_year"] = year_results[1]
    train_year = end_period.year
    df["years_since_last_order"] = np.where(
        df["last_order_year"].notna(),
        train_year - df["last_order_year"],
        np.nan,
    )
    df["active_year_span"] = np.where(
        df["first_order_year"].notna() & df["last_order_year"].notna(),
        (df["last_order_year"] - df["first_order_year"]).astype(int) + 1,
        np.nan,
    )

    df["recency_to_gap_ratio"] = np.where(
        (df["mu_gap"] > 0) & df["mu_gap"].notna(),
        df["delta_recency"] / df["mu_gap"],
        np.nan,
    )
    df["delta_vs_expected_gap"] = np.where(
        df["mu_gap"].notna(),
        df["delta_recency"] - df["mu_gap"],
        np.nan,
    )
    df["is_overdue"] = np.where(
        (df["mu_gap"] > 0) & df["mu_gap"].notna(),
        (df["delta_recency"] > df["mu_gap"]).astype(float),
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
    df = df.merge(spend_recent, on=["legal_entity_id", "eclass"], how="left")

    customer = _read_customer(customer_path)
    customer["legal_entity_id"] = customer["legal_entity_id"].astype(str)

    if args.nace_codes:
        nace_ref = _read_nace_codes(Path(args.nace_codes))
        nace_hierarchy = nace_ref[["nace_code", "toplevel_section", "nace_3digits"]].drop_duplicates(subset="nace_code")
        customer = customer.merge(nace_hierarchy, on="nace_code", how="left")
        customer = customer.rename(columns={
            "toplevel_section": "nace_section",
            "nace_3digits": "nace_3"
        })

    context_cols = ["legal_entity_id", "estimated_number_employees", "nace_code", "secondary_nace_code"]
    if "nace_section" in customer.columns:
        context_cols += ["nace_section", "nace_3"]

    cust_sub = customer[[c for c in context_cols if c in customer.columns]].drop_duplicates(
        subset="legal_entity_id"
    )
    df = df.merge(cust_sub, on="legal_entity_id", how="left")

    emp = pd.to_numeric(df["estimated_number_employees"], errors="coerce").fillna(0)
    df["log_employees"] = np.log1p(emp)

    df["nace_2"] = df["nace_code"].astype(str).str.strip().str[:2].replace("nan", "")
    df["secondary_nace_2"] = df["secondary_nace_code"].astype(str).str.strip().str[:2].replace("nan", "")
    df["nace_mismatch"] = (
        (df["nace_2"] != "")
        & (df["secondary_nace_2"] != "")
        & (df["nace_2"] != df["secondary_nace_2"])
    ).astype(float)

    df["has_gap_history"] = (
        df["mu_gap"].notna() & df["sigma_gap"].notna()
    ).astype(float)
    df["has_recent_6m_activity"] = (
        df["recent_6m_count"].fillna(0) > 0
    ).astype(float)
    df["has_recent_spend_6m"] = (
        df["spend_recent_6m"].fillna(0) > 0
    ).astype(float)
    sec_nace = df["secondary_nace_code"].astype(str).str.strip().replace("nan", "")
    df["has_secondary_nace"] = (sec_nace != "").astype(float)
    df["has_multi_year_history"] = (
        df["active_year_span"].fillna(0) > 1
    ).astype(float)

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
        "nace_section",
        "nace_3",
        "nace_2",
        "nace_code",
        "secondary_nace_2",
        "secondary_nace_code",
        "nace_mismatch",
        "has_gap_history",
        "has_recent_6m_activity",
        "has_recent_spend_6m",
        "has_secondary_nace",
        "has_multi_year_history",
        "n_orders_in_lookback",
        "lookback_spend",
        "avg_spend_per_order",
        "days_since_last",
        "history_cohort",
        "buyer_tenure_months",
        "pair_tenure_months",
        "effective_lookback_months",
        "avg_monthly_orders_buyer_tenure",
        "avg_monthly_orders_pair_tenure",
        "avg_monthly_spend_buyer_tenure",
        "avg_monthly_spend_pair_tenure",
        "avg_monthly_orders_in_lookback",
        "avg_monthly_spend_in_lookback",
    ]

    df = df[[c for c in out_cols if c in df.columns]]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} feature rows to {out_path}")


if __name__ == "__main__":
    main()
