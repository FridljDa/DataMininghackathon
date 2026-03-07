"""
Sanity-check whether vk_per_item behaves like a deterministic lookup and whether
prices drift materially over time. Writes summary CSV and two plots to data/06_plots.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PLOT_LOOKUP = "vk_lookup_like_share_by_key.png"
PLOT_DRIFT = "vk_material_drift_by_key.png"
SUMMARY_CSV = "vk_price_stability_summary.csv"

KEY_LEVELS = [
    ("eclass", ["eclass"]),
    ("buyer_eclass", ["legal_entity_id", "eclass"]),
    ("buyer_eclass_mfr", ["legal_entity_id", "eclass", "manufacturer"]),
]


def load_plis(plis_path: Path) -> pd.DataFrame:
    cols = ["orderdate", "legal_entity_id", "eclass", "manufacturer", "quantityvalue", "vk_per_item"]
    df = pd.read_csv(plis_path, sep="\t", usecols=cols, low_memory=False)
    df["orderdate"] = pd.to_datetime(df["orderdate"], format="%Y-%m-%d")
    df["spend"] = df["quantityvalue"].astype(float) * df["vk_per_item"].astype(float)
    df["eclass"] = df["eclass"].astype(str).str.strip()
    df["legal_entity_id"] = df["legal_entity_id"].astype(str).str.strip()
    df["manufacturer"] = df["manufacturer"].fillna("").astype(str).str.strip()
    return df.dropna(subset=["orderdate", "quantityvalue", "vk_per_item"])


def lookup_like_metrics(df: pd.DataFrame, key_cols: list[str], min_share: float) -> dict:
    """Spend-weighted share of keys where one vk_per_item dominates (>= min_share of key spend)."""
    total_spend = df["spend"].sum()
    if total_spend <= 0:
        return {"pct_spend_lookup_like": 0.0, "n_keys": 0, "n_keys_lookup_like": 0}

    # Per (key, vk_per_item): spend
    by_key_vk = df.groupby(key_cols + ["vk_per_item"], dropna=False)["spend"].sum().reset_index()
    # Per key: total spend and max spend (dominant vk)
    key_totals = by_key_vk.groupby(key_cols)["spend"].agg(["sum", "max"]).reset_index()
    key_totals["top_share"] = key_totals["max"] / key_totals["sum"].replace(0, np.nan)
    lookup_like = key_totals["top_share"] >= min_share
    spend_lookup_like = key_totals.loc[lookup_like, "sum"].sum()
    return {
        "pct_spend_lookup_like": (spend_lookup_like / total_spend * 100) if total_spend else 0.0,
        "n_keys": len(key_totals),
        "n_keys_lookup_like": lookup_like.sum(),
    }


def drift_metrics(
    df: pd.DataFrame,
    key_cols: list[str],
    early_months: int,
    late_months: int,
    drift_pct_threshold: float,
    min_orders_per_period: int,
) -> tuple[dict, pd.DataFrame]:
    """
    Early-vs-late median vk_per_item per key; flag material drift.
    Returns summary dict and per-key pct_change for plotting.
    """
    total_spend = df["spend"].sum()
    df = df.copy()
    df["ym"] = df["orderdate"].dt.to_period("M")

    # Global first/last months
    all_months = df["ym"].unique()
    if len(all_months) < early_months + late_months:
        return (
            {"pct_spend_material_drift": 0.0, "n_keys_drift": 0},
            pd.DataFrame(columns=["key_level", "pct_change_abs"]),
        )

    all_months = sorted(all_months)
    early_yms = set(all_months[:early_months])
    late_yms = set(all_months[-late_months:])

    early_df = df[df["ym"].isin(early_yms)].groupby(key_cols).agg(
        median_vk=("vk_per_item", "median"),
        n=("spend", "size"),
        spend=("spend", "sum"),
    ).reset_index()
    late_df = df[df["ym"].isin(late_yms)].groupby(key_cols).agg(
        median_vk=("vk_per_item", "median"),
        n=("spend", "size"),
    ).reset_index()

    early_df = early_df.rename(columns={"median_vk": "median_vk_early", "n": "n_early", "spend": "spend_key"})
    late_df = late_df.rename(columns={"median_vk": "median_vk_late", "n": "n_late"})
    join = early_df.merge(
        late_df,
        on=key_cols,
        how="inner",
    )
    join = join[(join["n_early"] >= min_orders_per_period) & (join["n_late"] >= min_orders_per_period)]
    join["pct_change"] = (join["median_vk_late"] / join["median_vk_early"].replace(0, np.nan)) - 1
    join["pct_change_abs"] = join["pct_change"].abs()
    join["material_drift"] = join["pct_change_abs"] >= drift_pct_threshold

    spend_drift = join.loc[join["material_drift"], "spend_key"].sum()
    pct_spend_drift = (spend_drift / total_spend * 100) if total_spend else 0.0
    summary = {
        "pct_spend_material_drift": pct_spend_drift,
        "n_keys_drift": join["material_drift"].sum(),
        "n_keys_with_drift_metric": len(join),
    }
    return summary, join[["pct_change_abs", "spend_key"]].copy()


def run_checks(
    df: pd.DataFrame,
    lookup_min_share: float,
    early_months: int,
    late_months: int,
    drift_pct_threshold: float,
    min_orders_per_period: int,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run lookup-like and drift checks for each key level. Returns summary rows and drift dfs for plots."""
    rows = []
    drift_dfs = {}

    for key_name, key_cols in KEY_LEVELS:
        lookup = lookup_like_metrics(df, key_cols, lookup_min_share)
        drift_summary, drift_detail = drift_metrics(
            df, key_cols, early_months, late_months, drift_pct_threshold, min_orders_per_period
        )
        rows.append({
            "key_level": key_name,
            "pct_spend_lookup_like": lookup["pct_spend_lookup_like"],
            "n_keys": lookup["n_keys"],
            "n_keys_lookup_like": lookup["n_keys_lookup_like"],
            "pct_spend_material_drift": drift_summary["pct_spend_material_drift"],
            "n_keys_drift": drift_summary["n_keys_drift"],
            "n_keys_with_drift_metric": drift_summary["n_keys_with_drift_metric"],
        })
        drift_detail["key_level"] = key_name
        drift_dfs[key_name] = drift_detail

    summary_df = pd.DataFrame(rows)
    summary_df["total_spend"] = df["spend"].sum()
    return summary_df, drift_dfs


def plot_lookup_share(summary_df: pd.DataFrame, out_path: Path) -> None:
    """Spend-weighted share of lookup-like keys by key level."""
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(summary_df))
    ax.bar(x, summary_df["pct_spend_lookup_like"], color="steelblue", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["key_level"], rotation=15, ha="right")
    ax.set_ylabel("% of spend in lookup-like keys")
    ax.set_xlabel("Key level")
    ax.set_title("vk_per_item lookup-like share by key (spend-weighted)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()


def plot_material_drift(drift_dfs: dict[str, pd.DataFrame], out_path: Path) -> None:
    """Distribution of |pct_change| early vs late by key level (spend-weighted histogram or boxplot)."""
    combined = []
    for key_level, d in drift_dfs.items():
        if d.empty:
            continue
        for _, row in d.iterrows():
            combined.append({"key_level": key_level, "pct_change_abs": row["pct_change_abs"] * 100, "spend_key": row["spend_key"]})
    if not combined:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title("vk_per_item material drift by key (no keys with sufficient data)")
        fig.savefig(out_path, dpi=150)
        plt.close()
        return
    plot_df = pd.DataFrame(combined)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=plot_df, x="key_level", y="pct_change_abs", ax=ax)
    ax.axhline(y=20, color="red", linestyle="--", alpha=0.7, label="20% threshold")
    ax.set_ylabel("|Early→Late % change|")
    ax.set_xlabel("Key level")
    ax.set_title("vk_per_item early-vs-late % change by key level")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plis", required=True, help="Path to cleaned plis (e.g. data/02_clean/plis_training.csv)")
    parser.add_argument("--output-dir", required=True, dest="output_dir", help="Output directory (e.g. data/06_plots)")
    parser.add_argument(
        "--lookup-min-share",
        dest="lookup_min_share",
        type=float,
        default=0.95,
        help="Min spend share of dominant vk_per_item within key to count as lookup-like (default 0.95)",
    )
    parser.add_argument("--early-months", type=int, default=6, help="Number of months for early window (default 6)")
    parser.add_argument("--late-months", type=int, default=6, help="Number of months for late window (default 6)")
    parser.add_argument(
        "--drift-threshold",
        dest="drift_threshold",
        type=float,
        default=0.20,
        help="|pct_change| >= this to flag material drift (default 0.20)",
    )
    parser.add_argument(
        "--min-orders-per-period",
        dest="min_orders_per_period",
        type=int,
        default=2,
        help="Min orders in early and late window per key to compute drift (default 2)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_plis(Path(args.plis))
    summary_df, drift_dfs = run_checks(
        df,
        lookup_min_share=args.lookup_min_share,
        early_months=args.early_months,
        late_months=args.late_months,
        drift_pct_threshold=args.drift_threshold,
        min_orders_per_period=args.min_orders_per_period,
    )

    summary_df.to_csv(out_dir / SUMMARY_CSV, index=False)
    plot_lookup_share(summary_df, out_dir / PLOT_LOOKUP)
    plot_material_drift(drift_dfs, out_dir / PLOT_DRIFT)


if __name__ == "__main__":
    main()
