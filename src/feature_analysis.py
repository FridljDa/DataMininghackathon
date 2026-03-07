"""
Feature analysis: summary statistics and informativeness plots.

Reads full features parquet, computes per-feature stats (null rate, cardinality,
variance, std) and writes a summary CSV. Generates distribution and correlation
plots for downstream feature selection decisions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Optional matplotlib for plots
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    _HAS_PLOTTING = True
except ImportError:
    _HAS_PLOTTING = False


def _plot_temporal_series(plis: pd.DataFrame, value_col: str, ylabel: str, title: str, out_path: Path) -> None:
    """Plot sums by month, quarter, and year in one figure."""
    if plis.empty:
        return

    tmp = plis.copy()
    tmp["orderdate"] = pd.to_datetime(tmp["orderdate"], errors="coerce")
    tmp = tmp.dropna(subset=["orderdate"])
    if tmp.empty:
        return

    monthly = (
        tmp.set_index("orderdate")[value_col]
        .resample("ME")
        .sum()
    )
    quarterly = (
        tmp.set_index("orderdate")[value_col]
        .resample("QE")
        .sum()
    )
    yearly = (
        tmp.set_index("orderdate")[value_col]
        .resample("YE")
        .sum()
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    monthly.plot(ax=axes[0], color="#1f77b4")
    axes[0].set_title("By month")
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel(ylabel)

    quarterly.plot(ax=axes[1], marker="o", color="#ff7f0e")
    axes[1].set_title("By quarter")
    axes[1].set_xlabel("Quarter")
    axes[1].set_ylabel(ylabel)

    yearly.plot(ax=axes[2], marker="o", color="#2ca02c")
    axes[2].set_title("By year")
    axes[2].set_xlabel("Year")
    axes[2].set_ylabel(ylabel)

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, help="Path to features_all parquet.")
    parser.add_argument("--summary-csv", required=True, dest="summary_csv", help="Path to output feature summary CSV.")
    parser.add_argument("--plis", required=True, help="Path to PLIs TSV for temporal purchase plots.")
    parser.add_argument(
        "--distributions-plot",
        default="",
        dest="distributions_plot",
        help="Path to output distributions plot (optional).",
    )
    parser.add_argument(
        "--correlations-plot",
        default="",
        dest="correlations_plot",
        help="Path to output correlations plot (optional).",
    )
    parser.add_argument(
        "--value-by-period-plot",
        required=True,
        dest="value_by_period_plot",
        help="Path to output purchase value-by-period plot.",
    )
    parser.add_argument(
        "--quantity-by-period-plot",
        required=True,
        dest="quantity_by_period_plot",
        help="Path to output purchase quantity-by-period plot.",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.features)

    # Key columns always retained; exclude from numeric summary
    key_cols = {"legal_entity_id", "eclass"}
    feature_cols = [c for c in df.columns if c not in key_cols]

    rows = []
    for col in feature_cols:
        s = df[col]
        null_count = s.isna().sum()
        n = len(s)
        d = {
            "feature": col,
            "dtype": str(s.dtype),
            "n": n,
            "null_count": int(null_count),
            "null_rate": null_count / n if n else 0.0,
            "n_unique": int(s.nunique()),
        }
        if pd.api.types.is_numeric_dtype(s):
            d["mean"] = s.mean()
            d["std"] = s.std()
            d["min"] = s.min()
            d["max"] = s.max()
        rows.append(d)

    summary = pd.DataFrame(rows)
    out_csv = Path(args.summary_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)
    print(f"Wrote feature summary ({len(summary)} rows) to {out_csv}")

    if not _HAS_PLOTTING:
        raise ImportError(
            "matplotlib/seaborn are required for mandatory feature-analysis plots."
        )

    numeric = df[feature_cols].select_dtypes(include=["number"])
    if numeric.empty:
        return

    if args.distributions_plot:
        p_path = Path(args.distributions_plot)
        p_path.parent.mkdir(parents=True, exist_ok=True)
        ncols = min(4, len(numeric.columns))
        nrows = (len(numeric.columns) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes.reshape(1, -1)
        elif ncols == 1:
            axes = axes.reshape(-1, 1)
        for ax, col in zip(axes.flat, numeric.columns):
            numeric[col].dropna().hist(ax=ax, bins=50, edgecolor="black", alpha=0.7)
            ax.set_title(col, fontsize=10)
            ax.set_xlabel("")
        for ax in axes.flat[len(numeric.columns) :]:
            ax.set_visible(False)
        plt.tight_layout()
        plt.savefig(p_path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"Wrote distributions plot to {p_path}")

    if args.correlations_plot:
        c_path = Path(args.correlations_plot)
        c_path.parent.mkdir(parents=True, exist_ok=True)
        corr = numeric.corr()
        _, ax = plt.subplots(figsize=(max(8, len(corr) * 0.4), max(6, len(corr) * 0.35)))
        sns.heatmap(corr, annot=len(corr) <= 12, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, square=True)
        plt.tight_layout()
        plt.savefig(c_path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"Wrote correlations plot to {c_path}")

    plis = pd.read_csv(args.plis, sep="\t", low_memory=False)
    required = {"orderdate", "quantityvalue", "vk_per_item"}
    missing = required.difference(plis.columns)
    if missing:
        raise ValueError(
            f"plis is missing required columns for temporal plots: {sorted(missing)}"
        )
    plis["quantityvalue"] = pd.to_numeric(plis["quantityvalue"], errors="coerce").fillna(0.0)
    plis["vk_per_item"] = pd.to_numeric(plis["vk_per_item"], errors="coerce").fillna(0.0)
    plis["purchase_value"] = plis["quantityvalue"] * plis["vk_per_item"]

    _plot_temporal_series(
        plis=plis,
        value_col="purchase_value",
        ylabel="Purchase value",
        title="Purchase value over time (month / quarter / year)",
        out_path=Path(args.value_by_period_plot),
    )
    print(f"Wrote value-by-period plot to {args.value_by_period_plot}")

    _plot_temporal_series(
        plis=plis,
        value_col="quantityvalue",
        ylabel="Quantity",
        title="Purchase quantity over time (month / quarter / year)",
        out_path=Path(args.quantity_by_period_plot),
    )
    print(f"Wrote quantity-by-period plot to {args.quantity_by_period_plot}")


if __name__ == "__main__":
    main()
