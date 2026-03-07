"""
Feature analysis: summary statistics and informativeness plots.

Reads full features parquet, computes per-feature stats (null rate, cardinality,
variance, std) and writes a summary CSV. Generates distribution and per-feature
vs historical_purchase_value_total correlation plots for downstream feature selection.
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

# Target column for correlation panels (historical train-period purchase value)
TARGET_COL = "historical_purchase_value_total"
# Columns to exclude from predictor list (keys + target and its direct transform)
EXCLUDE_FROM_PREDICTORS = {"legal_entity_id", "eclass", TARGET_COL, "historical_purchase_value_sqrt"}

# Heuristic: use hexbin when row count above this to avoid overplotting
HEXBIN_MIN_ROWS = 1500


def _binned_median(x: pd.Series, y: pd.Series, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Compute bin centers (x) and median y per bin. Drops duplicate quantile edges."""
    valid = x.notna() & y.notna()
    xv, yv = x.loc[valid].values, y.loc[valid].values
    if len(xv) < 2 or n_bins < 2:
        return np.array([]), np.array([])
    edges = np.nanpercentile(xv, np.linspace(0, 100, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 2:
        return np.array([]), np.array([])
    bin_idx = np.searchsorted(edges[1:], xv, side="right")
    bin_idx = np.clip(bin_idx, 0, len(edges) - 2)
    centers = (edges[:-1] + edges[1:]) / 2
    medians = np.full(len(centers), np.nan)
    for i in range(len(centers)):
        mask = bin_idx == i
        if mask.sum() > 0:
            medians[i] = np.nanmedian(yv[mask])
    valid_bins = np.isfinite(medians)
    return centers[valid_bins], medians[valid_bins]


def _plot_feature_vs_target(
    ax: plt.Axes,
    x: pd.Series,
    y: pd.Series,
    xlabel: str,
    rho: float,
    use_hexbin: bool,
) -> None:
    """Single panel: scatter or hexbin, binned median line, Spearman in title."""
    valid = x.notna() & y.notna()
    xv, yv = x.loc[valid], y.loc[valid]
    if xv.empty or yv.empty:
        ax.set_title(f"{xlabel} (rho=n/a)")
        return
    if use_hexbin:
        ax.hexbin(xv, yv, gridsize=25, mincnt=1, cmap="Blues", edgecolors="none")
    else:
        ax.scatter(xv, yv, alpha=0.3, s=8, c="tab:blue", edgecolors="none")
    cx, my = _binned_median(x, y, n_bins=10)
    if len(cx) > 0:
        ax.plot(cx, my, color="red", linewidth=2, label="Median (binned)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(TARGET_COL)
    ax.set_title(f"{xlabel} (Spearman ρ = {rho:.3f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, help="Path to features_all parquet.")
    parser.add_argument("--summary-csv", required=True, dest="summary_csv", help="Path to output feature summary CSV.")
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
        help="Path to output per-feature vs target correlation plot (optional).",
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
        if TARGET_COL not in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, f"Target column '{TARGET_COL}' not in features.", ha="center", va="center", fontsize=12)
            ax.axis("off")
            plt.savefig(c_path, dpi=100, bbox_inches="tight")
            plt.close()
            print(f"Wrote placeholder correlations plot (missing target) to {c_path}")
        else:
            predictor_cols = [
                c for c in numeric.columns
                if c not in EXCLUDE_FROM_PREDICTORS
            ]
            if not predictor_cols:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.text(0.5, 0.5, "No predictor columns after exclusions.", ha="center", va="center", fontsize=12)
                ax.axis("off")
                plt.savefig(c_path, dpi=100, bbox_inches="tight")
                plt.close()
                print(f"Wrote placeholder correlations plot (no predictors) to {c_path}")
            else:
                n = len(df)
                use_hexbin = n >= HEXBIN_MIN_ROWS
                ncols = min(4, len(predictor_cols))
                nrows = (len(predictor_cols) + ncols - 1) // ncols
                fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
                if nrows == 1 and ncols == 1:
                    axes = np.array([[axes]])
                elif nrows == 1:
                    axes = axes.reshape(1, -1)
                elif ncols == 1:
                    axes = axes.reshape(-1, 1)
                y = df[TARGET_COL]
                for ax, col in zip(axes.flat, predictor_cols):
                    x = df[col]
                    valid = x.notna() & y.notna()
                    if valid.sum() < 2:
                        rho = np.nan
                    else:
                        rho = x.loc[valid].corr(y.loc[valid], method="spearman")
                    _plot_feature_vs_target(ax, x, y, col, float(rho) if not np.isnan(rho) else 0.0, use_hexbin)
                for ax in axes.flat[len(predictor_cols) :]:
                    ax.set_visible(False)
                plt.tight_layout()
                plt.savefig(c_path, dpi=100, bbox_inches="tight")
                plt.close()
                print(f"Wrote correlations plot to {c_path}")


if __name__ == "__main__":
    main()
