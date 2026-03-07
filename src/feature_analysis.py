"""
Feature analysis: summary statistics and informativeness plots.

Reads full features parquet and (optionally) plis to attach validation-period
label and s_val. Computes per-feature stats: null/zero rate, quantiles,
cardinality, variance; and target-aware stats (univariate signal vs recurrence
label and vs positive-case spend). Writes feature_summary.csv and optionally
feature_redundancy.csv and distribution/correlation plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

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
# Base key columns; manufacturer added when present (level 2)
EXCLUDE_FROM_PREDICTORS_BASE = {"legal_entity_id", "eclass", TARGET_COL, "historical_purchase_value_sqrt"}

# Heuristic: use hexbin when row count above this to avoid overplotting
HEXBIN_MIN_ROWS = 1500
# Minimum valid pairs for target-aware stats
MIN_VALID_FOR_SIGNAL = 20


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
        "--plis",
        default="",
        help="Path to plis_training (split) TSV for validation-period labels/spend (optional).",
    )
    parser.add_argument("--val-start", default="", dest="val_start", help="Validation period start (YYYY-MM-DD).")
    parser.add_argument("--val-end", default="", dest="val_end", help="Validation period end (YYYY-MM-DD).")
    parser.add_argument(
        "--n-min-label",
        type=int,
        default=1,
        dest="n_min_label",
        help="Min orders in val period for positive label (default: 1).",
    )
    parser.add_argument(
        "--redundancy-csv",
        default="",
        dest="redundancy_csv",
        help="Path to output feature-feature redundancy CSV (optional).",
    )
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

    # Infer level from schema: level 2 has manufacturer in key
    entity_key = ["legal_entity_id", "eclass", "manufacturer"] if "manufacturer" in df.columns else ["legal_entity_id", "eclass"]
    exclude_from_predictors = set(EXCLUDE_FROM_PREDICTORS_BASE)
    if "manufacturer" in df.columns:
        exclude_from_predictors.add("manufacturer")

    # Attach validation label and s_val if plis and val window provided
    label_series: pd.Series | None = None
    s_val_series: pd.Series | None = None
    if args.plis and args.val_start and args.val_end:
        plis = pd.read_csv(args.plis, sep="\t", low_memory=False)
        plis["orderdate"] = pd.to_datetime(plis["orderdate"], format="%Y-%m-%d")
        plis["legal_entity_id"] = plis["legal_entity_id"].astype(str)
        plis["eclass"] = plis["eclass"].astype(str).str.strip().replace("nan", "")
        val_start = pd.Timestamp(args.val_start)
        val_end = pd.Timestamp(args.val_end)
        plis = plis[(plis["orderdate"] >= val_start) & (plis["orderdate"] <= val_end)]
        q = pd.to_numeric(plis["quantityvalue"], errors="coerce").fillna(0)
        v = pd.to_numeric(plis["vk_per_item"], errors="coerce").fillna(0)
        plis["_spend"] = q * v
        plis = plis[plis["eclass"] != ""]
        if "manufacturer" in plis.columns and "manufacturer" in df.columns:
            plis["manufacturer"] = plis["manufacturer"].astype(str).str.strip().replace("nan", "")
            plis = plis[plis["manufacturer"] != ""]
        val_agg = (
            plis.groupby(entity_key)
            .agg(n_orders_val=("_spend", "count"), s_val=("_spend", "sum"))
            .reset_index()
        )
        df = df.merge(val_agg, on=entity_key, how="left")
        df["n_orders_val"] = df["n_orders_val"].fillna(0).astype(int)
        df["s_val"] = df["s_val"].fillna(0)
        df["label"] = (df["n_orders_val"] >= args.n_min_label).astype(int)
        label_series = df["label"]
        s_val_series = df["s_val"]

    # Key columns always retained; exclude from numeric summary
    key_cols = set(entity_key)
    feature_cols = [c for c in df.columns if c not in key_cols and c not in {"label", "s_val", "n_orders_val"}]

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
            zero_count = (s == 0).sum()
            d["zero_rate"] = zero_count / n if n else 0.0
            d["top_category_share"] = np.nan
            for qname, qval in [("p01", 1), ("p05", 5), ("p50", 50), ("p95", 95), ("p99", 99)]:
                d[qname] = s.quantile(qval / 100.0)
        else:
            d["zero_rate"] = np.nan
            vc = s.fillna("").value_counts(normalize=True, dropna=False)
            d["top_category_share"] = float(vc.iloc[0]) if len(vc) else np.nan
        # Target-aware stats (recurrence label and positive-case spend)
        if label_series is not None and pd.api.types.is_numeric_dtype(s):
            valid = s.notna() & label_series.notna()
            if valid.sum() >= MIN_VALID_FOR_SIGNAL and label_series.loc[valid].nunique() >= 2:
                try:
                    auc = roc_auc_score(label_series.loc[valid], s.loc[valid])
                    d["label_auc"] = max(auc, 1.0 - auc)
                except Exception:
                    d["label_auc"] = np.nan
                rho = s.loc[valid].corr(label_series.loc[valid].astype(float), method="spearman")
                d["abs_spearman_label"] = abs(float(rho)) if pd.notna(rho) else np.nan
            else:
                d["label_auc"] = np.nan
                d["abs_spearman_label"] = np.nan
            pos = label_series == 1
            if pos.sum() >= MIN_VALID_FOR_SIGNAL and s_val_series is not None:
                s_pos = s.loc[pos]
                v_pos = np.log1p(s_val_series.loc[pos])
                valid_pos = s_pos.notna() & v_pos.notna()
                if valid_pos.sum() >= MIN_VALID_FOR_SIGNAL:
                    rho_val = s_pos.loc[valid_pos].corr(v_pos.loc[valid_pos], method="spearman")
                    d["value_spearman_pos"] = float(rho_val) if pd.notna(rho_val) else np.nan
                else:
                    d["value_spearman_pos"] = np.nan
            else:
                d["value_spearman_pos"] = np.nan
        else:
            d["label_auc"] = np.nan
            d["abs_spearman_label"] = np.nan
            d["value_spearman_pos"] = np.nan
        rows.append(d)

    summary = pd.DataFrame(rows)
    out_csv = Path(args.summary_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)
    print(f"Wrote feature summary ({len(summary)} rows) to {out_csv}")

    # Feature-feature redundancy (numeric pairs with high |Spearman|)
    REDUNDANCY_THRESHOLD = 0.85
    if args.redundancy_csv:
        numeric_cols = [
            r["feature"] for _, r in summary.iterrows()
            if r["feature"] in df.columns and ("int" in str(r["dtype"]) or "float" in str(r["dtype"]))
        ]
        numeric_cols = [c for c in numeric_cols if c in df.columns]
        redundancy_rows = []
        if len(numeric_cols) >= 2:
            X = df[numeric_cols].copy()
            for c in numeric_cols:
                X[c] = pd.to_numeric(X[c], errors="coerce")
            corr = X.corr(method="spearman")
            for i, a in enumerate(numeric_cols):
                for j, b in enumerate(numeric_cols):
                    if i >= j:
                        continue
                    val = corr.iloc[i, j]
                    if pd.notna(val) and abs(val) >= REDUNDANCY_THRESHOLD:
                        redundancy_rows.append({"feature_a": a, "feature_b": b, "abs_spearman": round(abs(float(val)), 4)})
        redundancy_path = Path(args.redundancy_csv)
        redundancy_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(redundancy_rows).to_csv(redundancy_path, index=False)
        print(f"Wrote feature redundancy ({len(redundancy_rows)} pairs) to {redundancy_path}")

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
                if c not in exclude_from_predictors
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
