"""
Two-stage LightGBM scorer for Level 1: EU = p_recur * v_hat * r - F.

Stage A: binary classifier for recurrence (label = n_orders_val >= n_min_label).
Stage B: regressor on s_val for positive examples only; predict v_hat for all.
Reads candidates.parquet and plis (split) to attach label and s_val; outputs
scores.parquet with score_base = EU so select_portfolio works unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    raise ImportError("lightgbm is required. Install with: uv add lightgbm")

FEATURE_COLS = [
    "n_orders",
    "m_active",
    "rho_freq",
    "delta_recency",
    "sigma_gap",
    "CV_gap",
    "s_total_sqrt",
    "s_median_line",
    "w_e_b",
    "delta_trend",
    "log_employees",
]
CATEGORICAL_FEATURES = ["nace_2"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, help="Path to candidates.parquet.")
    parser.add_argument(
        "--plis",
        required=True,
        help="Path to plis_training (split) TSV for validation-period labels/spend.",
    )
    parser.add_argument("--output", required=True, help="Path to output scores.parquet.")
    parser.add_argument("--val-start", required=True, dest="val_start", help="Validation period start (YYYY-MM-DD).")
    parser.add_argument("--val-end", required=True, dest="val_end", help="Validation period end (YYYY-MM-DD).")
    parser.add_argument(
        "--n-min-label",
        type=int,
        default=2,
        dest="n_min_label",
        help="Min orders in val period for positive label (default: 2).",
    )
    parser.add_argument(
        "--savings-rate",
        type=float,
        default=0.10,
        dest="savings_rate",
        help="Savings rate for EU (default: 0.10).",
    )
    parser.add_argument(
        "--fixed-fee-eur",
        type=float,
        default=10.0,
        dest="fixed_fee_eur",
        help="Fixed fee per element for EU (default: 10).",
    )
    parser.add_argument(
        "--val-months",
        type=float,
        default=6.0,
        dest="val_months",
        help="Validation window length in months for per-month normalisation (default: 6).",
    )
    parser.add_argument("--lgb-params-classifier", type=str, default="", dest="lgb_params_classifier")
    parser.add_argument("--lgb-params-regressor", type=str, default="", dest="lgb_params_regressor")
    args = parser.parse_args()

    candidates_path = Path(args.candidates)
    plis_path = Path(args.plis)
    out_path = Path(args.output)
    val_start = pd.Timestamp(args.val_start)
    val_end = pd.Timestamp(args.val_end)

    df = pd.read_parquet(candidates_path)
    for col in FEATURE_COLS:
        if col not in df.columns:
            raise ValueError(f"candidates must contain '{col}'. Got: {list(df.columns)}")

    # Attach validation labels and spend (same logic as train_baseline)
    plis = pd.read_csv(plis_path, sep="\t", low_memory=False)
    plis["orderdate"] = pd.to_datetime(plis["orderdate"], format="%Y-%m-%d")
    plis["legal_entity_id"] = plis["legal_entity_id"].astype(str)
    plis["eclass"] = plis["eclass"].astype(str).str.strip().replace("nan", "")
    plis = plis[(plis["orderdate"] >= val_start) & (plis["orderdate"] <= val_end)]
    q = pd.to_numeric(plis["quantityvalue"], errors="coerce").fillna(0)
    v = pd.to_numeric(plis["vk_per_item"], errors="coerce").fillna(0)
    plis["_spend"] = q * v
    plis = plis[plis["eclass"] != ""]

    val_agg = (
        plis.groupby(["legal_entity_id", "eclass"])
        .agg(n_orders_val=("_spend", "count"), s_val=("_spend", "sum"))
        .reset_index()
    )
    df = df.merge(val_agg, on=["legal_entity_id", "eclass"], how="left")
    df["n_orders_val"] = df["n_orders_val"].fillna(0).astype(int)
    df["s_val"] = df["s_val"].fillna(0)
    df["label"] = (df["n_orders_val"] >= args.n_min_label).astype(int)

    # Prepare feature matrix: fill NaN, handle categorical
    use_cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    X_cols = [c for c in FEATURE_COLS if c in df.columns]
    if use_cat:
        X_cols = X_cols + use_cat

    X = df[X_cols].copy()
    for c in X_cols:
        if c in use_cat:
            X[c] = X[c].fillna("").astype("category")
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)

    y_label = df["label"].values
    s_val = df["s_val"].values

    # Stage A: recurrence classifier
    default_clf = {"objective": "binary", "verbosity": -1, "num_leaves": 31, "n_estimators": 100}
    clf_params = default_clf.copy()
    if args.lgb_params_classifier:
        clf_params.update(json.loads(args.lgb_params_classifier))
    model_a = lgb.LGBMClassifier(**clf_params)
    model_a.fit(X, y_label, categorical_feature=use_cat if use_cat else "auto")
    p_recur = model_a.predict_proba(X)[:, 1]

    # Stage B: conditional value regressor (positive examples only)
    pos = y_label == 1
    if pos.sum() < 10:
        v_hat = pd.Series(0.0, index=df.index)
        v_hat.loc[pos] = df.loc[pos, "s_val"]
    else:
        X_pos = X.loc[pos]
        y_val_pos = df.loc[pos, "s_val"].values
        default_reg = {"objective": "regression", "verbosity": -1, "num_leaves": 31, "n_estimators": 100}
        reg_params = default_reg.copy()
        if args.lgb_params_regressor:
            reg_params.update(json.loads(args.lgb_params_regressor))
        model_b = lgb.LGBMRegressor(**reg_params)
        model_b.fit(X_pos, y_val_pos, categorical_feature=use_cat if use_cat else "auto")
        v_hat = pd.Series(model_b.predict(X), index=df.index)
        v_hat = v_hat.clip(lower=0)

    # Per-month scale for savings
    scale = 1.0 / args.val_months
    df["score_base"] = (
        p_recur * v_hat * scale * args.savings_rate - args.fixed_fee_eur
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} scored rows to {out_path}")

    # Illustrative offline score (EU > 0, top-15 per buyer)
    df_sorted = df[df["score_base"] > 0].sort_values(
        ["legal_entity_id", "score_base"], ascending=[True, False]
    )
    selected = df_sorted.groupby("legal_entity_id").head(15)
    savings = (selected["s_val"] * scale * args.savings_rate * selected["label"]).sum()
    fees = len(selected) * args.fixed_fee_eur
    score_per_month = savings - fees
    print(
        f"Offline score (per-month norm, EU>0 top-15): {score_per_month:.2f} "
        f"(savings={savings:.2f} fees={fees:.2f})"
    )


if __name__ == "__main__":
    main()
