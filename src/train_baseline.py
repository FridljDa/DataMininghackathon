"""
Baseline scorer for Level 1: score_base(b,e) = α·m_active + β·√s_total − γ·δ_recency.

Reads candidates.parquet and plis (split) to attach validation labels and validation
spend. Outputs scores.parquet with score_base, label, s_val. Prints per-month
normalised offline euro score for a simple policy (score_base > threshold, top-K).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


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
    parser.add_argument("--alpha", type=float, default=1.0, help="Coefficient for m_active.")
    parser.add_argument("--beta", type=float, default=1.0, help="Coefficient for sqrt(s_total).")
    parser.add_argument("--gamma", type=float, default=0.5, help="Coefficient for delta_recency.")
    parser.add_argument(
        "--savings-rate",
        type=float,
        default=0.10,
        dest="savings_rate",
        help="For offline score print (default: 0.10).",
    )
    parser.add_argument(
        "--fixed-fee-eur",
        type=float,
        default=10.0,
        dest="fixed_fee_eur",
        help="For offline score print (default: 10).",
    )
    parser.add_argument(
        "--val-months",
        type=float,
        default=6.0,
        dest="val_months",
        help="Validation window length in months for per-month normalisation (default: 6).",
    )
    args = parser.parse_args()

    candidates_path = Path(args.candidates)
    plis_path = Path(args.plis)
    out_path = Path(args.output)
    val_start = pd.Timestamp(args.val_start)
    val_end = pd.Timestamp(args.val_end)

    df = pd.read_parquet(candidates_path)
    for col in ("legal_entity_id", "eclass", "m_active", "s_total_sqrt", "delta_recency"):
        if col not in df.columns:
            raise ValueError(f"candidates must contain '{col}'. Got: {list(df.columns)}")

    # Baseline score
    df["score_base"] = (
        args.alpha * df["m_active"].fillna(0)
        + args.beta * df["s_total_sqrt"].fillna(0)
        - args.gamma * df["delta_recency"].fillna(0)
    )

    # Validation-period spend and order count from plis
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
    df = df.merge(
        val_agg,
        on=["legal_entity_id", "eclass"],
        how="left",
    )
    df["n_orders_val"] = df["n_orders_val"].fillna(0).astype(int)
    df["s_val"] = df["s_val"].fillna(0)
    df["label"] = (df["n_orders_val"] >= args.n_min_label).astype(int)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} scored rows to {out_path}")

    # Per-month normalised offline score (illustrative: include if score_base > 0, cap 15 per buyer)
    scale = 1.0 / args.val_months
    df_sorted = df[df["score_base"] > 0].sort_values(
        ["legal_entity_id", "score_base"], ascending=[True, False]
    )
    selected = df_sorted.groupby("legal_entity_id").head(15)
    savings = (selected["s_val"] * scale * args.savings_rate * selected["label"]).sum()
    fees = len(selected) * args.fixed_fee_eur
    score_per_month = savings - fees
    print(
        f"Offline score (per-month norm, score_base>0 top-15): {score_per_month:.2f} "
        f"(savings={savings:.2f} fees={fees:.2f})"
    )


if __name__ == "__main__":
    main()
