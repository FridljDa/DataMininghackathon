"""
Selection policy: threshold, evidence guardrails, per-buyer cap K.

Reads scores.parquet; keeps pairs with score_base > threshold that pass at least
one guardrail; caps at top K per buyer by score_base.
Outputs portfolio.parquet: Level 1 = (legal_entity_id, eclass); Level 2 = (legal_entity_id, eclass, manufacturer).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True, help="Path to scores.parquet.")
    parser.add_argument("--output", required=True, help="Path to output portfolio.parquet.")
    parser.add_argument(
        "--level",
        required=True,
        choices=("1", "2"),
        help="Level 1 or 2; level 2 portfolio includes manufacturer.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        dest="score_threshold",
        help="Keep only pairs with score_base > this (default: 0).",
    )
    parser.add_argument(
        "--min-orders-guardrail",
        type=int,
        default=3,
        dest="min_orders_guardrail",
        help="Pass guardrail if n_orders >= this (default: 3).",
    )
    parser.add_argument(
        "--min-months-guardrail",
        type=int,
        default=2,
        dest="min_months_guardrail",
        help="Pass guardrail if m_active >= this (default: 2).",
    )
    parser.add_argument(
        "--high-spend-guardrail",
        type=float,
        default=500.0,
        dest="high_spend_guardrail",
        help="Pass guardrail if historical_purchase_value_total >= this (default: 500).",
    )
    parser.add_argument(
        "--min-avg-monthly-spend",
        type=float,
        default=0.0,
        dest="min_avg_monthly_spend",
        help="Pass guardrail if avg_monthly_spend_buyer_tenure >= this when column present (0 = disabled; default: 0).",
    )
    parser.add_argument(
        "--top-k-per-buyer",
        type=int,
        default=15,
        dest="top_k_per_buyer",
        help="Max eclasses per buyer (default: 15).",
    )
    args = parser.parse_args()

    scores_path = Path(args.scores)
    out_path = Path(args.output)

    df = pd.read_parquet(scores_path)
    required = ["legal_entity_id", "eclass", "score_base", "n_orders", "m_active", "historical_purchase_value_total"]
    if args.level == "2":
        required.append("manufacturer")
    for col in required:
        if col not in df.columns:
            raise ValueError(f"scores must contain '{col}'. Got: {list(df.columns)}")

    # Threshold
    df = df[df["score_base"] > args.score_threshold].copy()

    # Guardrail: at least one of n_orders >= X, m_active >= Y, historical_purchase_value_total >= tau_high,
    # or (when present) avg_monthly_spend_buyer_tenure >= min_avg_monthly_spend for late joiners
    guard = (
        (df["n_orders"] >= args.min_orders_guardrail)
        | (df["m_active"] >= args.min_months_guardrail)
        | (df["historical_purchase_value_total"] >= args.high_spend_guardrail)
    )
    if args.min_avg_monthly_spend > 0 and "avg_monthly_spend_buyer_tenure" in df.columns:
        guard = guard | (df["avg_monthly_spend_buyer_tenure"].fillna(0) >= args.min_avg_monthly_spend)
    df = df[guard].copy()

    # Top K per buyer by score_base
    df = (
        df.sort_values(["legal_entity_id", "score_base"], ascending=[True, False])
        .groupby("legal_entity_id", as_index=False)
        .head(args.top_k_per_buyer)
    )

    portfolio_cols = ["legal_entity_id", "eclass", "manufacturer"] if args.level == "2" else ["legal_entity_id", "eclass"]
    portfolio = df[[c for c in portfolio_cols if c in df.columns]].drop_duplicates()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    portfolio.to_parquet(out_path, index=False)
    print(f"Wrote {len(portfolio)} portfolio rows to {out_path}")


if __name__ == "__main__":
    main()
