"""
Hybrid portfolio: take primary (e.g. lgbm_two_stage) ranked rows first, then backfill
from secondary (e.g. phase3_repro) until target_per_buyer is reached per buyer.

Reads two portfolio.parquet files and optionally secondary scores.parquet for ranking.
Outputs a single portfolio.parquet with the same schema (level 1 or 2).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", required=True, help="Path to primary portfolio.parquet (e.g. lgbm_two_stage).")
    parser.add_argument("--secondary", required=True, help="Path to secondary portfolio.parquet (e.g. phase3_repro).")
    parser.add_argument(
        "--scores-secondary",
        default=None,
        dest="scores_secondary",
        help="Optional path to secondary scores.parquet to rank backfill by score_base.",
    )
    parser.add_argument("--output", required=True, help="Path to output portfolio.parquet.")
    parser.add_argument(
        "--level",
        required=True,
        choices=("1", "2"),
        help="Level 1 or 2; level 2 includes manufacturer.",
    )
    parser.add_argument(
        "--target-per-buyer",
        type=int,
        default=400,
        dest="target_per_buyer",
        help="Max rows per buyer in output (default: 400).",
    )
    args = parser.parse_args()

    key_cols = ["legal_entity_id", "eclass", "manufacturer"] if args.level == "2" else ["legal_entity_id", "eclass"]
    primary = pd.read_parquet(args.primary)
    secondary = pd.read_parquet(args.secondary)

    for c in key_cols:
        if c not in primary.columns or c not in secondary.columns:
            raise ValueError(f"Both portfolios must have columns {key_cols}. Primary: {list(primary.columns)}, secondary: {list(secondary.columns)}")

    primary = primary[key_cols].drop_duplicates()
    secondary = secondary[key_cols].drop_duplicates()

    # Rank secondary by score_base if provided
    if args.scores_secondary and Path(args.scores_secondary).is_file():
        scores = pd.read_parquet(args.scores_secondary)
        if "score_base" in scores.columns and all(c in scores.columns for c in key_cols):
            secondary = secondary.merge(
                scores[key_cols + ["score_base"]].drop_duplicates(key_cols),
                on=key_cols,
                how="left",
            )
            secondary = (
                secondary.sort_values(["legal_entity_id", "score_base"], ascending=[True, False])
                .drop_duplicates(key_cols, keep="first")
                [key_cols]
            )
        else:
            secondary = secondary.drop_duplicates(key_cols)

    rows = []
    for buyer, grp_prim in primary.groupby("legal_entity_id"):
        taken = set(grp_prim.apply(lambda r: tuple(r[c] for c in key_cols), axis=1))
        n = len(taken)
        for _, r in grp_prim.iterrows():
            rows.append(r[key_cols].to_dict())
        if n >= args.target_per_buyer:
            continue
        sec_buyer = secondary[secondary["legal_entity_id"] == buyer]
        for _, r in sec_buyer.iterrows():
            if n >= args.target_per_buyer:
                break
            t = tuple(r[c] for c in key_cols)
            if t not in taken:
                taken.add(t)
                rows.append(r[key_cols].to_dict())
                n += 1

    out = pd.DataFrame(rows).drop_duplicates(key_cols)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(f"Wrote {len(out)} portfolio rows to {args.output}")


if __name__ == "__main__":
    main()
