"""
Merge multiple portfolio.parquet files in priority order: take rows from the first
portfolio per buyer, then backfill from the second (optionally ranked by its scores),
then the third, etc., until target_per_buyer is reached per buyer.

Accepts ordered --portfolios and optional --scores (one per portfolio; use empty or
omit to skip ranking for that source). Outputs a single portfolio.parquet with the
same schema (level 1 or 2).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--portfolios",
        nargs="+",
        required=True,
        help="Ordered paths to portfolio.parquet files (first = highest priority).",
    )
    parser.add_argument(
        "--scores",
        nargs="*",
        default=None,
        help="Optional paths to scores.parquet, one per portfolio (same order). Omit or use empty to skip ranking.",
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

    if len(args.portfolios) < 2:
        raise SystemExit("At least two --portfolios are required.")

    scores_list = args.scores or []
    if scores_list and len(scores_list) != len(args.portfolios):
        raise SystemExit(
            f"Number of --scores ({len(scores_list)}) must match number of --portfolios ({len(args.portfolios)})."
        )
    if not scores_list:
        scores_list = [None] * len(args.portfolios)
    elif len(scores_list) < len(args.portfolios):
        scores_list = list(scores_list) + [None] * (len(args.portfolios) - len(scores_list))

    key_cols = ["legal_entity_id", "eclass", "manufacturer"] if args.level == "2" else ["legal_entity_id", "eclass"]

    def load_portfolio(path: str) -> pd.DataFrame:
        df = pd.read_parquet(path)
        for c in key_cols:
            if c not in df.columns:
                raise ValueError(f"Portfolio {path} missing columns {key_cols}; has {list(df.columns)}")
        return df[key_cols].drop_duplicates()

    def rank_by_scores(df: pd.DataFrame, scores_path: str | None) -> pd.DataFrame:
        if not scores_path or not Path(scores_path).is_file():
            return df.drop_duplicates(key_cols)
        scores = pd.read_parquet(scores_path)
        if "score_base" not in scores.columns or not all(c in scores.columns for c in key_cols):
            return df.drop_duplicates(key_cols)
        df = df.merge(
            scores[key_cols + ["score_base"]].drop_duplicates(key_cols),
            on=key_cols,
            how="left",
        )
        df = (
            df.sort_values(["legal_entity_id", "score_base"], ascending=[True, False])
            .drop_duplicates(key_cols, keep="first")
            [key_cols]
        )
        return df

    portfolios = [load_portfolio(p) for p in args.portfolios]
    for i, sp in enumerate(scores_list):
        if sp:
            portfolios[i] = rank_by_scores(portfolios[i], sp)

    all_buyers = set()
    for p in portfolios:
        all_buyers |= set(p["legal_entity_id"].unique())

    rows = []
    for buyer in sorted(all_buyers):
        taken: set[tuple] = set()
        for port in portfolios:
            if len(taken) >= args.target_per_buyer:
                break
            grp = port[port["legal_entity_id"] == buyer]
            for _, r in grp.iterrows():
                if len(taken) >= args.target_per_buyer:
                    break
                t = tuple(r[c] for c in key_cols)
                if t not in taken:
                    taken.add(t)
                    rows.append(r[key_cols].to_dict())

    out = pd.DataFrame(rows).drop_duplicates(key_cols)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(f"Wrote {len(out)} portfolio rows to {args.output}")


if __name__ == "__main__":
    main()
