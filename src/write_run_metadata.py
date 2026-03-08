"""
Write run-scoped metadata.json beside pipeline artifacts (e.g. data/12_predictions/.../run_id/).

Used by the run-scoped Snakemake rules to create metadata at the first step (training)
and consumed by the archive step when copying to data/15_scores. Schema matches
archive_score_run metadata (commit, branch, dirty, created_at, approach, level, config).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from run_metadata import build_run_metadata, write_run_metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write metadata.json for a run-scoped pipeline artifact (same schema as archive)."
    )
    parser.add_argument("--output", required=True, help="Path to output metadata.json.")
    parser.add_argument("--run-id", required=True, dest="run_id", help="Run id (e.g. timestamp_sha_index).")
    parser.add_argument("--approach", required=True, help="Modelling approach (e.g. lgbm_two_stage).")
    parser.add_argument("--level", type=int, required=True, help="Level (1 or 2).")
    parser.add_argument("--train-end", dest="train_end", default=None, help="modelling.windows.train_end.")
    parser.add_argument("--lookback-months", type=int, dest="lookback_months", default=None)
    parser.add_argument("--score-threshold", type=float, dest="score_threshold", default=None)
    parser.add_argument("--top-k-per-buyer", type=int, dest="top_k_per_buyer", default=None)
    parser.add_argument("--min-orders", type=int, dest="min_orders", default=None)
    parser.add_argument("--min-months", type=int, dest="min_months", default=None)
    parser.add_argument("--high-spend", type=float, dest="high_spend", default=None)
    parser.add_argument("--min-avg-monthly-spend", type=float, dest="min_avg_monthly_spend", default=None)
    parser.add_argument("--cold-start-top-k", type=int, dest="cold_start_top_k", default=None)
    parser.add_argument(
        "--selected-features",
        dest="selected_features",
        default=None,
        help="Comma-separated modelling.features.selected list.",
    )
    args = parser.parse_args()

    selected_features: list[str] | None = None
    if args.selected_features is not None:
        selected_features = [s.strip() for s in args.selected_features.split(",") if s.strip()]

    metadata = build_run_metadata(
        run_id=args.run_id,
        approach=args.approach,
        level=args.level,
        train_end=args.train_end,
        lookback_months=args.lookback_months,
        score_threshold=args.score_threshold,
        top_k_per_buyer=args.top_k_per_buyer,
        min_orders=args.min_orders,
        min_months=args.min_months,
        high_spend=args.high_spend,
        min_avg_monthly_spend=args.min_avg_monthly_spend,
        cold_start_top_k=args.cold_start_top_k,
        selected_features=selected_features,
    )
    out_path = Path(args.output)
    write_run_metadata(out_path, metadata)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
