"""
Single entrypoint for all modelling approaches. Dispatches by --approach.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src is on path when run as script (uv run src/modelling/run.py)
_here = Path(__file__).resolve().parent
_src = _here.parent
if _src not in [Path(p).resolve() for p in sys.path]:
    sys.path.insert(0, str(_src))

import pandas as pd

from modelling.approaches import get_approach
from modelling.common.labels import attach_validation_labels


def _print_offline_score(
    df: pd.DataFrame,
    val_months: float,
    savings_rate: float,
    fixed_fee_eur: float,
    top_k: int = 15,
) -> None:
    scale = 1.0 / val_months
    df_sorted = df[df["score_base"] > 0].sort_values(
        ["legal_entity_id", "score_base"], ascending=[True, False]
    )
    selected = df_sorted.groupby("legal_entity_id").head(top_k)
    savings = (selected["s_val"] * scale * savings_rate * selected["label"]).sum()
    fees = len(selected) * fixed_fee_eur
    score_per_month = savings - fees
    print(
        f"Offline score (per-month norm, score_base>0 top-{top_k}): {score_per_month:.2f} "
        f"(savings={savings:.2f} fees={fees:.2f})"
    )


def run_main() -> None:
    parser = argparse.ArgumentParser(description="Run a modelling approach and write scores.parquet.")
    parser.add_argument("--approach", required=True, help="Approach name: baseline | lgbm_two_stage | pass_through")
    parser.add_argument("--candidates", required=True, help="Path to features/candidates parquet.")
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
        default=1,
        dest="n_min_label",
        help="Min orders in val period for positive label.",
    )
    parser.add_argument("--val-months", type=float, default=6.0, dest="val_months")
    parser.add_argument("--savings-rate", type=float, default=0.10, dest="savings_rate")
    parser.add_argument("--fixed-fee-eur", type=float, default=10.0, dest="fixed_fee_eur")
    # Baseline
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    # LGBM
    parser.add_argument("--lgb-params-classifier", type=str, default="", dest="lgb_params_classifier")
    parser.add_argument("--lgb-params-regressor", type=str, default="", dest="lgb_params_regressor")
    # Phase 3 repro (sparse-history gate)
    parser.add_argument("--eta", type=int, default=2, dest="eta")
    parser.add_argument("--tau", type=float, default=100.0, dest="tau")
    parser.add_argument("--sparse-eta-multiplier", type=int, default=3, dest="sparse_eta_multiplier")
    parser.add_argument("--sparse-tau-multiplier", type=float, default=2.0, dest="sparse_tau_multiplier")
    args = parser.parse_args()

    candidates_path = Path(args.candidates)
    plis_path = Path(args.plis)
    out_path = Path(args.output)
    val_start = pd.Timestamp(args.val_start)
    val_end = pd.Timestamp(args.val_end)

    df = pd.read_parquet(candidates_path)
    df = attach_validation_labels(
        df, plis_path, val_start, val_end, args.n_min_label
    )

    approach = get_approach(args.approach)
    params = {
        "val_months": args.val_months,
        "savings_rate": args.savings_rate,
        "fixed_fee_eur": args.fixed_fee_eur,
        "alpha": args.alpha,
        "beta": args.beta,
        "gamma": args.gamma,
        "lgb_params_classifier": args.lgb_params_classifier,
        "lgb_params_regressor": args.lgb_params_regressor,
        "eta": args.eta,
        "tau": args.tau,
        "sparse_eta_multiplier": args.sparse_eta_multiplier,
        "sparse_tau_multiplier": args.sparse_tau_multiplier,
    }
    df = approach.run(df, **params)

    for col in ("legal_entity_id", "eclass", "score_base", "n_orders", "m_active", "historical_purchase_value_total"):
        if col not in df.columns:
            raise ValueError(f"Output must contain '{col}'. Got: {list(df.columns)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} scored rows to {out_path}")

    _print_offline_score(
        df, args.val_months, args.savings_rate, args.fixed_fee_eur, top_k=15
    )


if __name__ == "__main__":
    run_main()
