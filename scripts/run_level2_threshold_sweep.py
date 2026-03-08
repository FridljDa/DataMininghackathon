"""
Run level2-only negative-threshold sweep: selection-only experiments for lgbm_two_stage
with score_threshold in (-0.01, -0.03, -0.05, -0.10, -0.20), top_k=400, guardrails 0/1/0.

Each trial patches config modelling.selection.by_level.2, runs Snakemake from select_portfolio
through merged submission, and copies the submission to data/17_submission_tuning/sweep_level2/
so all trials are kept. Restores original by_level.2 at the end.

Usage (from repo root):
  uv run scripts/run_level2_threshold_sweep.py --dry-run   # print trials only
  uv run scripts/run_level2_threshold_sweep.py             # run all 5 trials, save submissions
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

LEVEL2_THRESHOLDS = [-0.01, -0.03, -0.05, -0.10, -0.20]
TOP_K = 400
GUARDRAILS_L2 = {"min_orders": 0, "min_months": 1, "high_spend": 0, "min_avg_monthly_spend": 0}


def _threshold_filename(thresh: float) -> str:
    """e.g. -0.01 -> threshold_m0.01"""
    s = str(thresh).replace("-", "m").replace(".", "p")
    return f"submission_threshold_{s}.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--mode", type=str, default="online", choices=("online", "offline"))
    parser.add_argument("--approach", type=str, default="lgbm_two_stage")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    if not config_path.is_file():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    try:
        import yaml
    except ImportError:
        print("PyYAML required: pip install pyyaml", file=sys.stderr)
        sys.exit(1)

    with config_path.open() as f:
        config = yaml.safe_load(f)

    sel = config.get("modelling", {}).get("selection", {})
    by_level = sel.get("by_level", {})
    original_level2 = dict(by_level.get("2", {})) if isinstance(by_level.get("2"), dict) else {}

    if args.dry_run:
        print("Level2 threshold sweep trials:")
        for thresh in LEVEL2_THRESHOLDS:
            print(f"  score_threshold={thresh} top_k_per_buyer={TOP_K} guardrails={GUARDRAILS_L2}")
        print(f"Submissions will be written to data/17_submission_tuning/sweep_level2/")
        return

    out_dir = Path("data/17_submission_tuning/sweep_level2")
    out_dir.mkdir(parents=True, exist_ok=True)

    submission_path = Path(f"data/14_submission/{args.mode}/{args.approach}/level2/submission.csv")
    portfolio_path = Path(f"data/13_portfolio/{args.mode}/{args.approach}/level2/portfolio.parquet")

    for thresh in LEVEL2_THRESHOLDS:
        sel = config.setdefault("modelling", {}).setdefault("selection", {})
        by_level = sel.setdefault("by_level", {})
        by_level["2"] = {
            "score_threshold": float(thresh),
            "top_k_per_buyer": TOP_K,
            "guardrails": GUARDRAILS_L2,
        }
        with config_path.open("w") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        print(f"Level2 threshold={thresh} ...")
        cmd = ["snakemake", str(portfolio_path), str(submission_path), "--configfile", str(config_path)]
        rc = subprocess.run(cmd, cwd=Path.cwd()).returncode
        if rc != 0:
            print(f"Snakemake failed for threshold={thresh}", file=sys.stderr)
            _restore(config_path, config, original_level2)
            sys.exit(rc)

        dest = out_dir / _threshold_filename(thresh)
        if submission_path.is_file():
            shutil.copy2(submission_path, dest)
            print(f"  -> {dest}")
        else:
            print(f"  WARNING: {submission_path} not found", file=sys.stderr)

    _restore(config_path, config, original_level2)
    print("Done. Restored config by_level.2.")


def _restore(config_path: Path, config: dict, original_level2: dict) -> None:
    sel = config.get("modelling", {}).get("selection", {})
    by_level = sel.get("by_level", {})
    if original_level2:
        by_level["2"] = original_level2
    else:
        by_level.pop("2", None)
    with config_path.open("w") as f:
        import yaml
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


if __name__ == "__main__":
    main()
