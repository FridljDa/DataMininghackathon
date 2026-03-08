"""
Run one or all trials of the threshold/top_k sweep by patching config.yaml
and running Snakemake from select_portfolio through merged submission.

Sweep matrix: score_threshold in (0.0, -0.01, -0.05, -0.10, -0.20),
top_k_per_buyer in (150, 300, 400), level in (1, 2).
One trial = one (level, threshold, top_k) combination.

Usage:
  uv run scripts/run_threshold_sweep.py --dry-run           # list trials
  uv run scripts/run_threshold_sweep.py --trial-index 0     # run one trial
  uv run scripts/run_threshold_sweep.py                    # run trial 0 (default)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Sweep matrix
THRESHOLDS = [0.0, -0.01, -0.05, -0.10, -0.20]
TOP_K_VALUES = [150, 300, 400]
LEVELS = [1, 2]


def _trials():
    for level in LEVELS:
        for thresh in THRESHOLDS:
            for top_k in TOP_K_VALUES:
                yield (level, thresh, top_k)


def _trial_list():
    return list(_trials())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--trial-index",
        type=int,
        default=0,
        help="Trial index (0-based). Default 0.",
    )
    parser.add_argument(
        "--approach",
        type=str,
        default="lgbm_two_stage",
        help="Approach to build (default: lgbm_two_stage)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="online",
        choices=("online", "offline"),
        help="Mode (default: online)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the list of trials and exit",
    )
    args = parser.parse_args()

    trials = _trial_list()
    n = len(trials)
    if args.dry_run:
        print(f"Total trials: {n}")
        for i, (level, thresh, top_k) in enumerate(trials):
            print(f"  {i}: level={level} score_threshold={thresh} top_k_per_buyer={top_k}")
        return

    if args.trial_index < 0 or args.trial_index >= n:
        print(f"Trial index must be 0..{n - 1}", file=sys.stderr)
        sys.exit(1)

    level, score_threshold, top_k_per_buyer = trials[args.trial_index]
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

    sel = config.setdefault("modelling", {}).setdefault("selection", {})
    by_level = sel.setdefault("by_level", {})
    by_level[str(level)] = {
        "score_threshold": float(score_threshold),
        "top_k_per_buyer": int(top_k_per_buyer),
        "guardrails": sel.get("guardrails") or {},
    }

    with config_path.open("w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    # Run from select_portfolio through merge for this mode/approach/level
    data_dir = Path("data")
    portfolio = data_dir / "13_portfolio" / args.mode / args.approach / f"level{level}" / "portfolio.parquet"
    submission = data_dir / "14_submission" / args.mode / args.approach / f"level{level}" / "submission.csv"
    cmd = [
        "snakemake",
        str(portfolio),
        str(submission),
        "--configfile",
        str(config_path),
    ]
    print(f"Trial {args.trial_index}: level={level} threshold={score_threshold} top_k={top_k_per_buyer}")
    print(" ".join(cmd))
    rc = subprocess.run(cmd, cwd=Path.cwd()).returncode
    sys.exit(rc)


if __name__ == "__main__":
    main()
