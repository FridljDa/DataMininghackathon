"""
Archive live score summary into a timestamp+sha run folder.
Writes score_summary_live.csv and metadata.json (with approach, level, and config snapshot).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        cmd,
        cwd=cwd or Path.cwd(),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    return (result.stdout or "").strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Archive live score summary into a timestamp+sha run folder (score_summary_live.csv, metadata.json)."
    )
    parser.add_argument("--live-summary", required=True, dest="live_summary", help="Path to live score summary CSV from portal.")
    parser.add_argument("--runs-dir", required=True, dest="runs_dir", help="Base directory for run folders (e.g. data/15_scores/online/runs/level1).")
    parser.add_argument("--approach", default=None, help="Modelling approach used (e.g. lgbm_two_stage, baseline).")
    parser.add_argument("--level", type=int, default=None, help="Level (1 or 2).")
    parser.add_argument("--train-end", dest="train_end", default=None, help="modelling.windows.train_end.")
    parser.add_argument("--lookback-months", type=int, dest="lookback_months", default=None, help="modelling.candidates.lookback_months.")
    parser.add_argument("--score-threshold", type=float, dest="score_threshold", default=None, help="modelling.selection.score_threshold.")
    parser.add_argument("--top-k-per-buyer", type=int, dest="top_k_per_buyer", default=None, help="modelling.selection.top_k_per_buyer.")
    parser.add_argument("--min-orders", type=int, dest="min_orders", default=None, help="modelling.selection.guardrails.min_orders.")
    parser.add_argument("--min-months", type=int, dest="min_months", default=None, help="modelling.selection.guardrails.min_months.")
    parser.add_argument("--high-spend", type=float, dest="high_spend", default=None, help="modelling.selection.guardrails.high_spend.")
    parser.add_argument("--min-avg-monthly-spend", type=float, dest="min_avg_monthly_spend", default=None, help="modelling.selection.guardrails.min_avg_monthly_spend.")
    parser.add_argument("--cold-start-top-k", type=int, dest="cold_start_top_k", default=None, help="submission.cold_start_top_k.")
    parser.add_argument("--selected-features", dest="selected_features", default=None, help="Comma-separated modelling.features.selected list.")
    parser.add_argument("--run-id", dest="run_id", default=None, help="Pre-generated run id; if set, use it instead of minting timestamp+sha.")
    args = parser.parse_args()

    root = Path.cwd()
    live_summary_path = Path(args.live_summary)
    runs_dir = Path(args.runs_dir)

    if not live_summary_path.is_file():
        raise FileNotFoundError(f"Live summary not found: {live_summary_path}")

    if args.run_id:
        run_id = args.run_id
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        try:
            short_sha = _run(["git", "rev-parse", "--short", "HEAD"], cwd=root)
        except (OSError, subprocess.TimeoutExpired):
            short_sha = "norepo"
        if not short_sha:
            short_sha = "norepo"
        run_id = f"{ts}_{short_sha}"

    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(live_summary_path, run_dir / "score_summary_live.csv")

    try:
        commit = _run(["git", "rev-parse", "HEAD"], cwd=root)
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root)
        porcelain = _run(["git", "status", "--porcelain"], cwd=root)
        dirty = bool(porcelain)
    except (OSError, subprocess.TimeoutExpired):
        commit = ""
        branch = ""
        dirty = False

    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    metadata = {
        "commit": commit,
        "branch": branch,
        "dirty": dirty,
        "created_at": created,
    }
    if args.approach is not None:
        metadata["approach"] = args.approach
    if args.level is not None:
        metadata["level"] = args.level

    config: dict = {}
    if args.train_end is not None:
        config["train_end"] = args.train_end
    if args.lookback_months is not None:
        config["lookback_months"] = args.lookback_months
    if args.score_threshold is not None:
        config["score_threshold"] = args.score_threshold
    if args.top_k_per_buyer is not None:
        config["top_k_per_buyer"] = args.top_k_per_buyer
    if args.min_orders is not None or args.min_months is not None or args.high_spend is not None or args.min_avg_monthly_spend is not None:
        config["guardrails"] = {}
        if args.min_orders is not None:
            config["guardrails"]["min_orders"] = args.min_orders
        if args.min_months is not None:
            config["guardrails"]["min_months"] = args.min_months
        if args.high_spend is not None:
            config["guardrails"]["high_spend"] = args.high_spend
        if args.min_avg_monthly_spend is not None:
            config["guardrails"]["min_avg_monthly_spend"] = args.min_avg_monthly_spend
    if args.cold_start_top_k is not None:
        config["cold_start_top_k"] = args.cold_start_top_k
    if args.selected_features is not None:
        config["selected_features"] = [s.strip() for s in args.selected_features.split(",") if s.strip()]
    if config:
        metadata["config"] = config

    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    if args.run_id:
        (run_dir / ".archived").write_text(run_id)

    print(f"Archived run {run_id} -> {run_dir}")


if __name__ == "__main__":
    main()
