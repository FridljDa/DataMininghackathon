"""
Archive current score summary and details into a timestamp+commit run folder.
Writes metadata.json (commit, branch, dirty, created_at) and appends to run_index.csv.
For online mode, can also append one row to a per-level runs_history.csv.
"""

from __future__ import annotations

import argparse
import csv
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
        description="Archive score outputs into a timestamp+commit run folder with metadata and index."
    )
    parser.add_argument("--summary", required=True, help="Path to score_summary.csv.")
    parser.add_argument("--details", required=True, help="Path to score_details.parquet.")
    parser.add_argument("--runs-dir", required=True, dest="runs_dir", help="Base directory for run folders (e.g. data/15_scores/offline/runs).")
    parser.add_argument("--index-csv", required=True, dest="index_csv", help="Path to run_index.csv (created/appended).")
    parser.add_argument("--history-csv", default=None, dest="history_csv", help="If set (online), append one row to this per-level runs_history.csv.")
    parser.add_argument("--approach", default=None, help="Required if --history-csv is set.")
    parser.add_argument("--level", type=int, default=None, help="Required if --history-csv is set.")
    args = parser.parse_args()

    root = Path.cwd()
    summary_path = Path(args.summary)
    details_path = Path(args.details)
    runs_dir = Path(args.runs_dir)
    index_path = Path(args.index_csv)

    if not summary_path.is_file():
        raise FileNotFoundError(f"Summary not found: {summary_path}")
    if not details_path.is_file():
        raise FileNotFoundError(f"Details not found: {details_path}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    try:
        short_sha = _run(["git", "rev-parse", "--short", "HEAD"], cwd=root)
    except (OSError, subprocess.TimeoutExpired):
        short_sha = "norepo"
    if not short_sha:
        short_sha = "norepo"

    run_id = f"{ts}_{short_sha}"
    try:
        porcelain = _run(["git", "status", "--porcelain"], cwd=root)
        if porcelain:
            run_id = f"{run_id}_dirty"
    except (OSError, subprocess.TimeoutExpired):
        pass

    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(summary_path, run_dir / "score_summary.csv")
    shutil.copy2(details_path, run_dir / "score_details.parquet")

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
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    index_path.parent.mkdir(parents=True, exist_ok=True)
    header = "run_id,commit_sha,branch,dirty,created_at,run_dir"
    row = f"{run_id},{commit},{branch},{str(dirty).lower()},{created},{run_dir}"
    if not index_path.exists():
        index_path.write_text(header + "\n", encoding="utf-8")
    with index_path.open("a", encoding="utf-8") as f:
        f.write(row + "\n")

    if args.history_csv and args.approach is not None and args.level is not None:
        history_path = Path(args.history_csv)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            score_row = next(reader)
        history_headers = [
            "run_id", "approach", "level", "created_at", "commit_sha", "branch", "dirty", "run_dir",
            "total_score", "total_savings", "total_fees", "num_hits", "num_predictions",
            "spend_capture_rate", "total_ground_spend",
        ]
        history_row = {
            "run_id": run_id,
            "approach": args.approach,
            "level": args.level,
            "created_at": created,
            "commit_sha": commit,
            "branch": branch,
            "dirty": str(dirty).lower(),
            "run_dir": str(run_dir),
            **score_row,
        }
        file_exists = history_path.exists()
        with history_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=history_headers, extrasaction="ignore")
            if not file_exists:
                w.writeheader()
            w.writerow(history_row)
    elif args.history_csv:
        raise ValueError("--history-csv requires --approach and --level")

    print(f"Archived run {run_id} -> {run_dir}")


if __name__ == "__main__":
    main()
