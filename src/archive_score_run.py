"""
Archive current score summary and details into a timestamp+commit run folder.
Writes metadata.json (commit, branch, dirty, created_at) and appends to run_index.csv.
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
        description="Archive score outputs into a timestamp+commit run folder with metadata and index."
    )
    parser.add_argument("--summary", required=True, help="Path to score_summary.csv.")
    parser.add_argument("--details", required=True, help="Path to score_details.parquet.")
    parser.add_argument("--runs-dir", required=True, dest="runs_dir", help="Base directory for run folders (e.g. data/15_scores/offline/runs).")
    parser.add_argument("--index-csv", required=True, dest="index_csv", help="Path to run_index.csv (created/appended).")
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

    print(f"Archived run {run_id} -> {run_dir}")


if __name__ == "__main__":
    main()
