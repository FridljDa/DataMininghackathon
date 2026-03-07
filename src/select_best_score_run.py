"""
Select the best online archived run per level by total_score and copy that run into 16_scores_best.

Scans data/15_scores/online/runs/level{level}/ for run directories (each with score_summary_live.csv
or score_summary.csv and metadata.json), ranks by total_score descending (tie-break: created_at desc,
then run_id), and copies the best run directory into the given best_run dir (overwrites existing).
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path


def _load_run_candidate(run_dir: Path) -> dict | None:
    """Load one run dir: score row and metadata. Return None if not a valid run."""
    live_path = run_dir / "score_summary_live.csv"
    summary_path = run_dir / "score_summary.csv"
    path_to_use = live_path if live_path.is_file() else summary_path
    meta_path = run_dir / "metadata.json"
    if not path_to_use.is_file():
        return None
    with path_to_use.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        score_row = next(reader, None)
    if not score_row or "total_score" not in score_row:
        return None
    try:
        total_score = float(score_row["total_score"])
    except (ValueError, TypeError):
        return None
    created_at = ""
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            created_at = (meta.get("created_at") or "").strip()
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "total_score": total_score,
        "created_at": created_at,
        "run_id": run_dir.name,
        "run_dir": run_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select best online archived run per level by total_score and copy into best_run dir."
    )
    parser.add_argument("--scores-dir", required=True, dest="scores_dir", help="e.g. data/15_scores/online")
    parser.add_argument("--level", required=True, type=int, help="Level (1 or 2)")
    parser.add_argument("--best-run-dir", required=True, dest="best_run_dir", help="Output directory for best run copy (e.g. data/16_scores_best/online/level1/best_run)")
    args = parser.parse_args()

    scores_dir = Path(args.scores_dir)
    level = args.level
    best_run_dir = Path(args.best_run_dir)

    runs_dir = scores_dir / "runs" / f"level{level}"
    if not runs_dir.is_dir():
        raise SystemExit(
            f"No runs directory for level {level}: {runs_dir}. "
            "Run archive_score_run for at least one approach/level first."
        )

    candidates: list[dict] = []
    for run_path in runs_dir.iterdir():
        if not run_path.is_dir():
            continue
        rec = _load_run_candidate(run_path)
        if rec is not None:
            candidates.append(rec)

    if not candidates:
        raise SystemExit(
            f"No archived runs found for level {level} under {runs_dir} "
            "(no score_summary_live.csv or score_summary.csv with total_score). "
            "Run archive_score_run for at least one approach/level first."
        )

    def _created_ts(r: dict) -> float:
        s = r.get("created_at") or ""
        if not s:
            return 0.0
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
        except (ValueError, TypeError):
            return 0.0

    candidates.sort(key=lambda r: (-r["total_score"], -_created_ts(r), r["run_id"]))
    best = candidates[0]

    best_run_dir = best_run_dir.resolve()
    if best_run_dir.exists():
        shutil.rmtree(best_run_dir)
    best_run_dir.mkdir(parents=True, exist_ok=True)

    for item in best["run_dir"].iterdir():
        dest = best_run_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)

    print(f"Best run for level {level}: {best['run_id']} (total_score={best['total_score']}) -> {best_run_dir}")


if __name__ == "__main__":
    main()
