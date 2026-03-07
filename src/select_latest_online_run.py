"""
Select the latest online archived run per level (by created_at) across all approaches.

Reads run_index_{approach}_level{level}.csv under data/15_scores/online, loads each run's
score_summary.csv, picks the run with the most recent created_at, and writes a single-row
latest_run_summary.csv for that level.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


def _parse_index_row(row: dict[str, str], index_path: Path, root: Path) -> dict | None:
    run_dir_str = row["run_dir"].strip()
    run_dir = root / run_dir_str.lstrip("/")
    summary_path = run_dir / "score_summary.csv"
    meta_path = run_dir / "metadata.json"
    if not summary_path.is_file():
        return None
    with summary_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        score_row = next(reader, None)
    if not score_row:
        return None
    created_at = row.get("created_at", "").strip()
    metadata = {}
    if meta_path.is_file():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    stem = index_path.stem
    approach = stem.replace("run_index_", "").rsplit("_level", 1)[0] if "_level" in stem else "unknown"
    return {
        "created_at": created_at,
        "approach": approach,
        "run_id": row["run_id"].strip(),
        "run_dir": run_dir_str,
        "commit_sha": row.get("commit_sha", "").strip(),
        "branch": row.get("branch", "").strip(),
        "dirty": row.get("dirty", "").strip().lower() == "true",
        "score_row": score_row,
        "metadata_json": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select latest online archived run per level by created_at and write latest_run_summary.csv."
    )
    parser.add_argument("--scores-dir", required=True, dest="scores_dir", help="e.g. data/15_scores/online")
    parser.add_argument("--level", required=True, type=int, help="Level (1 or 2)")
    parser.add_argument("--output-csv", required=True, dest="output_csv", help="Output path for latest_run_summary.csv")
    args = parser.parse_args()

    scores_dir = Path(args.scores_dir)
    level = args.level
    output_path = Path(args.output_csv)
    root = Path.cwd()

    index_glob = f"run_index_*_level{level}.csv"
    index_files = sorted(scores_dir.glob(index_glob))
    candidates: list[dict] = []

    for index_path in index_files:
        with index_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rec = _parse_index_row(row, index_path, root)
                if rec is not None:
                    candidates.append(rec)

    if not candidates:
        raise SystemExit(
            f"No archived runs found for level {level} under {scores_dir}. "
            f"Looked for {index_glob}. Run archive_score_run for at least one approach/level first."
        )

    def _created_ts(r):
        s = r.get("created_at") or ""
        if not s:
            return 0.0
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
        except (ValueError, TypeError):
            return 0.0

    candidates.sort(key=lambda r: (-_created_ts(r), r["approach"], r["run_id"]))
    latest = candidates[0]
    score_row = latest["score_row"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "total_score", "total_savings", "total_fees", "num_hits", "num_predictions",
        "spend_capture_rate", "total_ground_spend",
        "approach", "run_id", "run_dir", "commit_sha", "branch", "dirty", "created_at",
    ]
    row = {
        **score_row,
        "approach": latest["approach"],
        "run_id": latest["run_id"],
        "run_dir": latest["run_dir"],
        "commit_sha": latest["commit_sha"],
        "branch": latest["branch"],
        "dirty": str(latest["dirty"]).lower(),
        "created_at": latest["created_at"],
    }
    with output_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerow(row)

    print(f"Latest run for level {level}: {latest['approach']} {latest['run_id']} (created_at={latest['created_at']}) -> {output_path}")


if __name__ == "__main__":
    main()
