"""
Select the best online archived run per level by total_score and write to 16_scores_best.

Reads run_index_{approach}_level{level}.csv under data/15_scores/online, loads each run's
score_summary.csv and metadata, ranks by total_score descending (tie-break: created_at desc,
then approach/run_id), and writes the best run's summary + metadata to the given outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def _parse_index_row(row: dict[str, str], index_path: Path, root: Path) -> dict:
    run_dir_str = row["run_dir"].strip()
    run_dir = root / run_dir_str.lstrip("/")
    # Prefer live (portal) score for online runs; fall back to local score_summary
    live_path = run_dir / "score_summary_live.csv"
    summary_path = run_dir / "score_summary.csv"
    path_to_use = live_path if live_path.is_file() else summary_path
    meta_path = run_dir / "metadata.json"
    if not path_to_use.is_file():
        return None
    # Parse score summary (single row)
    with path_to_use.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        score_row = next(reader, None)
    if not score_row:
        return None
    total_score = float(score_row["total_score"])
    created_at = row.get("created_at", "").strip()
    metadata = {}
    if meta_path.is_file():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    # Derive approach from index path: run_index_{approach}_level{N}.csv
    stem = index_path.stem  # e.g. run_index_baseline_level1
    approach = stem.replace("run_index_", "").rsplit("_level", 1)[0] if "_level" in stem else "unknown"
    return {
        "total_score": total_score,
        "created_at": created_at,
        "approach": approach,
        "run_id": row["run_id"].strip(),
        "run_dir": run_dir,
        "run_dir_str": run_dir_str,
        "commit_sha": row.get("commit_sha", "").strip(),
        "branch": row.get("branch", "").strip(),
        "dirty": row.get("dirty", "").strip().lower() == "true",
        "source_index_csv": str(index_path),
        "score_row": score_row,
        "metadata_json": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select best online archived run per level by total_score and write to 16_scores_best."
    )
    parser.add_argument("--scores-dir", required=True, dest="scores_dir", help="e.g. data/15_scores/online")
    parser.add_argument("--level", required=True, type=int, help="Level (1 or 2)")
    parser.add_argument("--summary-csv", required=True, dest="summary_csv", help="Output path for best_run_summary.csv")
    parser.add_argument("--metadata-json", required=True, dest="metadata_json", help="Output path for best_run_metadata.json")
    args = parser.parse_args()

    scores_dir = Path(args.scores_dir)
    level = args.level
    summary_out = Path(args.summary_csv)
    metadata_out = Path(args.metadata_json)

    index_glob = f"run_index_*_level{level}.csv"
    index_files = sorted(scores_dir.glob(index_glob))
    candidates: list[dict] = []

    root = Path.cwd()
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

    # Sort: total_score desc, then created_at desc, then approach, run_id
    def _created_ts(r):
        s = r.get("created_at") or ""
        if not s:
            return 0.0
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
        except (ValueError, TypeError):
            return 0.0

    def key(r):
        return (-r["total_score"], -_created_ts(r), r["approach"], r["run_id"])

    candidates.sort(key=key)
    best = candidates[0]
    score_row = best["score_row"]

    best_dir = summary_out.parent
    best_dir.mkdir(parents=True, exist_ok=True)

    # best_run_summary.csv: score metrics + run identity
    summary_headers = list(score_row.keys()) + ["approach", "run_id", "run_dir", "commit_sha", "branch", "dirty", "created_at"]
    summary_row = {**score_row, "approach": best["approach"], "run_id": best["run_id"], "run_dir": best["run_dir_str"], "commit_sha": best["commit_sha"], "branch": best["branch"], "dirty": str(best["dirty"]).lower(), "created_at": best["created_at"]}
    with summary_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_headers)
        w.writeheader()
        w.writerow(summary_row)

    # best_run_metadata.json: full metadata for reproducibility (live CSV may have empty cells)
    def _float(v, default=0.0):
        if v is None or v == "":
            return default
        try:
            x = float(v)
            return default if x != x else x  # NaN
        except (ValueError, TypeError):
            return default

    def _int(v, default=0):
        if v is None or v == "":
            return default
        try:
            return int(float(v))
        except (ValueError, TypeError):
            return default

    selected_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    best_metadata = {
        "mode": "online",
        "level": level,
        "approach": best["approach"],
        "run_id": best["run_id"],
        "run_dir": best["run_dir_str"],
        "commit_sha": best["commit_sha"],
        "branch": best["branch"],
        "dirty": best["dirty"],
        "created_at": best["created_at"],
        "total_score": best["total_score"],
        "total_savings": _float(score_row.get("total_savings")),
        "total_fees": _float(score_row.get("total_fees")),
        "num_hits": _int(score_row.get("num_hits")),
        "num_predictions": _int(score_row.get("num_predictions")),
        "spend_capture_rate": _float(score_row.get("spend_capture_rate")),
        "total_ground_spend": _float(score_row.get("total_ground_spend")),
        "source_index_csv": best["source_index_csv"],
        "selected_at_utc": selected_at,
        "run_metadata": best["metadata_json"],
    }
    metadata_out.write_text(json.dumps(best_metadata, indent=2), encoding="utf-8")

    # Copy best run artifacts into 16_scores_best for reproducibility
    run_dir = best["run_dir"]
    for name in ("score_summary.csv", "score_summary_live.csv", "score_details.parquet"):
        src = run_dir / name
        if src.is_file():
            shutil.copy2(src, best_dir / name)

    print(f"Best run for level {level}: {best['approach']} {best['run_id']} (total_score={best['total_score']}) -> {best_dir}")


if __name__ == "__main__":
    main()
