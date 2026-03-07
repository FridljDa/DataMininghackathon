"""Unit tests for online score persistence: archive run folder, live submission history helper."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import pytest


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=30, check=False)


@pytest.fixture
def project_root() -> Path:
    """Project root (parent of src/ and data/)."""
    return Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# archive_score_run
# ---------------------------------------------------------------------------


def test_archive_score_run_writes_flattened_run_folder_only(tmp_path: Path, project_root: Path) -> None:
    """Archive writes only runs_dir/<timestamp_sha>/ with score_summary.csv, score_details.parquet, metadata.json (no index/history, no _dirty)."""
    summary = tmp_path / "score_summary.csv"
    details = tmp_path / "score_details.parquet"
    summary.write_text(
        "total_score,total_savings,total_fees,num_hits,num_predictions,spend_capture_rate,total_ground_spend\n"
        "-100.0,0.0,100.0,0,10,0.0,5000.0\n",
        encoding="utf-8",
    )
    details.write_bytes(b"\x00" * 100)  # placeholder parquet

    runs_dir = tmp_path / "runs" / "level1"

    result = _run(
        [
            "uv", "run", "src/archive_score_run.py",
            "--summary", str(summary),
            "--details", str(details),
            "--runs-dir", str(runs_dir),
        ],
        cwd=project_root,
    )
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"

    # One run folder: runs/level1/<YYYYMMDD_HHMMSS_shortsha> (no _dirty)
    run_subdirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    assert len(run_subdirs) == 1
    run_dir = run_subdirs[0]
    run_id = run_dir.name
    assert "_dirty" not in run_id
    assert run_id.count("_") >= 2  # timestamp_sha

    assert (run_dir / "score_summary.csv").exists()
    assert (run_dir / "score_summary.csv").read_text() == summary.read_text()
    assert (run_dir / "score_details.parquet").exists()
    assert (run_dir / "metadata.json").exists()
    meta = json.loads((run_dir / "metadata.json").read_text())
    assert "commit" in meta
    assert "branch" in meta
    assert "dirty" in meta
    assert "created_at" in meta

    # No index or history files
    assert not (tmp_path / "run_index_baseline_level1.csv").exists()
    assert not (tmp_path / "history").exists()


# ---------------------------------------------------------------------------
# submit.py _append_live_submission_history
# ---------------------------------------------------------------------------


def test_append_live_submission_history_appends_one_row(tmp_path: Path) -> None:
    """_append_live_submission_history creates or appends one row to submissions_live_history.csv."""
    from src.submit import _append_live_submission_history

    history_path = tmp_path / "submissions_live_history.csv"
    row = {
        "total_score": 17746.61,
        "total_savings": 417196.61,
        "total_fees": 399450.0,
        "num_hits": 9432,
        "num_predictions": None,
        "spend_capture_rate": 0.1665,
        "total_ground_spend": None,
    }
    _append_live_submission_history(
        history_path,
        submission_id="3ace9b1b-ecd3-494c-9416-3940ca03d1b7",
        approach="baseline",
        level=2,
        submission_path=tmp_path / "submission.csv",
        row=row,
    )
    assert history_path.exists()
    rows = list(csv.DictReader(history_path.read_text().splitlines()))
    assert len(rows) == 1
    assert rows[0]["submission_id"] == "3ace9b1b-ecd3-494c-9416-3940ca03d1b7"
    assert rows[0]["approach"] == "baseline"
    assert rows[0]["level"] == "2"
    assert rows[0]["total_score"] == "17746.61"
    assert "submitted_at_utc" in rows[0]

    # Append second row
    _append_live_submission_history(
        history_path,
        submission_id="another-uuid",
        approach="lgbm_two_stage",
        level=2,
        submission_path=tmp_path / "sub2.csv",
        row={**row, "total_score": 20000.0},
    )
    rows2 = list(csv.DictReader(history_path.read_text().splitlines()))
    assert len(rows2) == 2
    assert rows2[1]["submission_id"] == "another-uuid"
    assert rows2[1]["approach"] == "lgbm_two_stage"
