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
    """Archive writes only runs_dir/<timestamp_sha>/ with score_summary_live.csv and metadata.json (no score_summary.csv, no score_details.parquet)."""
    live_summary = tmp_path / "score_summary_live.csv"
    live_summary.write_text(
        "total_score,total_savings,total_fees,num_hits,num_predictions,spend_capture_rate,total_ground_spend,submission_id\n"
        "-100.0,0.0,100.0,0,10,0.0,5000.0,sub-123\n",
        encoding="utf-8",
    )

    runs_dir = tmp_path / "runs" / "level1"

    result = _run(
        [
            "uv", "run", "src/archive_score_run.py",
            "--live-summary", str(live_summary),
            "--runs-dir", str(runs_dir),
            "--approach", "lgbm_two_stage",
            "--level", "1",
            "--train-end", "2025-12-31",
            "--top-k-per-buyer", "15",
            "--selected-features", "n_orders,CV_gap",
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

    assert (run_dir / "score_summary_live.csv").exists()
    assert (run_dir / "score_summary_live.csv").read_text() == live_summary.read_text()
    assert (run_dir / "metadata.json").exists()
    meta = json.loads((run_dir / "metadata.json").read_text())
    assert "commit" in meta
    assert "branch" in meta
    assert "dirty" in meta
    assert "created_at" in meta
    assert meta.get("approach") == "lgbm_two_stage"
    assert meta.get("level") == 1
    assert "config" in meta
    assert meta["config"].get("train_end") == "2025-12-31"
    assert meta["config"].get("top_k_per_buyer") == 15
    assert meta["config"].get("selected_features") == ["n_orders", "CV_gap"]

    # Only these two files; no offline score artifacts
    assert not (run_dir / "score_summary.csv").exists()
    assert not (run_dir / "score_details.parquet").exists()


def test_archive_score_run_uses_upstream_metadata_when_provided(tmp_path: Path, project_root: Path) -> None:
    """When --metadata is provided, archive copies upstream metadata and adds archived_at."""
    live_summary = tmp_path / "score_summary_live.csv"
    live_summary.write_text(
        "total_score,total_savings,total_fees,num_hits,num_predictions,spend_capture_rate,total_ground_spend,submission_id\n"
        "100.0,200.0,100.0,5,10,0.5,1000.0,sub-456\n",
        encoding="utf-8",
    )
    upstream_meta = tmp_path / "metadata.json"
    upstream_meta.write_text(
        json.dumps({
            "commit": "abc123",
            "branch": "main",
            "dirty": False,
            "created_at": "2026-03-08T07:00:00Z",
            "approach": "lgbm_two_stage",
            "level": 1,
            "config": {"score_threshold": -0.05, "top_k_per_buyer": 400},
        }, indent=2),
        encoding="utf-8",
    )
    runs_dir = tmp_path / "runs" / "level1"

    result = _run(
        [
            "uv", "run", "src/archive_score_run.py",
            "--live-summary", str(live_summary),
            "--runs-dir", str(runs_dir),
            "--run-id", "20260308_072447_ec1a9a4_0",
            "--metadata", str(upstream_meta),
        ],
        cwd=project_root,
    )
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"

    run_dir = runs_dir / "20260308_072447_ec1a9a4_0"
    assert run_dir.is_dir()
    assert (run_dir / "metadata.json").exists()
    meta = json.loads((run_dir / "metadata.json").read_text())
    assert meta.get("commit") == "abc123"
    assert meta.get("approach") == "lgbm_two_stage"
    assert meta.get("level") == 1
    assert meta.get("config", {}).get("score_threshold") == -0.05
    assert "archived_at" in meta


# ---------------------------------------------------------------------------
# run_metadata (shared metadata builder for run-scoped artifacts)
# ---------------------------------------------------------------------------


def test_run_metadata_build_produces_expected_keys() -> None:
    """build_run_metadata produces commit, branch, dirty, created_at, and optional config."""
    from src.run_metadata import build_run_metadata

    meta = build_run_metadata(
        run_id="20260308_abc_0",
        approach="lgbm_two_stage",
        level=1,
        train_end="2025-12-31",
        score_threshold=-0.05,
        top_k_per_buyer=400,
        selected_features=["n_orders", "CV_gap"],
    )
    assert "commit" in meta
    assert "branch" in meta
    assert "dirty" in meta
    assert "created_at" in meta
    assert meta.get("approach") == "lgbm_two_stage"
    assert meta.get("level") == 1
    assert "config" in meta
    assert meta["config"].get("train_end") == "2025-12-31"
    assert meta["config"].get("score_threshold") == -0.05
    assert meta["config"].get("selected_features") == ["n_orders", "CV_gap"]


def test_write_run_metadata_script_writes_file(project_root: Path, tmp_path: Path) -> None:
    """write_run_metadata.py writes a valid metadata.json to --output."""
    out = tmp_path / "run_dir" / "metadata.json"
    result = _run(
        [
            "uv", "run", "src/write_run_metadata.py",
            "--output", str(out),
            "--run-id", "20260308_abc_1",
            "--approach", "lgbm_two_stage",
            "--level", "1",
            "--train-end", "2025-12-31",
            "--score-threshold", "-0.05",
            "--selected-features", "n_orders,CV_gap",
        ],
        cwd=project_root,
    )
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    assert out.exists()
    meta = json.loads(out.read_text())
    assert meta.get("approach") == "lgbm_two_stage"
    assert meta.get("level") == 1
    assert meta.get("config", {}).get("score_threshold") == -0.05


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
