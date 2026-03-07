"""Unit tests for online score persistence: latest run, runs history, live submission history, best-run artifact copy."""

from __future__ import annotations

import csv
import subprocess
import tempfile
from pathlib import Path

import pytest


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=30)


@pytest.fixture
def project_root() -> Path:
    """Project root (parent of src/ and data/)."""
    return Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# select_latest_online_run
# ---------------------------------------------------------------------------


def test_select_latest_online_run_picks_newest_by_created_at(tmp_path: Path, project_root: Path) -> None:
    """Latest run per level is the one with the most recent created_at across all approaches."""
    level = 1
    older_ts = "2026-03-07T15:40:34Z"
    newer_ts = "2026-03-07T16:38:23Z"
    # Run dirs must be under project_root so that with cwd=project_root the script resolves run_dir
    with tempfile.TemporaryDirectory(dir=project_root) as td:
        run_base = Path(td)
        run_baseline = run_base / "baseline" / f"level{level}" / "20260307_163823_e02a8a2_dirty"
        run_baseline.mkdir(parents=True)
        (run_baseline / "score_summary.csv").write_text(
            "total_score,total_savings,total_fees,num_hits,num_predictions,spend_capture_rate,total_ground_spend\n"
            "-399450.0,0.0,399450.0,0,39945,0.0,10207465.32\n",
            encoding="utf-8",
        )
        (run_baseline / "metadata.json").write_text('{"commit":"abc","branch":"main","dirty":true,"created_at":"' + newer_ts + '"}', encoding="utf-8")

        run_lgbm = run_base / "lgbm_two_stage" / f"level{level}" / "20260307_154116_eec09f1_dirty"
        run_lgbm.mkdir(parents=True)
        (run_lgbm / "score_summary.csv").write_text(
            "total_score,total_savings,total_fees,num_hits,num_predictions,spend_capture_rate,total_ground_spend\n"
            "-42000.0,0.0,42000.0,0,4200,0.0,10608197.85\n",
            encoding="utf-8",
        )
        (run_lgbm / "metadata.json").write_text('{"commit":"def","branch":"cold","dirty":true,"created_at":"' + older_ts + '"}', encoding="utf-8")

        rel_baseline = str(run_baseline.relative_to(project_root))
        rel_lgbm = str(run_lgbm.relative_to(project_root))
        scores_dir = tmp_path / "online"
        scores_dir.mkdir(parents=True)
        (scores_dir / f"run_index_baseline_level{level}.csv").write_text(
            "run_id,commit_sha,branch,dirty,created_at,run_dir\n"
            f"20260307_163823_e02a8a2_dirty,e02a8a2,alex,true,{newer_ts},{rel_baseline}\n",
            encoding="utf-8",
        )
        (scores_dir / f"run_index_lgbm_two_stage_level{level}.csv").write_text(
            "run_id,commit_sha,branch,dirty,created_at,run_dir\n"
            f"20260307_154116_eec09f1_dirty,eec09f1,cold,true,{older_ts},{rel_lgbm}\n",
            encoding="utf-8",
        )

        out_csv = tmp_path / "latest_run_summary.csv"
        result = _run(
            [
                "uv", "run", "src/select_latest_online_run.py",
                "--scores-dir", str(scores_dir),
                "--level", str(level),
                "--output-csv", str(out_csv),
            ],
            cwd=project_root,
        )
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert out_csv.exists()
        rows = list(csv.DictReader(out_csv.read_text().splitlines()))
        assert len(rows) == 1
        assert rows[0]["approach"] == "baseline"
        assert rows[0]["created_at"] == newer_ts
        assert rows[0]["run_id"] == "20260307_163823_e02a8a2_dirty"


# ---------------------------------------------------------------------------
# select_best_score_run artifact copy
# ---------------------------------------------------------------------------


def test_select_best_score_run_copies_artifacts_to_best_dir(tmp_path: Path, project_root: Path) -> None:
    """Best run selection copies score_summary.csv and score_details.parquet into 16_scores_best/online/level{N}/."""
    level = 1
    with tempfile.TemporaryDirectory(dir=project_root) as td:
        run_base = Path(td)
        run_dir = run_base / "baseline" / f"level{level}" / "20260307_best_run"
        run_dir.mkdir(parents=True)
        (run_dir / "score_summary.csv").write_text(
            "total_score,total_savings,total_fees,num_hits,num_predictions,spend_capture_rate,total_ground_spend\n"
            "100.0,1000.0,900.0,10,100,0.5,2000.0\n",
            encoding="utf-8",
        )
        (run_dir / "score_details.parquet").write_bytes(b"parquet_magic_bytes_placeholder")
        (run_dir / "metadata.json").write_text('{"commit":"abc","branch":"main","dirty":false,"created_at":"2026-03-07T12:00:00Z"}', encoding="utf-8")

        scores_dir = tmp_path / "online"
        scores_dir.mkdir(parents=True)
        rel_run_dir = str(run_dir.relative_to(project_root))
        (scores_dir / f"run_index_baseline_level{level}.csv").write_text(
            "run_id,commit_sha,branch,dirty,created_at,run_dir\n"
            f"20260307_best_run,abc,main,false,2026-03-07T12:00:00Z,{rel_run_dir}\n",
            encoding="utf-8",
        )

        best_dir = tmp_path / "16_scores_best" / "online" / f"level{level}"
        result = _run(
            [
                "uv", "run", "src/select_best_score_run.py",
                "--scores-dir", str(scores_dir),
                "--level", str(level),
                "--summary-csv", str(best_dir / "best_run_summary.csv"),
                "--metadata-json", str(best_dir / "best_run_metadata.json"),
            ],
            cwd=project_root,
        )
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert (best_dir / "best_run_summary.csv").exists()
        assert (best_dir / "best_run_metadata.json").exists()
        assert (best_dir / "score_summary.csv").exists()
        assert (best_dir / "score_details.parquet").exists()
        assert (best_dir / "score_summary.csv").read_text() == (run_dir / "score_summary.csv").read_text()


# ---------------------------------------------------------------------------
# archive_score_run history append
# ---------------------------------------------------------------------------


def test_archive_score_run_appends_to_runs_history(tmp_path: Path, project_root: Path) -> None:
    """When --history-csv, --approach, --level are set, archive appends one row to runs_history.csv."""
    summary = tmp_path / "score_summary.csv"
    details = tmp_path / "score_details.parquet"
    summary.write_text(
        "total_score,total_savings,total_fees,num_hits,num_predictions,spend_capture_rate,total_ground_spend\n"
        "-100.0,0.0,100.0,0,10,0.0,5000.0\n",
        encoding="utf-8",
    )
    details.write_bytes(b"\x00" * 100)  # placeholder parquet

    runs_dir = tmp_path / "runs" / "baseline" / "level1"
    index_csv = tmp_path / "run_index_baseline_level1.csv"
    history_csv = tmp_path / "history" / "level1" / "runs_history.csv"

    result = _run(
        [
            "uv", "run", "src/archive_score_run.py",
            "--summary", str(summary),
            "--details", str(details),
            "--runs-dir", str(runs_dir),
            "--index-csv", str(index_csv),
            "--history-csv", str(history_csv),
            "--approach", "baseline",
            "--level", "1",
        ],
        cwd=project_root,
    )
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    assert history_csv.exists()
    rows = list(csv.DictReader(history_csv.read_text().splitlines()))
    assert len(rows) == 1
    assert rows[0]["approach"] == "baseline"
    assert rows[0]["level"] == "1"
    assert "run_id" in rows[0]
    assert "total_score" in rows[0]
    assert rows[0]["total_score"] == "-100.0"


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
