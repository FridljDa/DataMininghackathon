"""Unit tests for submission tuning diagnostics: run parsing, param effects, submission shape, missing data."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import pytest

from src.analyze_submission_tuning import (
    _best_run_identifiers,
    _gather_runs,
    _load_customer_task_map,
    _load_run,
    _param_effects_df,
    _run_metrics_rows,
    _submission_shape,
)


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=30, check=False)


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# _load_run / _gather_runs
# ---------------------------------------------------------------------------


def test_load_run_valid_dir(tmp_path: Path) -> None:
    """Valid run dir with score_summary_live.csv and metadata.json is loaded."""
    run_dir = tmp_path / "20250307_120000_abc1234"
    run_dir.mkdir()
    (run_dir / "score_summary_live.csv").write_text(
        "total_score,total_savings,total_fees,num_hits,num_predictions,spend_capture_rate,total_ground_spend\n"
        "1000.5,2000.0,999.5,50,80,0.25,10000.0\n",
        encoding="utf-8",
    )
    (run_dir / "metadata.json").write_text(
        json.dumps({
            "approach": "lgbm_two_stage",
            "level": 1,
            "created_at": "2025-03-07T12:00:00Z",
            "config": {"top_k_per_buyer": 30, "lookback_months": 60},
        }),
        encoding="utf-8",
    )
    rec = _load_run(run_dir)
    assert rec is not None
    assert rec["run_id"] == "20250307_120000_abc1234"
    assert rec["score_row"]["total_score"] == "1000.5"
    assert rec["meta"]["approach"] == "lgbm_two_stage"
    assert rec["meta"]["config"]["top_k_per_buyer"] == 30


def test_load_run_invalid_dir(tmp_path: Path) -> None:
    """Dir without score summary returns None."""
    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()
    assert _load_run(run_dir) is None
    (run_dir / "metadata.json").write_text("{}", encoding="utf-8")
    assert _load_run(run_dir) is None


def test_load_run_fallback_score_summary(tmp_path: Path) -> None:
    """Fallback to score_summary.csv when score_summary_live.csv missing."""
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    (run_dir / "score_summary.csv").write_text(
        "total_score,total_savings,total_fees,num_hits,num_predictions,spend_capture_rate,total_ground_spend\n"
        "500.0,1000.0,500.0,20,40,0.1,5000.0\n",
        encoding="utf-8",
    )
    rec = _load_run(run_dir)
    assert rec is not None
    assert rec["score_row"]["total_score"] == "500.0"


def test_gather_runs_mixed_valid_invalid(tmp_path: Path) -> None:
    """_gather_runs returns only valid run dirs."""
    (tmp_path / "valid1").mkdir()
    (tmp_path / "valid1" / "score_summary_live.csv").write_text(
        "total_score,total_savings,total_fees,num_hits,num_predictions,spend_capture_rate,total_ground_spend\n"
        "1.0,2.0,1.0,0,1,0.0,0.0\n",
        encoding="utf-8",
    )
    (tmp_path / "valid1" / "metadata.json").write_text("{}", encoding="utf-8")
    (tmp_path / "invalid_no_csv").mkdir()
    (tmp_path / "invalid_no_csv" / "metadata.json").write_text("{}", encoding="utf-8")
    (tmp_path / "file_not_dir").write_text("x", encoding="utf-8")
    runs = _gather_runs(tmp_path)
    assert len(runs) == 1
    assert runs[0]["run_id"] == "valid1"


def test_gather_runs_empty_or_missing_dir(tmp_path: Path) -> None:
    """_gather_runs on non-dir or empty dir returns empty list."""
    assert _gather_runs(Path("/nonexistent/path")) == []
    assert _gather_runs(tmp_path) == []


# ---------------------------------------------------------------------------
# _run_metrics_rows / _param_effects_df
# ---------------------------------------------------------------------------


def test_run_metrics_rows_flattens_config() -> None:
    """Run metrics rows include flattened config and guardrails."""
    runs = [
        {
            "run_id": "r1",
            "score_row": {"total_score": "100", "total_savings": "200", "total_fees": "100",
                          "num_hits": "10", "num_predictions": "20", "spend_capture_rate": "0.5", "total_ground_spend": "1000"},
            "meta": {"approach": "lgbm", "created_at": "2025-01-01T00:00:00Z",
                     "config": {"top_k_per_buyer": 30, "lookback_months": 60,
                                "guardrails": {"min_orders": 2, "high_spend": 200}}},
            "run_dir": Path("."),
        },
    ]
    rows = _run_metrics_rows(runs)
    assert len(rows) == 1
    assert rows[0]["run_id"] == "r1"
    assert rows[0]["total_score"] == "100"
    assert rows[0]["config_top_k_per_buyer"] == 30
    assert rows[0]["config_lookback_months"] == 60
    assert rows[0]["guardrail_min_orders"] == 2
    assert rows[0]["guardrail_high_spend"] == 200


def test_param_effects_df_aggregates_by_param() -> None:
    """Param effects group by param value and aggregate score."""
    import pandas as pd
    run_metrics_df = pd.DataFrame([
        {"run_id": "a", "total_score": "100", "config_top_k_per_buyer": 30},
        {"run_id": "b", "total_score": "200", "config_top_k_per_buyer": 30},
        {"run_id": "c", "total_score": "150", "config_top_k_per_buyer": 60},
    ])
    effects = _param_effects_df(run_metrics_df)
    assert not effects.empty
    assert "param" in effects.columns and "value" in effects.columns
    assert "mean_score" in effects.columns and "max_score" in effects.columns
    top_k30 = effects[(effects["param"] == "config_top_k_per_buyer") & (effects["value"] == 30)]
    assert len(top_k30) == 1
    assert top_k30.iloc[0]["count"] == 2
    assert top_k30.iloc[0]["mean_score"] == 150.0
    assert top_k30.iloc[0]["max_score"] == 200.0


def test_param_effects_df_empty() -> None:
    """Empty run metrics yields empty param effects with expected columns."""
    import pandas as pd
    effects = _param_effects_df(pd.DataFrame())
    assert effects.empty
    assert "param" in effects.columns and "value" in effects.columns


# ---------------------------------------------------------------------------
# _submission_shape
# ---------------------------------------------------------------------------


def test_submission_shape_minimal_csv(tmp_path: Path) -> None:
    """Submission shape from minimal valid submission CSV."""
    csv_path = tmp_path / "submission.csv"
    csv_path.write_text(
        "legal_entity_id,cluster\n"
        "b1,e1\n"
        "b1,e2\n"
        "b2,e1\n",
        encoding="utf-8",
    )
    shape = _submission_shape(csv_path, "test_approach")
    assert shape["approach"] == "test_approach"
    assert shape["n_predictions"] == 3
    assert shape["n_buyers"] == 2
    assert shape["avg_predictions_per_buyer"] == 1.5
    assert shape["median_predictions_per_buyer"] == 1.5
    assert shape["duplicate_rate"] == 0.0
    # one cluster e1 has 2 predictions, e2 has 1 -> top cluster share 2/3
    assert shape["top_cluster_share"] == pytest.approx(2 / 3)
    # without task_map, warm/cold are None
    assert shape["n_warm_buyers_submitted"] is None
    assert shape["n_cold_buyers_submitted"] is None
    assert shape["warm_buyer_share"] is None
    assert shape["cold_buyer_share"] is None


def test_submission_shape_duplicate_rate(tmp_path: Path) -> None:
    """Duplicate (buyer, cluster) rows increase duplicate_rate."""
    csv_path = tmp_path / "submission.csv"
    csv_path.write_text(
        "legal_entity_id,cluster\n"
        "b1,e1\n"
        "b1,e1\n"
        "b1,e2\n",
        encoding="utf-8",
    )
    shape = _submission_shape(csv_path, "x")
    assert shape["n_predictions"] == 2  # deduplicated
    assert shape["duplicate_rate"] == pytest.approx(1 / 3)


def test_submission_shape_missing_file() -> None:
    """Missing submission path yields None metrics."""
    shape = _submission_shape(Path("/nonexistent/submission.csv"), "x")
    assert shape["n_predictions"] is None
    assert shape["n_buyers"] is None


def test_submission_shape_missing_columns(tmp_path: Path) -> None:
    """Submission without legal_entity_id or cluster still reports n_predictions."""
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")
    shape = _submission_shape(csv_path, "x")
    assert shape["n_predictions"] == 1
    assert shape["n_buyers"] is None


def test_submission_shape_warm_cold_counts(tmp_path: Path) -> None:
    """With task_map, shape reports n_warm_buyers_submitted, n_cold_buyers_submitted, shares."""
    csv_path = tmp_path / "submission.csv"
    csv_path.write_text(
        "legal_entity_id,cluster\n"
        "w1,e1\n"
        "w1,e2\n"
        "c1,e1\n",
        encoding="utf-8",
    )
    task_map = {"w1": "warm", "c1": "cold"}
    shape = _submission_shape(csv_path, "test_approach", task_map=task_map)
    assert shape["n_buyers"] == 2
    assert shape["n_warm_buyers_submitted"] == 1
    assert shape["n_cold_buyers_submitted"] == 1
    assert shape["n_unknown_task_buyers_submitted"] == 0
    assert shape["warm_buyer_share"] == pytest.approx(0.5)
    assert shape["cold_buyer_share"] == pytest.approx(0.5)


def test_submission_shape_unknown_task_buyers(tmp_path: Path) -> None:
    """Buyers not in task_map or with other task are counted as unknown."""
    csv_path = tmp_path / "submission.csv"
    csv_path.write_text(
        "legal_entity_id,cluster\n"
        "w1,e1\n"
        "unknown_buyer,e1\n",
        encoding="utf-8",
    )
    task_map = {"w1": "warm"}
    shape = _submission_shape(csv_path, "x", task_map=task_map)
    assert shape["n_buyers"] == 2
    assert shape["n_warm_buyers_submitted"] == 1
    assert shape["n_cold_buyers_submitted"] == 0
    assert shape["n_unknown_task_buyers_submitted"] == 1
    assert shape["warm_buyer_share"] == pytest.approx(0.5)
    assert shape["cold_buyer_share"] == pytest.approx(0.0)


def test_load_customer_task_map(tmp_path: Path) -> None:
    """_load_customer_task_map returns warm/cold/unknown from customer_test TSV."""
    customer_tsv = tmp_path / "customer_test.csv"
    customer_tsv.write_text(
        "legal_entity_id\testimated_number_employees\tnace_code\ttask\n"
        "1\t100\t861\tcold start\n"
        "2\t200\t3511\tpredict future\n"
        "3\t300\t1089\tother\n",
        encoding="utf-8",
    )
    task_map = _load_customer_task_map(customer_tsv)
    assert task_map["1"] == "cold"
    assert task_map["2"] == "warm"
    assert task_map["3"] == "unknown"


# ---------------------------------------------------------------------------
# _best_run_identifiers
# ---------------------------------------------------------------------------


def test_best_run_identifiers_from_dir(tmp_path: Path) -> None:
    """Read total_score and created_at from best_run dir."""
    (tmp_path / "score_summary_live.csv").write_text(
        "total_score,total_savings,total_fees,num_hits,num_predictions,spend_capture_rate,total_ground_spend\n"
        "999.0,1500.0,501.0,40,60,0.2,8000.0\n",
        encoding="utf-8",
    )
    (tmp_path / "metadata.json").write_text(
        json.dumps({"run_id": "20250307_abc", "created_at": "2025-03-07T14:00:00Z"}),
        encoding="utf-8",
    )
    run_id, score, created = _best_run_identifiers(tmp_path)
    assert run_id == "20250307_abc"
    assert score == 999.0
    assert created == "2025-03-07T14:00:00Z"


def test_best_run_identifiers_missing_dir() -> None:
    """Non-dir or missing path returns (None, None, None)."""
    assert _best_run_identifiers(Path("/nonexistent")) == (None, None, None)


# ---------------------------------------------------------------------------
# main(): graceful behavior when 15_scores / 16_scores_best missing
# ---------------------------------------------------------------------------


def test_main_writes_placeholder_when_no_runs(tmp_path: Path, project_root: Path) -> None:
    """When runs_dir is empty/missing, script writes placeholder CSVs and plots without failing."""
    out_dir = tmp_path / "out"
    runs_dir = tmp_path / "runs_level1"
    runs_dir.mkdir(parents=True)
    best_run_dir = tmp_path / "best_run"
    best_run_dir.mkdir(parents=True)
    customer_test = tmp_path / "customer_test.csv"
    customer_test.write_text("legal_entity_id\ttask\n1\tcold start\n", encoding="utf-8")
    result = _run(
        [
            "uv", "run", "src/analyze_submission_tuning.py",
            "--level", "1",
            "--output-dir", str(out_dir),
            "--runs-dir", str(runs_dir),
            "--best-run-dir", str(best_run_dir),
            "--customer-test", str(customer_test),
            "--submissions",
        ],
        cwd=project_root,
    )
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    assert (out_dir / "run_metrics_level1.csv").exists()
    assert (out_dir / "param_effects_level1.csv").exists()
    assert (out_dir / "current_submission_shape_level1.csv").exists()
    assert (out_dir / "score_vs_predictions_level1.png").exists()
    assert (out_dir / "score_vs_capture_level1.png").exists()
    assert (out_dir / "run_timeline_level1.png").exists()
    assert (out_dir / "submission_size_vs_score_level1.png").exists()
    content = (out_dir / "run_metrics_level1.csv").read_text()
    assert "No archived runs" in content or "run_id" in content
