"""
Live integration test: submit to https://unite-evaluator.vercel.app/challenges/2.

Runs when executing the integration test suite. Requires portal_credentials.team
and portal_credentials.password to be set in config.yaml.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


def _read_single_summary_row(summary_csv: Path) -> dict[str, str]:
    content = summary_csv.read_text(encoding="utf-8")
    rows = list(csv.DictReader(content.splitlines()))
    assert rows, f"{summary_csv.name} must contain at least one data row. Got:\n{content}"
    return rows[0]


def _assert_summary_metrics(summary_csv: Path) -> None:
    """Assert generic structural checks for a live summary (single prediction: fees=10, score=savings-fees)."""
    assert summary_csv.exists(), f"Live summary CSV not written: {summary_csv}"
    row = _read_single_summary_row(summary_csv)

    required_cols = ("total_score", "total_savings", "total_fees", "submission_id")
    for col in required_cols:
        assert col in row, f"{summary_csv.name} must contain {col}. Row: {row}"
        assert row[col] not in ("", None), f"{summary_csv.name} has empty {col}. Row: {row}"

    net_score = float(row["total_score"])
    savings = float(row["total_savings"])
    fees = float(row["total_fees"])

    # One submitted prediction should always incur a €10 fee.
    assert fees == 10.0, f"Expected total_fees=10.0 for one prediction. Got {fees}. Row: {row}"
    # Net Score is defined as Savings - Fees.
    assert net_score == pytest.approx(savings - fees), (
        f"Expected total_score == total_savings - total_fees. "
        f"Got score={net_score}, savings={savings}, fees={fees}. Row: {row}"
    )


# Expected portal metrics for tests/integration/resources/phase3_repro_level1_submission.csv (level 1).
PHASE3_FIXTURE_LEVEL1_EXPECTED = {
    "total_score": 816651.86,
    "total_savings": 982081.86,
    "total_fees": 165430.00,
    "num_hits": 11896,
    "spend_capture_rate": 0.3919,  # 39.19%
}


def _assert_phase3_level1_fixture_metrics(summary_csv: Path) -> None:
    """Assert exact portal metrics for the phase3_repro level-1 submission fixture."""
    assert summary_csv.exists(), f"Live summary CSV not written: {summary_csv}"
    row = _read_single_summary_row(summary_csv)

    assert float(row["total_score"]) == pytest.approx(PHASE3_FIXTURE_LEVEL1_EXPECTED["total_score"]), (
        f"total_score: got {row['total_score']}, expected {PHASE3_FIXTURE_LEVEL1_EXPECTED['total_score']}"
    )
    assert float(row["total_savings"]) == pytest.approx(PHASE3_FIXTURE_LEVEL1_EXPECTED["total_savings"]), (
        f"total_savings: got {row['total_savings']}, expected {PHASE3_FIXTURE_LEVEL1_EXPECTED['total_savings']}"
    )
    assert float(row["total_fees"]) == pytest.approx(PHASE3_FIXTURE_LEVEL1_EXPECTED["total_fees"]), (
        f"total_fees: got {row['total_fees']}, expected {PHASE3_FIXTURE_LEVEL1_EXPECTED['total_fees']}"
    )
    assert int(row["num_hits"]) == PHASE3_FIXTURE_LEVEL1_EXPECTED["num_hits"], (
        f"num_hits: got {row['num_hits']}, expected {PHASE3_FIXTURE_LEVEL1_EXPECTED['num_hits']}"
    )
    assert float(row["spend_capture_rate"]) == pytest.approx(PHASE3_FIXTURE_LEVEL1_EXPECTED["spend_capture_rate"]), (
        f"spend_capture_rate: got {row['spend_capture_rate']}, expected {PHASE3_FIXTURE_LEVEL1_EXPECTED['spend_capture_rate']}"
    )
    assert float(row["total_score"]) == pytest.approx(
        float(row["total_savings"]) - float(row["total_fees"])
    ), f"total_score should equal total_savings - total_fees. Row: {row}"


# Expected portal metrics for tests/integration/resources/lgbm_two_stage_submission.csv (level 2).
LGBM_FIXTURE_LEVEL2_EXPECTED = {
    "total_score": -41760.00,
    "total_savings": 0.00,
    "total_fees": 41760.00,
    "num_hits": 0,
    "spend_capture_rate": 0.00,  # 0.00%
}


def _assert_lgbm_fixture_metrics(summary_csv: Path) -> None:
    """Assert exact portal metrics for the lgbm_two_stage submission fixture at level 2."""
    assert summary_csv.exists(), f"Live summary CSV not written: {summary_csv}"
    row = _read_single_summary_row(summary_csv)

    assert float(row["total_score"]) == pytest.approx(LGBM_FIXTURE_LEVEL2_EXPECTED["total_score"]), (
        f"total_score: got {row['total_score']}, expected {LGBM_FIXTURE_LEVEL2_EXPECTED['total_score']}"
    )
    assert float(row["total_savings"]) == pytest.approx(LGBM_FIXTURE_LEVEL2_EXPECTED["total_savings"]), (
        f"total_savings: got {row['total_savings']}, expected {LGBM_FIXTURE_LEVEL2_EXPECTED['total_savings']}"
    )
    assert float(row["total_fees"]) == pytest.approx(LGBM_FIXTURE_LEVEL2_EXPECTED["total_fees"]), (
        f"total_fees: got {row['total_fees']}, expected {LGBM_FIXTURE_LEVEL2_EXPECTED['total_fees']}"
    )
    assert int(row["num_hits"]) == LGBM_FIXTURE_LEVEL2_EXPECTED["num_hits"], (
        f"num_hits: got {row['num_hits']}, expected {LGBM_FIXTURE_LEVEL2_EXPECTED['num_hits']}"
    )
    assert float(row["spend_capture_rate"]) == pytest.approx(LGBM_FIXTURE_LEVEL2_EXPECTED["spend_capture_rate"]), (
        f"spend_capture_rate: got {row['spend_capture_rate']}, expected {LGBM_FIXTURE_LEVEL2_EXPECTED['spend_capture_rate']}"
    )
    # Structural sanity: score = savings - fees
    assert float(row["total_score"]) == pytest.approx(
        float(row["total_savings"]) - float(row["total_fees"])
    ), f"total_score should equal total_savings - total_fees. Row: {row}"


def _run_live_submit(
    *,
    project_root: Path,
    submission_csv: Path,
    level: int,
    summary_csv: Path,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "-m",
        "src.submit",
        "--challenge",
        "2",
        "--file",
        str(submission_csv),
        "--level",
        str(level),
        "--summary-csv",
        str(summary_csv),
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    return subprocess.run(
        cmd,
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=360,
        check=False,
    )


@pytest.mark.integration
def test_submit_challenge_2_live(tmp_path: Path) -> None:
    """Run submit script for challenge 2 levels 1 and 2; validate live summary metrics."""
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "config.yaml"
    if not config_path.exists():
        pytest.fail("config.yaml not found at project root.")
    with config_path.open() as f:
        config = yaml.safe_load(f)
    creds = config.get("portal_credentials") or {}
    team = (creds.get("team") or "").strip()
    password = (creds.get("password") or "").strip()
    if not team or not password:
        pytest.fail(
            "portal_credentials.team and portal_credentials.password must be set in config.yaml "
            "to run the live submission test."
        )

    test_cases = [
        (1, "legal_entity_id,cluster\n1,30020903\n"),
        (2, "legal_entity_id,cluster\n1,30020903|Bissell\n"),
    ]
    for level, csv_content in test_cases:
        submission_csv = tmp_path / f"submission_level{level}.csv"
        submission_csv.write_text(csv_content, encoding="utf-8")
        summary_csv = tmp_path / f"score_summary_live_level{level}.csv"

        result = _run_live_submit(
            project_root=project_root,
            submission_csv=submission_csv,
            level=level,
            summary_csv=summary_csv,
        )

        assert result.returncode == 0, (
            f"submit (level {level}) exited with code {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert "Submission accepted" in result.stdout or "✓ Submission accepted" in result.stdout, (
            f"Expected submission acceptance message for level {level} in stdout. "
            f"Got:\n{result.stdout}"
        )
        _assert_summary_metrics(summary_csv)

    # Real submission fixture: phase3_repro at level 1; assert exact portal metrics.
    phase3_level1_csv = project_root / "tests" / "integration" / "resources" / "phase3_repro_level1_submission.csv"
    assert phase3_level1_csv.exists(), f"Fixture not found: {phase3_level1_csv}"
    summary_phase3_l1 = tmp_path / "score_summary_live_phase3_fixture_level1.csv"
    result_phase3_l1 = _run_live_submit(
        project_root=project_root,
        submission_csv=phase3_level1_csv,
        level=1,
        summary_csv=summary_phase3_l1,
    )
    assert result_phase3_l1.returncode == 0, (
        f"submit (phase3 fixture, level 1) exited with code {result_phase3_l1.returncode}\n"
        f"stdout:\n{result_phase3_l1.stdout}\nstderr:\n{result_phase3_l1.stderr}"
    )
    assert "Submission accepted" in result_phase3_l1.stdout or "✓ Submission accepted" in result_phase3_l1.stdout, (
        f"Expected submission acceptance for phase3 level-1 fixture. Got:\n{result_phase3_l1.stdout}"
    )
    _assert_phase3_level1_fixture_metrics(summary_phase3_l1)

    # Real submission fixture: lgbm_two_stage at level 2; assert exact portal metrics.
    fixture_csv = project_root / "tests" / "integration" / "resources" / "lgbm_two_stage_submission.csv"
    assert fixture_csv.exists(), f"Fixture not found: {fixture_csv}"
    summary_fixture = tmp_path / "score_summary_live_lgbm_fixture_level2.csv"
    result_fixture = _run_live_submit(
        project_root=project_root,
        submission_csv=fixture_csv,
        level=2,
        summary_csv=summary_fixture,
    )
    assert result_fixture.returncode == 0, (
        f"submit (lgbm fixture, level 2) exited with code {result_fixture.returncode}\n"
        f"stdout:\n{result_fixture.stdout}\nstderr:\n{result_fixture.stderr}"
    )
    assert "Submission accepted" in result_fixture.stdout or "✓ Submission accepted" in result_fixture.stdout, (
        f"Expected submission acceptance for lgbm fixture. Got:\n{result_fixture.stdout}"
    )
    _assert_lgbm_fixture_metrics(summary_fixture)
