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
