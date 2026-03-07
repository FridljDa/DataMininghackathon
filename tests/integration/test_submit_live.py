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


@pytest.mark.integration
def test_submit_challenge_2_live(tmp_path: Path) -> None:
    """Run submit script against the live portal; assert login, upload, and acceptance."""
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

    submission_csv = tmp_path / "submission.csv"
    submission_csv.write_text(
        "legal_entity_id,cluster\n1,30020903|Bissell\n",
        encoding="utf-8",
    )
    summary_csv = tmp_path / "score_summary_live.csv"

    cmd = [
        sys.executable,
        "-m",
        "src.submit",
        "--challenge",
        "2",
        "--file",
        str(submission_csv),
        "--level",
        "2",
        "--summary-csv",
        str(summary_csv),
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    result = subprocess.run(
        cmd,
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=360,
        check=False,
    )

    assert result.returncode == 0, (
        f"submit exited with code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "Submission accepted" in result.stdout or "✓ Submission accepted" in result.stdout, (
        f"Expected submission acceptance message in stdout. Got:\n{result.stdout}"
    )
    assert summary_csv.exists(), f"Live summary CSV not written: {summary_csv}"
    content = summary_csv.read_text()
    assert "total_score" in content, f"score_summary_live.csv must contain total_score. Got:\n{content}"
    assert "total_fees" in content, f"score_summary_live.csv must contain total_fees. Got:\n{content}"
    assert "submission_id" in content, f"score_summary_live.csv must contain submission_id. Got:\n{content}"

    rows = list(csv.DictReader(content.splitlines()))
    assert rows, f"score_summary_live.csv must contain at least one data row. Got:\n{content}"

    score = float(rows[0]["total_score"])
    fees = float(rows[0]["total_fees"])
    assert fees == 10.0, f"Expected total_fees=10.0 for one prediction. Got {fees}. CSV:\n{content}"
    assert score == -10.0, f"Expected total_score=-10.0 for one prediction with no savings. Got {score}. CSV:\n{content}"
