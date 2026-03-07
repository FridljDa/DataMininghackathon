"""
Live integration test: submit to https://unite-evaluator.vercel.app/challenges/2.

Runs when executing the integration test suite. Requires TEAM and PASSWORD
in the environment (e.g. from .env loaded by the shell or pytest-env).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv


@pytest.mark.integration
def test_submit_challenge_2_live(tmp_path: Path) -> None:
    """Run submit script against the live portal; assert login, upload, and acceptance."""
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")
    team = (os.environ.get("TEAM") or os.environ.get("UNITE_TEAM_NAME") or "").strip().strip("'\"")
    password = (os.environ.get("PASSWORD") or os.environ.get("UNITE_PASSWORD") or "").strip().strip("'\"")
    if not team or not password:
        pytest.fail(
            "TEAM and PASSWORD (or UNITE_TEAM_NAME and UNITE_PASSWORD) must be set in .env or "
            "the environment to run the live submission test."
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
    env["TEAM"] = team
    env["PASSWORD"] = password
    env["PYTHONPATH"] = str(project_root)

    result = subprocess.run(
        cmd,
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
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
    assert "submission_id" in content, f"score_summary_live.csv must contain submission_id. Got:\n{content}"
