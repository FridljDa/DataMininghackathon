"""
Shared run metadata builder for pipeline artifacts.

Produces the same metadata.json payload used by the archive step, so run-scoped
artifacts under data/12_predictions, data/13_portfolio, and data/14_submission
carry a consistent provenance and config snapshot.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        cmd,
        cwd=cwd or Path.cwd(),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    return (result.stdout or "").strip()


def build_run_metadata(
    *,
    run_id: str,
    approach: str | None = None,
    level: int | None = None,
    train_end: str | None = None,
    lookback_months: int | None = None,
    score_threshold: float | None = None,
    top_k_per_buyer: int | None = None,
    min_orders: int | None = None,
    min_months: int | None = None,
    high_spend: float | None = None,
    min_avg_monthly_spend: float | None = None,
    cold_start_top_k: int | None = None,
    selected_features: list[str] | None = None,
    root: Path | None = None,
) -> dict:
    """Build metadata dict (commit, branch, dirty, created_at, approach, level, config)."""
    root = root or Path.cwd()
    try:
        commit = _run(["git", "rev-parse", "HEAD"], cwd=root)
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root)
        porcelain = _run(["git", "status", "--porcelain"], cwd=root)
        dirty = bool(porcelain)
    except (OSError, subprocess.TimeoutExpired):
        commit = ""
        branch = ""
        dirty = False

    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    metadata: dict = {
        "commit": commit,
        "branch": branch,
        "dirty": dirty,
        "created_at": created,
    }
    if approach is not None:
        metadata["approach"] = approach
    if level is not None:
        metadata["level"] = level

    config: dict = {}
    if train_end is not None:
        config["train_end"] = train_end
    if lookback_months is not None:
        config["lookback_months"] = lookback_months
    if score_threshold is not None:
        config["score_threshold"] = score_threshold
    if top_k_per_buyer is not None:
        config["top_k_per_buyer"] = top_k_per_buyer
    if (
        min_orders is not None
        or min_months is not None
        or high_spend is not None
        or min_avg_monthly_spend is not None
    ):
        config["guardrails"] = {}
        if min_orders is not None:
            config["guardrails"]["min_orders"] = min_orders
        if min_months is not None:
            config["guardrails"]["min_months"] = min_months
        if high_spend is not None:
            config["guardrails"]["high_spend"] = high_spend
        if min_avg_monthly_spend is not None:
            config["guardrails"]["min_avg_monthly_spend"] = min_avg_monthly_spend
    if cold_start_top_k is not None:
        config["cold_start_top_k"] = cold_start_top_k
    if selected_features is not None:
        config["selected_features"] = selected_features
    if config:
        metadata["config"] = config

    return metadata


def write_run_metadata(path: Path, metadata: dict) -> None:
    """Write metadata dict to path as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
