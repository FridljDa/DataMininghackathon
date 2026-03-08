"""
Short deadline sweep for lgbm_two_stage: pending level-2 thresholds plus
aggressive level-1 warm-cap expansion.

Uses the run-scoped Snakemake pipeline: each trial gets a generated run_id, builds
predictions/portfolio/submission under data/12, 13, 14/.../level{level}/{run_id}/,
including a metadata.json sidecar (created at data/12, propagated through 13 and 14).
Submits online and archives to data/15_scores/online/runs/level{level}/{approach}/{run_id}/,
reusing that upstream metadata. No flat sweep_level1/ or sweep_level2/ copies; all runs
are in the main run archive. The script writes a temporary config per trial and never
rewrites config.yaml.

Runs the remaining high-value deadline trials as an explicit ordered list of
per-run configs. This avoids accidental cartesian-product expansion and makes
the last-hour run plan easy to edit safely.

Usage (from repo root):
  uv run scripts/run_level2_threshold_sweep.py --dry-run
  uv run scripts/run_level2_threshold_sweep.py --cores 1
"""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Explicit ordered trial plan for the deadline push.
# Prioritize level 1 volume expansion first: leaderboard gap is much larger there,
# and recent evidence suggests warm-cap expansion is a stronger lever than small
# threshold nudges. Each trial may optionally set `approach`; otherwise the
# command-line `--approach` value is used.
TRIAL_SPECS = [
    {
        "approach": "lgbm_two_stage",
        "level": "1",
        "score_threshold": 0.0,
        "top_k_per_buyer": 1200,
        "cold_start_top_k": 500,
        "guardrails": {"min_orders": 0, "min_months": 1, "high_spend": 0, "min_avg_monthly_spend": 0},
    },
    {
        "approach": "phase3_repro",
        "level": "1",
        "score_threshold": 0.0,
        "top_k_per_buyer": 150,
        "cold_start_top_k": 50,
        "guardrails": {"min_orders": 2, "min_months": 2, "high_spend": 200.0, "min_avg_monthly_spend": 0},
    },
]


def _make_run_id(trial_index: int) -> str:
    """Generate a unique run_id for a sweep trial (timestamp_shortsha_index)."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        short_sha = (r.stdout or "").strip() or "norepo"
    except (OSError, subprocess.TimeoutExpired):
        short_sha = "norepo"
    return f"{ts}_{short_sha}_{trial_index}"


def _normalized_by_level(by_level: dict | None) -> dict[str, dict]:
    """Return by_level with only string keys '1' and '2', merging int and str sources."""
    result: dict[str, dict] = {}
    if not by_level:
        return result
    for key in ("1", "2"):
        val = by_level.get(key) or (by_level.get(int(key)) if key.isdigit() else None)
        if isinstance(val, dict):
            result[key] = dict(val)
    return result


def _get_by_level(config: dict, section: str) -> dict[str, dict]:
    """Return normalized by_level for section (modelling.selection or submission)."""
    if section == "selection":
        parent = config.get("modelling", {}).get("selection", {})
    else:
        parent = config.get("submission", {})
    return _normalized_by_level(parent.get("by_level"))


def _set_by_level_canonical(parent: dict, by_level: dict[str, dict]) -> None:
    """Set parent['by_level'] to the given dict (string keys only). Replaces any existing by_level."""
    parent["by_level"] = dict(by_level)


def _build_trial_config(
    base_config: dict,
    level: str,
    selection_overrides: dict,
    submission_overrides: dict,
    original_selection: dict[str, dict],
    original_submission: dict[str, dict],
) -> dict:
    """Build a config dict for one trial: normalized by_level with trial overrides, other level restored."""
    config = copy.deepcopy(base_config)
    sel = config.setdefault("modelling", {}).setdefault("selection", {})
    sub = config.setdefault("submission", {})

    selection_by_level = dict(original_selection)
    selection_by_level[level] = selection_overrides
    _set_by_level_canonical(sel, selection_by_level)

    submission_by_level = dict(original_submission)
    submission_by_level[level] = submission_overrides
    _set_by_level_canonical(sub, submission_by_level)

    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--mode", type=str, default="online", choices=("online", "offline"))
    parser.add_argument("--approach", type=str, default="lgbm_two_stage")
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.mode != "online":
        print("Sweep uses online submission; --mode is ignored (online).", file=sys.stderr)

    config_path = args.config.resolve()
    if not config_path.is_file():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    try:
        import yaml
    except ImportError:
        print("PyYAML required: pip install pyyaml", file=sys.stderr)
        sys.exit(1)

    with config_path.open() as f:
        base_config = yaml.safe_load(f)

    original_selection = _get_by_level(base_config, "selection")
    original_submission = _get_by_level(base_config, "submission")

    trial_index = 0
    if args.dry_run:
        for trial in TRIAL_SPECS:
            approach = str(trial.get("approach", args.approach))
            level = str(trial["level"])
            score_threshold = float(trial["score_threshold"])
            top_k = int(trial["top_k_per_buyer"])
            cold_start_top_k = int(trial["cold_start_top_k"])
            run_id = _make_run_id(trial_index)
            trial_index += 1
            print(
                f"approach={approach} level={level} run_id={run_id} score_threshold={score_threshold} "
                f"top_k_per_buyer={top_k} cold_start_top_k={cold_start_top_k} "
                f"guardrails={trial['guardrails']}"
            )
            print(f"  -> data/15_scores/online/runs/level{level}/{approach}/{run_id}/")
        return

    last_level: str | None = None
    for trial in TRIAL_SPECS:
        approach = str(trial.get("approach", args.approach))
        level = str(trial["level"])
        score_threshold = float(trial["score_threshold"])
        top_k = int(trial["top_k_per_buyer"])
        cold_start_top_k = int(trial["cold_start_top_k"])
        run_id = _make_run_id(trial_index)
        trial_index += 1

        if last_level is not None and level != last_level:
            print(f"Done level {last_level}.")

        trial_config = _build_trial_config(
            base_config,
            level,
            {
                "score_threshold": score_threshold,
                "top_k_per_buyer": top_k,
                "guardrails": dict(trial["guardrails"]),
            },
            {"cold_start_top_k": cold_start_top_k},
            original_selection,
            original_submission,
        )

        archive_sentinel = Path(f"data/15_scores/online/runs/level{level}/{approach}/{run_id}/.archived")
        print(
            f"approach={approach} level={level} run_id={run_id} threshold={score_threshold}, "
            f"top_k={top_k}, cold_start_top_k={cold_start_top_k} ..."
        )

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            delete=False,
            prefix="sweep_config_",
        ) as tmp:
            yaml.safe_dump(trial_config, tmp, default_flow_style=False, allow_unicode=True, sort_keys=False)
            tmp.flush()
            tmp_path = Path(tmp.name)
        try:
            cmd = [
                "uv", "run", "snakemake",
                str(archive_sentinel),
                "--configfile", str(tmp_path),
                "--cores", str(args.cores),
            ]
            rc = subprocess.run(cmd, cwd=Path.cwd()).returncode
            if rc != 0:
                print(f"Snakemake failed for level{level} run_id={run_id}", file=sys.stderr)
                sys.exit(rc)
            print(f"  -> {archive_sentinel.parent}")
        finally:
            tmp_path.unlink(missing_ok=True)

        last_level = level

    if last_level is not None:
        print(f"Done level {last_level}.")

    print("Sweep complete.")


if __name__ == "__main__":
    main()
