"""
Dual-level sweep for lgbm_two_stage: level 1 volume-seeking, level 2 ROI-focused.

Uses the run-scoped Snakemake pipeline: each trial gets a generated run_id, builds
predictions/portfolio/submission under data/12, 13, 14/.../level{level}/{run_id}/,
submits online, and archives to data/15_scores/online/runs/level{level}/{run_id}/.
No flat sweep_level1/ or sweep_level2/ copies; all runs are in the main run archive.

Level 1: aggressive grid (score_threshold, top_k_per_buyer, cold_start_top_k).
Level 2: smaller grid. Mode is forced to online so every trial goes through the portal.

Usage (from repo root):
  uv run scripts/run_level2_threshold_sweep.py --dry-run
  uv run scripts/run_level2_threshold_sweep.py --cores 1
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Per-level sweep specs. Keys are string level ids "1", "2".
LEVEL_SPECS = {
    "1": {
        "thresholds": [-0.02, -0.05],
        "top_k_per_buyer": [150, 400],
        "cold_start_top_k": [50, 200],
        "guardrails": {"min_orders": 0, "min_months": 1, "high_spend": 0, "min_avg_monthly_spend": 0},
    },
    "2": {
        "thresholds": [-0.03],
        "top_k_per_buyer": [400],
        "cold_start_top_k": [200],
        "guardrails": {"min_orders": 0, "min_months": 1, "high_spend": 0, "min_avg_monthly_spend": 0},
    },
}


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


def _get_by_level(config: dict, section: str) -> dict:
    """Return by_level dict for section (modelling.selection or submission), normalizing keys to str."""
    if section == "selection":
        parent = config.get("modelling", {}).get("selection", {})
    else:
        parent = config.get("submission", {})
    by_level = parent.get("by_level") or {}
    result = {}
    for key in ("1", "2"):
        val = by_level.get(key) or (by_level.get(int(key)) if key.isdigit() else None)
        if isinstance(val, dict):
            result[key] = dict(val)
    return result


def _set_by_level(config: dict, section: str, level: str, value: dict | None) -> None:
    """Set by_level[level] for section using string key level."""
    if section == "selection":
        parent = config.setdefault("modelling", {}).setdefault("selection", {})
    else:
        parent = config.setdefault("submission", {})
    by_level = parent.setdefault("by_level", {})
    if value is not None:
        by_level[level] = value
    else:
        by_level.pop(level, None)


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
        config = yaml.safe_load(f)

    original_selection = _get_by_level(config, "selection")
    original_submission = _get_by_level(config, "submission")

    trial_index = 0
    if args.dry_run:
        for level in ("1", "2"):
            spec = LEVEL_SPECS[level]
            print(f"Level {level} trials (run-scoped, online submit + archive):")
            for thresh in spec["thresholds"]:
                for top_k in spec["top_k_per_buyer"]:
                    for cold in spec["cold_start_top_k"]:
                        run_id = _make_run_id(trial_index)
                        trial_index += 1
                        print(
                            f"  run_id={run_id} score_threshold={thresh} top_k_per_buyer={top_k} "
                            f"cold_start_top_k={cold} guardrails={spec['guardrails']}"
                        )
                        print(f"    -> data/15_scores/online/runs/level{level}/{args.approach}/{run_id}/")
        return

    for level in ("1", "2"):
        spec = LEVEL_SPECS[level]
        for thresh in spec["thresholds"]:
            for top_k in spec["top_k_per_buyer"]:
                for cold_start_top_k in spec["cold_start_top_k"]:
                    run_id = _make_run_id(trial_index)
                    trial_index += 1

                    _set_by_level(config, "selection", level, {
                        "score_threshold": float(thresh),
                        "top_k_per_buyer": int(top_k),
                        "guardrails": dict(spec["guardrails"]),
                    })
                    _set_by_level(config, "submission", level, {"cold_start_top_k": int(cold_start_top_k)})
                    if level == "1":
                        _set_by_level(config, "selection", "2", original_selection.get("2"))
                        _set_by_level(config, "submission", "2", original_submission.get("2"))
                    else:
                        _set_by_level(config, "selection", "1", original_selection.get("1"))
                        _set_by_level(config, "submission", "1", original_submission.get("1"))

                    with config_path.open("w") as f:
                        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

                    archive_sentinel = Path(f"data/15_scores/online/runs/level{level}/{args.approach}/{run_id}/.archived")
                    print(f"Level{level} run_id={run_id} threshold={thresh}, top_k={top_k}, cold_start_top_k={cold_start_top_k} ...")
                    cmd = [
                        "uv", "run", "snakemake",
                        str(archive_sentinel),
                        "--configfile", str(config_path),
                        "--cores", str(args.cores),
                    ]
                    rc = subprocess.run(cmd, cwd=Path.cwd()).returncode
                    if rc != 0:
                        print(f"Snakemake failed for level{level} run_id={run_id}", file=sys.stderr)
                        _restore_all(config_path, config, original_selection, original_submission)
                        sys.exit(rc)
                    print(f"  -> {archive_sentinel.parent}")

        _restore_all(config_path, config, original_selection, original_submission)
        print(f"Done level {level}. Restored config by_level for both levels.")

    print("Sweep complete.")


def _restore_all(
    config_path: Path,
    config: dict,
    original_selection: dict[str, dict],
    original_submission: dict[str, dict],
) -> None:
    """Restore both levels' selection and submission by_level using string keys."""
    sel = config.setdefault("modelling", {}).setdefault("selection", {})
    by_level_sel = sel.setdefault("by_level", {})
    for level in ("1", "2"):
        val = original_selection.get(level)
        if val:
            by_level_sel[level] = val
        else:
            by_level_sel.pop(level, None)
    sub = config.setdefault("submission", {})
    by_level_sub = sub.setdefault("by_level", {})
    for level in ("1", "2"):
        val = original_submission.get(level)
        if val:
            by_level_sub[level] = val
        else:
            by_level_sub.pop(level, None)
    with config_path.open("w") as f:
        import yaml
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


if __name__ == "__main__":
    main()
