"""
Focused level-1 experiment set: hybrid (lgbm_two_stage primary + phase3_repro backfill)
with a small threshold sweep, after routing and hybrid-union fixes.

Uses the run-scoped Snakemake pipeline: each trial gets a generated run_id, builds
predictions/portfolio/submission under data/12, 13, 14/.../level{level}/{run_id}/,
including a metadata.json sidecar. Submits online and archives to
data/15_scores/online/runs/level{level}/{approach}/{run_id}/. Writes a temporary
config per trial and never rewrites config.yaml.

Trials are defined by override YAMLs in --sweeps-dir (default config/sweeps). Each
override is merged on top of config.yaml; _sweep.approach and _sweep.level select the
run. For approach=hybrid_lgbm_phase3, both lgbm_two_stage and phase3_repro are built
for the same run_id and merged (union of buyers, backfill); override can set
modelling.enabled_approaches to include hybrid_lgbm_phase3 and phase3_repro.

Usage (from repo root):
  uv run scripts/run_level2_threshold_sweep.py --dry-run
  uv run scripts/run_level2_threshold_sweep.py --cores 1
  uv run scripts/run_level2_threshold_sweep.py --sweeps-dir config/sweeps --cores 1
"""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_SWEEPS_DIR = Path("config/sweeps")


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


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override into base recursively. Mutates base; returns base. Dicts are merged, other values replace."""
    for key, ov_val in override.items():
        if key.startswith("_"):
            continue
        if key in base and isinstance(base[key], dict) and isinstance(ov_val, dict):
            _deep_merge(base[key], ov_val)
        else:
            base[key] = copy.deepcopy(ov_val)
    return base


def _normalize_by_level_in_config(config: dict) -> None:
    """Normalize all by_level dicts in config to canonical string keys '1' and '2'. Mutates config."""
    for parent in [
        config.get("modelling", {}).get("selection", {}),
        config.get("submission", {}),
    ]:
        if "by_level" in parent and isinstance(parent["by_level"], dict):
            parent["by_level"] = _normalized_by_level(parent["by_level"])


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


def _load_override_and_merge(
    base_config: dict, override_path: Path, default_approach: str = "lgbm_two_stage"
) -> tuple[dict, str, str]:
    """Load an override YAML, extract _sweep (level, approach), merge rest onto base, normalize. Returns (merged_config, level, approach)."""
    import yaml

    with override_path.open() as f:
        override = yaml.safe_load(f) or {}
    sweep = override.pop("_sweep", None) or {}
    level = str(sweep.get("level", "1"))
    approach = str(sweep.get("approach") or default_approach)

    config = copy.deepcopy(base_config)
    _deep_merge(config, override)
    _normalize_by_level_in_config(config)
    # For normal trials, build only the selected approach.
    # For combined trials, keep the override-provided source approaches so the
    # synthetic merge approach has upstream portfolios/scores to combine.
    mod = config.setdefault("modelling", {})
    if approach == "combined_enabled_approaches":
        enabled = mod.get("enabled_approaches")
        if isinstance(enabled, list) and enabled:
            if approach not in enabled:
                enabled.append(approach)
            mod["enabled_approaches"] = enabled
        else:
            base_enabled = list(base_config.get("modelling", {}).get("enabled_approaches", []))
            if approach not in base_enabled:
                base_enabled.append(approach)
            mod["enabled_approaches"] = base_enabled
    else:
        mod["enabled_approaches"] = [approach]
    mod["enabled_levels"] = [int(level) if level.isdigit() else level]
    return config, level, approach


def _resolved_selection_for_level(config: dict, level: str) -> dict:
    """Return resolved selection params for level from config (for display)."""
    return _get_by_level(config, "selection").get(level, {})


def _resolved_submission_for_level(config: dict, level: str) -> dict:
    """Return resolved submission params for level from config (for display)."""
    return _get_by_level(config, "submission").get(level, {})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--sweeps-dir", type=Path, default=DEFAULT_SWEEPS_DIR)
    parser.add_argument("--mode", type=str, default="online", choices=("online", "offline"))
    parser.add_argument("--approach", type=str, default="lgbm_two_stage", help="Default approach when _sweep.approach is not set in override")
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.mode != "online":
        print("Sweep uses online submission; --mode is ignored (online).", file=sys.stderr)

    config_path = args.config.resolve()
    if not config_path.is_file():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    sweeps_dir = args.sweeps_dir.resolve()
    if not sweeps_dir.is_dir():
        print(f"Sweeps dir not found: {sweeps_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        import yaml
    except ImportError:
        print("PyYAML required: pip install pyyaml", file=sys.stderr)
        sys.exit(1)

    with config_path.open() as f:
        base_config = yaml.safe_load(f)

    override_files = sorted(sweeps_dir.glob("*.yaml"), key=lambda p: p.name)
    if not override_files:
        print(f"No *.yaml override files in {sweeps_dir}", file=sys.stderr)
        sys.exit(1)

    trial_index = 0
    if args.dry_run:
        for override_path in override_files:
            trial_config, level, approach = _load_override_and_merge(base_config, override_path, args.approach)
            sel = _resolved_selection_for_level(trial_config, level)
            sub = _resolved_submission_for_level(trial_config, level)
            run_id = _make_run_id(trial_index)
            trial_index += 1
            print(
                f"override={override_path.name} approach={approach} level={level} run_id={run_id} "
                f"score_threshold={sel.get('score_threshold')} top_k_per_buyer={sel.get('top_k_per_buyer')} "
                f"cold_start_top_k={sub.get('cold_start_top_k')} guardrails={sel.get('guardrails', {})}"
            )
            print(f"  -> data/15_scores/online/runs/level{level}/{approach}/{run_id}/")
        return

    last_level: str | None = None
    for override_path in override_files:
        trial_config, level, approach = _load_override_and_merge(base_config, override_path, args.approach)
        sel = _resolved_selection_for_level(trial_config, level)
        sub = _resolved_submission_for_level(trial_config, level)
        run_id = _make_run_id(trial_index)
        trial_index += 1

        if last_level is not None and level != last_level:
            print(f"Done level {last_level}.")

        archive_sentinel = Path(f"data/15_scores/online/runs/level{level}/{approach}/{run_id}/.archived")
        print(
            f"override={override_path.name} approach={approach} level={level} run_id={run_id} "
            f"threshold={sel.get('score_threshold')}, top_k={sel.get('top_k_per_buyer')}, "
            f"cold_start_top_k={sub.get('cold_start_top_k')} ..."
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
