"""Regression tests for canonical string-only by_level keys."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


def _load_sweep_module(project_root: Path):
    script_path = project_root / "scripts" / "run_level2_threshold_sweep.py"
    spec = importlib.util.spec_from_file_location("run_level2_threshold_sweep", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repo_config_uses_string_only_by_level_keys() -> None:
    """Checked-in config should use only canonical string level ids."""
    project_root = Path(__file__).resolve().parent.parent.parent
    config_path = project_root / "config.yaml"
    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    selection_by_level = config["modelling"]["selection"]["by_level"]
    submission_by_level = config["submission"]["by_level"]

    assert set(selection_by_level.keys()) == {"1", "2"}
    assert set(submission_by_level.keys()) == {"1", "2"}
    assert all(isinstance(key, str) for key in selection_by_level)
    assert all(isinstance(key, str) for key in submission_by_level)


def test_sweep_helpers_normalize_mixed_keys_to_canonical_strings() -> None:
    """Sweep config builder should collapse int/string duplicates to string keys only."""
    project_root = Path(__file__).resolve().parent.parent.parent
    sweep = _load_sweep_module(project_root)
    get_by_level = getattr(sweep, "_get_by_level")
    build_trial_config = getattr(sweep, "_build_trial_config")

    base_config = {
        "modelling": {
            "selection": {
                "by_level": {
                    1: {"score_threshold": 0.0, "top_k_per_buyer": 400, "guardrails": {"min_orders": 0}},
                    "1": {"score_threshold": -0.05, "top_k_per_buyer": 400, "guardrails": {"min_orders": 1}},
                    "2": {"score_threshold": 0.0, "top_k_per_buyer": 400, "guardrails": {"min_orders": 0}},
                }
            }
        },
        "submission": {
            "by_level": {
                1: {"cold_start_top_k": 50},
                "1": {"cold_start_top_k": 75},
                "2": {"cold_start_top_k": 200},
            }
        },
    }

    original_selection = get_by_level(base_config, "selection")
    original_submission = get_by_level(base_config, "submission")

    trial_config = build_trial_config(
        base_config=base_config,
        level="2",
        selection_overrides={
            "score_threshold": -0.03,
            "top_k_per_buyer": 400,
            "guardrails": {
                "min_orders": 0,
                "min_months": 1,
                "high_spend": 0,
                "min_avg_monthly_spend": 0,
            },
        },
        submission_overrides={"cold_start_top_k": 200},
        original_selection=original_selection,
        original_submission=original_submission,
    )

    selection_by_level = trial_config["modelling"]["selection"]["by_level"]
    submission_by_level = trial_config["submission"]["by_level"]

    assert set(selection_by_level.keys()) == {"1", "2"}
    assert set(submission_by_level.keys()) == {"1", "2"}
    assert all(isinstance(key, str) for key in selection_by_level)
    assert all(isinstance(key, str) for key in submission_by_level)
    assert selection_by_level["1"]["score_threshold"] == -0.05
    assert submission_by_level["1"]["cold_start_top_k"] == 75
