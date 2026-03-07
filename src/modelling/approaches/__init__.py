"""Modelling approaches: each implements run(df, **params) and returns df with score_base set."""

import modelling.approaches.baseline as baseline_mod
import modelling.approaches.lgbm_two_stage as lgbm_two_stage_mod

APPROACHES = {
    "baseline": baseline_mod,
    "lgbm_two_stage": lgbm_two_stage_mod,
}


def get_approach(name: str):
    if name not in APPROACHES:
        raise ValueError(f"Unknown approach {name!r}. Choose from: {list(APPROACHES)}")
    return APPROACHES[name]
