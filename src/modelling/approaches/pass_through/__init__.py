"""
Pass-through: submit all candidates with no further selection.
Sets score_base to a constant positive value so every row passes threshold/guardrails when configured for pass-through.
"""

from __future__ import annotations

import pandas as pd


def run(
    df: pd.DataFrame,
    *,
    score_base_constant: float = 1.0,
    **_: object,
) -> pd.DataFrame:
    """
    Set score_base to a constant positive value for every row so all candidates are kept by selection policy
    when threshold/guardrails/top_k are configured for pass-through (e.g. threshold very low, guardrails permissive, K large).
    """
    df = df.copy()
    df["score_base"] = score_base_constant
    return df
