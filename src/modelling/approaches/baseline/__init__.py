"""
Baseline scorer: score_base(b,e) = α·m_active + β·√historical_purchase_value − γ·δ_recency.
"""

from __future__ import annotations

import pandas as pd


def run(
    df: pd.DataFrame,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.5,
    **_: object,
) -> pd.DataFrame:
    """
    Compute baseline score. df must contain m_active, historical_purchase_value_sqrt, delta_recency.
    """
    for col in ("m_active", "historical_purchase_value_sqrt", "delta_recency"):
        if col not in df.columns:
            raise ValueError(f"Baseline requires '{col}'. Got: {list(df.columns)}")
    df = df.copy()
    df["score_base"] = (
        alpha * df["m_active"].fillna(0)
        + beta * df["historical_purchase_value_sqrt"].fillna(0)
        - gamma * df["delta_recency"].fillna(0)
    )
    return df
