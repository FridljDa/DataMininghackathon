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
    Compute baseline score.

    Compatibility notes:
    - if `m_active` is missing but `n_orders` exists, use `n_orders` as `m_active`
    - if `historical_purchase_value_sqrt` is missing but `historical_purchase_value_total`
      exists, derive it as sqrt(max(total, 0))
    """
    df = df.copy()

    if "m_active" not in df.columns and "n_orders" in df.columns:
        df["m_active"] = pd.to_numeric(df["n_orders"], errors="coerce").fillna(0)

    if (
        "historical_purchase_value_sqrt" not in df.columns
        and "historical_purchase_value_total" in df.columns
    ):
        total = pd.to_numeric(df["historical_purchase_value_total"], errors="coerce").fillna(0)
        df["historical_purchase_value_sqrt"] = total.clip(lower=0) ** 0.5

    for col in ("m_active", "historical_purchase_value_sqrt", "delta_recency"):
        if col not in df.columns:
            raise ValueError(f"Baseline requires '{col}'. Got: {list(df.columns)}")

    df["score_base"] = (
        alpha * df["m_active"].fillna(0)
        + beta * df["historical_purchase_value_sqrt"].fillna(0)
        - gamma * df["delta_recency"].fillna(0)
    )
    return df
