"""
Phase 3 reproduction: exact standalone score formula and sparse-history emission gate.

score_base = n_orders_in_lookback * exp(-days_since_last / 365) * avg_spend_per_order * savings_rate
With use_monthly_lookback_rates=True (and columns present): uses avg_monthly_orders_in_lookback and
avg_monthly_spend_in_lookback so scores are comparable across buyer tenure (late joiners).
Sparse-history cohort: zero out rows that fail n_orders_in_lookback >= eta*sparse_eta_multiplier
and lookback_spend >= tau*sparse_tau_multiplier. eta/tau/multipliers come from config.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Cohort label produced by engineer_features; must match for gate to apply
SPARSE_HISTORY_COHORT = "sparse_history"


def _get_series(df: pd.DataFrame, candidates: tuple[str, ...], default: float = 0.0) -> pd.Series:
    """Return first available numeric column as float series, else constant default."""
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype="float64")


def run(
    df: pd.DataFrame,
    *,
    savings_rate: float = 0.10,
    eta: int = 2,
    tau: float = 100.0,
    sparse_eta_multiplier: int = 3,
    sparse_tau_multiplier: float = 2.0,
    use_monthly_lookback_rates: bool = False,
    recency_decay_days: float = 365.0,
    **_: object,
) -> pd.DataFrame:
    """
    Compute Phase 3 score_base from lookback-style activity signals.
    Prefers exact lookback columns when present, but falls back to compatible
    derived columns for reduced feature sets.
    """
    df = df.copy()
    days_since = _get_series(df, ("days_since_last", "delta_recency"), default=0.0)
    decay_divisor = max(recency_decay_days, 1e-6)
    recurrence_decay = np.exp(-days_since / decay_divisor)

    if use_monthly_lookback_rates and "avg_monthly_orders_in_lookback" in df.columns and "avg_monthly_spend_in_lookback" in df.columns:
        orders_rate = _get_series(df, ("avg_monthly_orders_in_lookback",), default=0.0)
        spend_rate = _get_series(df, ("avg_monthly_spend_in_lookback",), default=0.0)
        df["score_base"] = orders_rate * recurrence_decay * spend_rate * savings_rate
    else:
        n_orders_lb = _get_series(
            df,
            ("n_orders_in_lookback", "n_orders"),
            default=0.0,
        )
        avg_spend = _get_series(
            df,
            ("avg_spend_per_order",),
            default=0.0,
        )
        recurrence = n_orders_lb * recurrence_decay
        df["score_base"] = recurrence * avg_spend * savings_rate

    # Sparse-history emission gate only when required fields are available.
    required_for_sparse_gate = {"history_cohort", "n_orders_in_lookback", "lookback_spend"}
    if required_for_sparse_gate.issubset(df.columns):
        is_sparse = df["history_cohort"] == SPARSE_HISTORY_COHORT
        sparse_fails = is_sparse & ~(
            (df["n_orders_in_lookback"] >= eta * sparse_eta_multiplier)
            & (df["lookback_spend"] >= tau * sparse_tau_multiplier)
        )
        df.loc[sparse_fails, "score_base"] = 0.0
    return df
