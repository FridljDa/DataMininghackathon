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


def run(
    df: pd.DataFrame,
    *,
    savings_rate: float = 0.10,
    eta: int = 2,
    tau: float = 100.0,
    sparse_eta_multiplier: int = 3,
    sparse_tau_multiplier: float = 2.0,
    use_monthly_lookback_rates: bool = False,
    **_: object,
) -> pd.DataFrame:
    """
    Compute Phase 3 score_base from exact lookback features.
    Requires: n_orders_in_lookback, lookback_spend, avg_spend_per_order, days_since_last, history_cohort.
    When use_monthly_lookback_rates is True and avg_monthly_orders_in_lookback/avg_monthly_spend_in_lookback
    exist, score uses those so late joiners are comparable.
    """
    df = df.copy()
    days_since = df["days_since_last"].fillna(0)
    recurrence_decay = np.exp(-days_since / 365.0)

    if use_monthly_lookback_rates and "avg_monthly_orders_in_lookback" in df.columns and "avg_monthly_spend_in_lookback" in df.columns:
        orders_rate = df["avg_monthly_orders_in_lookback"].fillna(0)
        spend_rate = df["avg_monthly_spend_in_lookback"].fillna(0)
        df["score_base"] = orders_rate * recurrence_decay * spend_rate * savings_rate
    else:
        n_orders_lb = df["n_orders_in_lookback"].fillna(0)
        avg_spend = df["avg_spend_per_order"].fillna(0)
        recurrence = n_orders_lb * recurrence_decay
        df["score_base"] = recurrence * avg_spend * savings_rate

    # Sparse-history emission gate: zero out if cohort is sparse and fails stricter gate
    is_sparse = df["history_cohort"] == SPARSE_HISTORY_COHORT
    sparse_fails = is_sparse & ~(
        (df["n_orders_in_lookback"] >= eta * sparse_eta_multiplier)
        & (df["lookback_spend"] >= tau * sparse_tau_multiplier)
    )
    df.loc[sparse_fails, "score_base"] = 0.0
    return df
