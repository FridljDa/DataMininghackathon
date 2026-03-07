"""
Phase 3 reproduction: exact standalone score formula and sparse-history emission gate.

score_base = n_orders_in_lookback * exp(-days_since_last / 365) * avg_spend_per_order * savings_rate
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
    **_: object,
) -> pd.DataFrame:
    """
    Compute Phase 3 score_base from exact lookback features.
    Requires: n_orders_in_lookback, lookback_spend, avg_spend_per_order, days_since_last, history_cohort.
    eta, tau, sparse_eta_multiplier, sparse_tau_multiplier are read from config (modelling.approaches.phase3_repro).
    """
    df = df.copy()
    n_orders_lb = df["n_orders_in_lookback"].fillna(0)
    days_since = df["days_since_last"].fillna(0)
    avg_spend = df["avg_spend_per_order"].fillna(0)
    recurrence = n_orders_lb * np.exp(-days_since / 365.0)
    df["score_base"] = recurrence * avg_spend * savings_rate

    # Sparse-history emission gate: zero out if cohort is sparse and fails stricter gate
    is_sparse = df["history_cohort"] == SPARSE_HISTORY_COHORT
    sparse_fails = is_sparse & ~(
        (df["n_orders_in_lookback"] >= eta * sparse_eta_multiplier)
        & (df["lookback_spend"] >= tau * sparse_tau_multiplier)
    )
    df.loc[sparse_fails, "score_base"] = 0.0
    return df
