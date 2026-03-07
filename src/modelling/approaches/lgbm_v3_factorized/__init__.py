"""
Three-stage factorized model (v3): EU = sqrt(n_hat * avg_price_per_unit) / val_months - F.
Stage A: Tweedie count regressor for future order count n_hat.
Stage B: historical quantity-weighted average price (avg_price_per_unit) from feature matrix, no model.
Stage C: combine into EU score.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError as exc:
    raise ImportError("lightgbm is required. Install with: uv add lightgbm") from exc

EXCLUDE_COLS = {
    "legal_entity_id",
    "eclass",
    "manufacturer",
    "label",
    "s_val",
    "n_orders_val",
    "score_base",
    "avg_price_per_unit",
}


def run(
    df: pd.DataFrame,
    *,
    val_months: float = 6.0,
    fixed_fee_eur: float = 10.0,
    lgb_params_classifier: str = "",
    **_: object,
) -> pd.DataFrame:
    """
    Fit Stage A (Tweedie count model for n_orders_val), use avg_price_per_unit as Stage B,
    set score_base = sqrt(n_hat * avg_price_per_unit) / val_months - fixed_fee_eur.
    """
    df = df.copy()
    X_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    use_cat = [c for c in X_cols if not pd.api.types.is_numeric_dtype(df[c])]

    X = df[X_cols].copy()
    for c in X_cols:
        if c in use_cat:
            X[c] = X[c].fillna("").astype(str).astype("category")
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)

    y_count = df["n_orders_val"].values.astype(np.float64)

    # Stage A: Tweedie count regressor (expected future order count)
    default_reg = {
        "objective": "tweedie",
        "tweedie_variance_power": 1.5,
        "verbosity": -1,
        "num_leaves": 31,
        "n_estimators": 100,
    }
    reg_params = default_reg.copy()
    if lgb_params_classifier:
        reg_params.update(json.loads(lgb_params_classifier))
    model_a = lgb.LGBMRegressor(**reg_params)
    model_a.fit(X, y_count, categorical_feature=use_cat if use_cat else "auto")
    n_hat = np.clip(model_a.predict(X), 0, None)

    # Stage B: quantity-weighted average price from lookback (no model)
    v_bar = pd.to_numeric(df["avg_price_per_unit"], errors="coerce").fillna(0).values
    v_bar = np.clip(v_bar, 0, None)

    # Stage C: EU = sqrt(n_hat * v_bar) / val_months - F (savings scaling ~ sqrt, per-month norm)
    z = n_hat * v_bar
    df["score_base"] = np.sqrt(np.maximum(z, 0)) / val_months - fixed_fee_eur
    return df
