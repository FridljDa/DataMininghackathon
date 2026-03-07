"""
Two-stage LightGBM: EU = p_recur * v_hat * r - F.
Stage A: binary classifier for recurrence; Stage B: regressor on s_val (positives only).
"""

from __future__ import annotations

import json

import pandas as pd

try:
    import lightgbm as lgb
except ImportError as exc:
    raise ImportError("lightgbm is required. Install with: uv add lightgbm") from exc

EXCLUDE_COLS = {
    "legal_entity_id",
    "eclass",
    "label",
    "s_val",
    "n_orders_val",
    "score_base",
}


def run(
    df: pd.DataFrame,
    *,
    val_months: float = 6.0,
    savings_rate: float = 0.10,
    fixed_fee_eur: float = 10.0,
    lgb_params_classifier: str = "",
    lgb_params_regressor: str = "",
    **_: object,
) -> pd.DataFrame:
    """
    Fit two-stage model and set score_base = EU (per-month scaled).
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

    y_label = df["label"].values

    # Stage A: recurrence classifier
    default_clf = {"objective": "binary", "verbosity": -1, "num_leaves": 31, "n_estimators": 100}
    clf_params = default_clf.copy()
    if lgb_params_classifier:
        clf_params.update(json.loads(lgb_params_classifier))
    model_a = lgb.LGBMClassifier(**clf_params)
    model_a.fit(X, y_label, categorical_feature=use_cat if use_cat else "auto")
    p_recur = model_a.predict_proba(X)[:, 1]

    # Stage B: conditional value regressor (positive examples only)
    pos = y_label == 1
    if pos.sum() < 10:
        v_hat = pd.Series(0.0, index=df.index)
        v_hat.loc[pos] = df.loc[pos, "s_val"]
    else:
        X_pos = X.loc[pos]
        y_val_pos = df.loc[pos, "s_val"].values
        default_reg = {"objective": "regression", "verbosity": -1, "num_leaves": 31, "n_estimators": 100}
        reg_params = default_reg.copy()
        if lgb_params_regressor:
            reg_params.update(json.loads(lgb_params_regressor))
        model_b = lgb.LGBMRegressor(**reg_params)
        model_b.fit(X_pos, y_val_pos, categorical_feature=use_cat if use_cat else "auto")
        v_hat = pd.Series(model_b.predict(X), index=df.index)
        v_hat = v_hat.clip(lower=0)

    scale = 1.0 / val_months
    df["score_base"] = p_recur * v_hat * scale * savings_rate - fixed_fee_eur
    return df
