"""Unit tests for LGBM approaches: numeric NaNs preserved for native handling; categorical missing → empty string."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _minimal_df_two_stage() -> pd.DataFrame:
    """Minimal DataFrame for lgbm_two_stage: required excluded cols + one numeric (with NaN) + one categorical (with NaN)."""
    return pd.DataFrame({
        "legal_entity_id": ["b1", "b2", "b3"],
        "eclass": ["e1", "e1", "e1"],
        "manufacturer": ["m1", "m1", "m1"],
        "label": [1, 0, 1],
        "s_val": [10.0, 0.0, 20.0],
        "n_orders_val": [1.0, 0.0, 2.0],
        "n_orders": [1.0, np.nan, 3.0],  # numeric with NaN
        "nace_2": ["A", np.nan, "B"],  # categorical with NaN (object dtype)
    })


def _minimal_df_v3() -> pd.DataFrame:
    """Minimal DataFrame for lgbm_v3_factorized: same idea, includes n_orders_val and avg_price_per_unit."""
    return pd.DataFrame({
        "legal_entity_id": ["b1", "b2", "b3"],
        "eclass": ["e1", "e1", "e1"],
        "manufacturer": ["m1", "m1", "m1"],
        "label": [1, 0, 1],
        "s_val": [10.0, 0.0, 20.0],
        "n_orders_val": [1.0, 0.0, 2.0],
        "score_base": [0.0, 0.0, 0.0],
        "avg_price_per_unit": [5.0, 5.0, 5.0],
        "n_orders": [1.0, np.nan, 3.0],
        "nace_2": ["A", np.nan, "B"],
    })


def test_lgbm_two_stage_preserves_numeric_nan_and_categorical_empty_string() -> None:
    """Numeric columns passed to fit keep NaN; categorical columns use empty string for missing."""
    from unittest.mock import patch

    from modelling.approaches.lgbm_two_stage import run as run_two_stage

    fit_calls: list[tuple] = []

    def capture_fit(self, X, y, **kwargs):
        fit_calls.append((X.copy(), y))

    with patch("modelling.approaches.lgbm_two_stage.lgb.LGBMClassifier.fit", capture_fit), patch(
        "modelling.approaches.lgbm_two_stage.lgb.LGBMRegressor.fit", capture_fit
    ):
        df = _minimal_df_two_stage()
        run_two_stage(df)

    assert len(fit_calls) >= 1
    X_clf = fit_calls[0][0]
    assert "n_orders" in X_clf.columns
    assert "nace_2" in X_clf.columns

    # Numeric: NaN must be preserved (not filled with 0)
    n_orders = X_clf["n_orders"]
    assert pd.isna(n_orders.iloc[1]), "numeric column with NaN should remain NaN in X"
    assert n_orders.iloc[0] == 1.0 and n_orders.iloc[2] == 3.0

    # Categorical: missing → empty string (then category)
    nace = X_clf["nace_2"]
    assert nace.dtype.name == "category"
    # Second row was NaN → should be empty string in category
    assert nace.iloc[1] == "" or (pd.isna(nace.iloc[1]) and "" in nace.cat.categories)


def test_lgbm_v3_factorized_preserves_numeric_nan_and_categorical_empty_string() -> None:
    """Numeric feature columns passed to Stage A fit keep NaN; categorical use empty string for missing."""
    from unittest.mock import patch

    from modelling.approaches.lgbm_v3_factorized import run as run_v3

    fit_calls: list[tuple] = []

    def capture_fit(self, X, y, **kwargs):
        fit_calls.append((X.copy(), y))

    with patch("modelling.approaches.lgbm_v3_factorized.lgb.LGBMRegressor.fit", capture_fit):
        df = _minimal_df_v3()
        run_v3(df)

    assert len(fit_calls) == 1
    X_fit = fit_calls[0][0]
    assert "n_orders" in X_fit.columns
    assert "nace_2" in X_fit.columns

    # Numeric: NaN preserved
    n_orders = X_fit["n_orders"]
    assert pd.isna(n_orders.iloc[1]), "numeric column with NaN should remain NaN in X"
    assert n_orders.iloc[0] == 1.0 and n_orders.iloc[2] == 3.0

    # Categorical: missing → empty string
    nace = X_fit["nace_2"]
    assert nace.dtype.name == "category"
    assert nace.iloc[1] == "" or (pd.isna(nace.iloc[1]) and "" in nace.cat.categories)
