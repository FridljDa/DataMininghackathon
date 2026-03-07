"""
Attach validation-period labels and spend to a candidates/features dataframe.

Uses plis (split) rows in [val_start, val_end] to compute n_orders_val, s_val, label.
Level 1: key = (legal_entity_id, eclass); Level 2: key = (legal_entity_id, eclass, manufacturer).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def attach_validation_labels(
    df: pd.DataFrame,
    plis_path: str | Path,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
    n_min_label: int,
    level: int = 1,
) -> pd.DataFrame:
    """
    Merge validation-period order count and spend onto df; add label.

    Adds columns: n_orders_val, s_val, label.
    label = 1 if n_orders_val >= n_min_label else 0.
    """
    key_cols = ["legal_entity_id", "eclass", "manufacturer"] if level == 2 else ["legal_entity_id", "eclass"]

    plis = pd.read_csv(plis_path, sep="\t", low_memory=False)
    plis["orderdate"] = pd.to_datetime(plis["orderdate"], format="%Y-%m-%d")
    plis["legal_entity_id"] = plis["legal_entity_id"].astype(str)
    plis["eclass"] = plis["eclass"].astype(str).str.strip().replace("nan", "")
    plis = plis[(plis["orderdate"] >= val_start) & (plis["orderdate"] <= val_end)]
    q = pd.to_numeric(plis["quantityvalue"], errors="coerce").fillna(0)
    v = pd.to_numeric(plis["vk_per_item"], errors="coerce").fillna(0)
    plis["_spend"] = q * v
    plis = plis[plis["eclass"] != ""]
    if level == 2:
        if "manufacturer" not in plis.columns:
            raise ValueError("plis must contain 'manufacturer' for level 2.")
        plis["manufacturer"] = plis["manufacturer"].astype(str).str.strip().replace("nan", "")
        plis = plis[plis["manufacturer"] != ""]

    val_agg = (
        plis.groupby(key_cols)
        .agg(n_orders_val=("_spend", "count"), s_val=("_spend", "sum"))
        .reset_index()
    )
    out = df.merge(val_agg, on=key_cols, how="left")
    out["n_orders_val"] = out["n_orders_val"].fillna(0).astype(int)
    out["s_val"] = out["s_val"].fillna(0)
    out["label"] = (out["n_orders_val"] >= n_min_label).astype(int)
    return out
