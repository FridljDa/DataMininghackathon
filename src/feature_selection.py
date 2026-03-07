"""
Config-driven feature selection: keep keys + selected feature columns.

Reads full features parquet and a list of feature names (from config).
Validates that all requested features exist; fails fast with a clear error
if any are missing. Writes parquet with legal_entity_id, eclass, and
selected feature columns only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

KEY_COLUMNS = ("legal_entity_id", "eclass")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, help="Path to features_all parquet.")
    parser.add_argument(
        "--selected-features",
        required=True,
        dest="selected_features",
        help="Comma-separated list of feature column names to keep.",
    )
    parser.add_argument("--output", required=True, help="Path to output selected features parquet.")
    args = parser.parse_args()

    df = pd.read_parquet(args.features)
    selected = [s.strip() for s in args.selected_features.split(",") if s.strip()]

    missing = [f for f in selected if f not in df.columns]
    if missing:
        raise ValueError(
            f"selected_features contain columns not in features parquet: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    for key in KEY_COLUMNS:
        if key not in df.columns:
            raise ValueError(f"Features parquet must contain key column '{key}'. Got: {list(df.columns)}")

    out_cols = [c for c in KEY_COLUMNS if c in df.columns] + [f for f in selected if f in df.columns]
    out = df[out_cols].copy()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"Wrote {len(out)} rows with {len(out_cols)} columns to {out_path}")


if __name__ == "__main__":
    main()
