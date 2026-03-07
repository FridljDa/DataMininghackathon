"""
Config-driven feature selection: keep keys + selected feature columns.

Reads full features parquet and a list of feature names (from config).
Level 1 keys: legal_entity_id, eclass; Level 2 adds manufacturer.
Validates that all requested features exist; fails fast with a clear error
if any are missing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

KEY_COLUMNS_L1 = ("legal_entity_id", "eclass")
KEY_COLUMNS_L2 = ("legal_entity_id", "eclass", "manufacturer")


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


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
    parser.add_argument(
        "--level",
        required=True,
        choices=("1", "2"),
        help="Level 1 or 2; level 2 keys include manufacturer.",
    )
    args = parser.parse_args()

    key_columns = KEY_COLUMNS_L2 if args.level == "2" else KEY_COLUMNS_L1

    df = pd.read_parquet(args.features)
    selected_raw = [s.strip() for s in args.selected_features.split(",") if s.strip()]
    selected = _dedupe_preserve_order(selected_raw)

    missing = [f for f in selected if f not in df.columns]
    if missing:
        raise ValueError(
            f"selected_features contain columns not in features parquet: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    for key in key_columns:
        if key not in df.columns:
            raise ValueError(f"Features parquet must contain key column '{key}'. Got: {list(df.columns)}")

    out_cols = _dedupe_preserve_order(
        [c for c in key_columns if c in df.columns] + [f for f in selected if f in df.columns]
    )
    out = df[out_cols].copy()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"Wrote {len(out)} rows with {len(out_cols)} columns to {out_path}")


if __name__ == "__main__":
    main()
