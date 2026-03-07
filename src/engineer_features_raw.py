"""
Raw feature pass-through: output only non-derived columns from candidates.

Reads raw candidates parquet (from generate_candidates) and writes the same
columns to data/08_features_raw/{mode}/level{level}/features_raw.parquet.
Level 2 includes manufacturer in key columns. No derivation; downstream
engineer_features_derived adds all computed features.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

RAW_COLUMNS_L1 = [
    "legal_entity_id",
    "eclass",
    "n_orders",
    "historical_purchase_value_total",
    "orderdate_min",
    "orderdate_max",
    "orderdates_str",
]
RAW_COLUMNS_L2 = [
    "legal_entity_id",
    "eclass",
    "manufacturer",
    "n_orders",
    "historical_purchase_value_total",
    "orderdate_min",
    "orderdate_max",
    "orderdates_str",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-raw", required=True, dest="candidates_raw", help="Path to raw candidates parquet.")
    parser.add_argument("--output", required=True, help="Path to output features_raw parquet.")
    parser.add_argument(
        "--level",
        required=True,
        choices=("1", "2"),
        help="Level 1 or 2; level 2 requires manufacturer in candidates.",
    )
    args = parser.parse_args()

    raw_path = Path(args.candidates_raw)
    out_path = Path(args.output)
    raw_columns = RAW_COLUMNS_L2 if args.level == "2" else RAW_COLUMNS_L1

    candidates = pd.read_parquet(raw_path)
    for col in raw_columns:
        if col not in candidates.columns:
            raise ValueError(f"candidates_raw must contain '{col}'. Got: {list(candidates.columns)}")

    out = candidates[[c for c in raw_columns if c in candidates.columns]].copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"Wrote {len(out)} raw feature rows to {out_path}")


if __name__ == "__main__":
    main()
