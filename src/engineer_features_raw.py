"""
Raw feature pass-through: output only non-derived columns from candidates.

Reads raw candidates parquet (from generate_candidates) and writes the same
columns to data/08_features_raw/{mode}/features_raw.parquet. No derivation;
downstream engineer_features_derived adds all computed features.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

RAW_COLUMNS = [
    "legal_entity_id",
    "eclass",
    "n_orders",
    "historical_purchase_value_total",
    "orderdate_min",
    "orderdate_max",
    "t_last",
    "orderdates_str",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-raw", required=True, dest="candidates_raw", help="Path to raw candidates parquet.")
    parser.add_argument("--output", required=True, help="Path to output features_raw parquet.")
    args = parser.parse_args()

    raw_path = Path(args.candidates_raw)
    out_path = Path(args.output)

    candidates = pd.read_parquet(raw_path)
    for col in RAW_COLUMNS:
        if col not in candidates.columns:
            raise ValueError(f"candidates_raw must contain '{col}'. Got: {list(candidates.columns)}")

    out = candidates[[c for c in RAW_COLUMNS if c in candidates.columns]].copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"Wrote {len(out)} raw feature rows to {out_path}")


if __name__ == "__main__":
    main()
