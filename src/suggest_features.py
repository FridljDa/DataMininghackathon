"""
Suggest modelling features from feature-analysis summary (machine-readable).

Reads feature_summary.csv from data/09_feature_analysis, applies heuristic
filters (null rate, variance, cardinality), and writes a YAML suggestion file
for manual copy into config.yaml modelling.selected_features.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

# Columns that are keys or non-predictors; never suggest
EXCLUDE_COLUMNS = {"legal_entity_id", "eclass"}

# Heuristic thresholds
MAX_NULL_RATE = 0.5
MIN_N_UNIQUE_NUMERIC = 2
MIN_N_UNIQUE_CATEGORICAL = 2
MAX_CARDINALITY_CATEGORICAL = 500


def _is_numeric_dtype(dtype_str: str) -> bool:
    return "int" in dtype_str or "float" in dtype_str


def suggest_features(summary: pd.DataFrame) -> tuple[list[str], list[dict]]:
    """
    Return (suggested_feature_names, list of {feature, reason} for dropped).
    Suggested list is sorted deterministically by name.
    """
    dropped = []
    suggested = []

    for _, row in summary.iterrows():
        name = row["feature"]
        if name in EXCLUDE_COLUMNS:
            dropped.append({"feature": name, "reason": "excluded (key/non-predictor)"})
            continue

        null_rate = row.get("null_rate", 0.0)
        if pd.isna(null_rate):
            null_rate = row["null_count"] / row["n"] if row["n"] else 0.0
        if null_rate > MAX_NULL_RATE:
            dropped.append({"feature": name, "reason": f"null_rate {null_rate:.2f} > {MAX_NULL_RATE}"})
            continue

        n_unique = row.get("n_unique")
        if pd.isna(n_unique):
            n_unique = int(row.get("n_unique", 0))
        else:
            n_unique = int(n_unique)

        if _is_numeric_dtype(str(row.get("dtype", ""))):
            if n_unique < MIN_N_UNIQUE_NUMERIC:
                dropped.append({"feature": name, "reason": f"numeric n_unique={n_unique} < {MIN_N_UNIQUE_NUMERIC}"})
                continue
            std = row.get("std")
            if pd.notna(std) and std == 0:
                dropped.append({"feature": name, "reason": "zero variance (std=0)"})
                continue
        else:
            if n_unique < MIN_N_UNIQUE_CATEGORICAL:
                dropped.append({"feature": name, "reason": f"categorical n_unique={n_unique} < {MIN_N_UNIQUE_CATEGORICAL}"})
                continue
            if n_unique > MAX_CARDINALITY_CATEGORICAL:
                dropped.append({"feature": name, "reason": f"categorical n_unique={n_unique} > {MAX_CARDINALITY_CATEGORICAL}"})
                continue

        suggested.append(name)

    suggested.sort()
    return suggested, dropped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", required=True, dest="summary_csv", help="Path to feature_summary.csv.")
    parser.add_argument("--output", required=True, help="Path to output YAML (feature list + optional metadata).")
    args = parser.parse_args()

    summary = pd.read_csv(args.summary_csv)
    if "feature" not in summary.columns:
        raise ValueError(f"summary CSV must have 'feature' column. Got: {list(summary.columns)}")

    suggested, dropped = suggest_features(summary)

    out = {
        "suggested_features": suggested,
        "meta": {
            "source": str(args.summary_csv),
            "n_suggested": len(suggested),
            "n_dropped": len(dropped),
            "dropped": dropped,
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"Suggested {len(suggested)} features; dropped {len(dropped)}. Wrote {out_path}")
    for d in dropped:
        print(f"  dropped: {d['feature']} — {d['reason']}")


if __name__ == "__main__":
    main()
