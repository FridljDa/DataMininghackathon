"""
Suggest modelling features from feature-analysis summary (machine-readable).

Reads feature_summary.csv (and optionally feature_redundancy.csv) from
data/10_feature_analysis. Applies hard filters (null rate, variance, cardinality),
then redundancy pruning so one representative per correlated group is kept.
Writes a YAML suggestion file for manual copy into config.yaml modelling.features.selected.
Output is advisory only; the pipeline contract is config modelling.features.selected / data/11_features_selected.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from collections import defaultdict

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


def _signal_score(row: pd.Series) -> float:
    """Higher = better for ranking; use abs_spearman_label then label_auc if present."""
    s = row.get("abs_spearman_label")
    if pd.notna(s):
        return float(s)
    a = row.get("label_auc")
    if pd.notna(a):
        return float(a) - 0.5  # 0.5 = random
    return 0.0


def _connected_components(edges: set[tuple[str, str]]) -> list[set[str]]:
    """Return list of connected component sets given undirected edges (a, b) with a < b."""
    adj: dict[str, set[str]] = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    seen = set()
    components = []
    for node in adj:
        if node in seen:
            continue
        comp = set()
        stack = [node]
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            comp.add(n)
            for nb in adj[n]:
                if nb not in seen:
                    stack.append(nb)
        if comp:
            components.append(comp)
    return components


def suggest_features(
    summary: pd.DataFrame,
    redundancy_df: pd.DataFrame | None = None,
) -> tuple[list[str], list[dict]]:
    """
    Return (suggested_feature_names, list of {feature, reason} for dropped).
    Applies hard filters, then redundancy pruning using optional redundancy CSV.
    Suggested list is sorted by name.
    """
    dropped = []
    pass_names = []

    summary_row = {row["feature"]: row for _, row in summary.iterrows()}

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

        pass_names.append(name)

    # Redundancy pruning: keep one per correlated group (best signal)
    if redundancy_df is not None and len(redundancy_df) > 0 and "feature_a" in redundancy_df.columns and "feature_b" in redundancy_df.columns:
        edges = set()
        for _, r in redundancy_df.iterrows():
            a, b = str(r["feature_a"]).strip(), str(r["feature_b"]).strip()
            if a != b and a in pass_names and b in pass_names:
                edges.add((min(a, b), max(a, b)))
        components = _connected_components(edges)
        keep_set = set(pass_names)
        for comp in components:
            if len(comp) <= 1:
                continue
            # Keep the one with highest signal; drop others
            ranked = sorted(comp, key=lambda f: _signal_score(summary_row.get(f, pd.Series())), reverse=True)
            kept = ranked[0]
            for f in ranked[1:]:
                keep_set.discard(f)
                dropped.append({"feature": f, "reason": f"redundant with {kept}"})
        suggested = sorted(keep_set)
    else:
        suggested = sorted(pass_names)

    return suggested, dropped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", required=True, dest="summary_csv", help="Path to feature_summary.csv.")
    parser.add_argument(
        "--redundancy-csv",
        default="",
        dest="redundancy_csv",
        help="Path to feature_redundancy.csv (optional); used to prune redundant features.",
    )
    parser.add_argument("--output", required=True, help="Path to output YAML (feature list + optional metadata).")
    args = parser.parse_args()

    summary = pd.read_csv(args.summary_csv)
    if "feature" not in summary.columns:
        raise ValueError(f"summary CSV must have 'feature' column. Got: {list(summary.columns)}")

    redundancy_df = None
    if args.redundancy_csv and Path(args.redundancy_csv).exists():
        redundancy_df = pd.read_csv(args.redundancy_csv)

    suggested, dropped = suggest_features(summary, redundancy_df=redundancy_df)

    out = {
        "suggested_features": suggested,
        "meta": {
            "source": str(args.summary_csv),
            "redundancy_source": str(args.redundancy_csv) if args.redundancy_csv else None,
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
