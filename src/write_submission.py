"""Write a baseline submission CSV with required columns legal_entity_id,cluster.

Emits deterministic dummy predictions: for each customer, top N eclass|manufacturer
clusters from their plis_training history (by row count); customers with no history
get a global default cluster. Schema remains legal_entity_id,cluster for scoring.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _read_tabular(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, encoding="utf-8-sig")


def _normalize_cluster(eclass: str | float, manufacturer: str | float) -> str:
    e = "" if pd.isna(eclass) else str(eclass).strip()
    m = "" if pd.isna(manufacturer) else str(manufacturer).strip()
    if e and m and e != "nan" and m != "nan":
        try:
            e = str(int(float(e)))
        except (ValueError, TypeError):
            pass
        return f"{e}|{m}"
    return ""


def _get_default_cluster(plis: pd.DataFrame) -> str:
    required = {"eclass", "manufacturer"}
    if not required.issubset(set(plis.columns)):
        raise ValueError(f"plis must contain {sorted(required)}. Got: {list(plis.columns)}")
    pairs = plis[["eclass", "manufacturer"]].fillna("")
    pairs = pairs[(pairs["eclass"] != "") & (pairs["manufacturer"] != "")]
    if pairs.empty:
        raise ValueError("No non-empty eclass/manufacturer pair found in training data.")
    top = pairs.value_counts().index[0]
    return f"{top[0]}|{top[1]}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write baseline submission CSV with legal_entity_id and cluster (dummy predictions)."
    )
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument(
        "--customer-test",
        required=True,
        help="Path to customer_test.csv (tab-separated).",
    )
    parser.add_argument(
        "--plis-training",
        required=True,
        help="Path to plis_training.csv (tab-separated).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        dest="top_n",
        help="Max clusters per customer from history (default: 3).",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    customer_test = _read_tabular(Path(args.customer_test))
    if "legal_entity_id" not in customer_test.columns:
        raise ValueError(
            f"{args.customer_test} must contain 'legal_entity_id'. "
            f"Got: {list(customer_test.columns)}"
        )

    plis = _read_tabular(Path(args.plis_training))
    for col in ("legal_entity_id", "eclass", "manufacturer"):
        if col not in plis.columns:
            raise ValueError(f"plis must contain '{col}'. Got: {list(plis.columns)}")

    plis["legal_entity_id"] = plis["legal_entity_id"].astype(str)
    e = plis["eclass"].astype(str).str.strip().replace("nan", "")
    m = plis["manufacturer"].astype(str).str.strip().replace("nan", "")
    plis["_cluster"] = e + "|" + m
    plis = plis[(e != "") & (m != "")]
    default_cluster = _get_default_cluster(plis)

    # Per-customer top-N clusters by count (deterministic)
    top_per_customer = (
        plis.groupby(["legal_entity_id", "_cluster"], as_index=False)
        .size()
        .sort_values(["legal_entity_id", "size"], ascending=[True, False])
    )
    rows = []
    for lid in customer_test["legal_entity_id"].astype(str).unique():
        subset = top_per_customer[top_per_customer["legal_entity_id"] == lid]
        clusters = subset.head(args.top_n)["_cluster"].tolist()
        if not clusters:
            rows.append({"legal_entity_id": lid, "cluster": default_cluster})
        else:
            for c in clusters:
                rows.append({"legal_entity_id": lid, "cluster": c})

    submission = pd.DataFrame(rows)
    submission.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
