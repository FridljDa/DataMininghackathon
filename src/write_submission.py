"""Write a baseline submission CSV with required columns legal_entity_id,cluster."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _read_tabular(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, encoding="utf-8-sig")


def _get_default_cluster(plis_training_path: Path) -> str:
    plis = _read_tabular(plis_training_path)
    required = {"eclass", "manufacturer"}
    if not required.issubset(set(plis.columns)):
        raise ValueError(
            f"{plis_training_path} must contain columns {sorted(required)}. "
            f"Got: {list(plis.columns)}"
        )
    pairs = plis[["eclass", "manufacturer"]].fillna("")
    pairs = pairs[(pairs["eclass"] != "") & (pairs["manufacturer"] != "")]
    if pairs.empty:
        raise ValueError("No non-empty eclass/manufacturer pair found in training data.")
    top = pairs.value_counts().index[0]
    return f"{top[0]}|{top[1]}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write baseline submission CSV with legal_entity_id and cluster."
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
    args = parser.parse_args()

    output_path = Path(args.output)
    customer_test = _read_tabular(Path(args.customer_test))
    if "legal_entity_id" not in customer_test.columns:
        raise ValueError(
            f"{args.customer_test} must contain 'legal_entity_id'. "
            f"Got: {list(customer_test.columns)}"
        )

    cluster = _get_default_cluster(Path(args.plis_training))
    submission = pd.DataFrame(
        {
            "legal_entity_id": customer_test["legal_entity_id"].astype(str),
            "cluster": cluster,
        }
    )
    submission.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
