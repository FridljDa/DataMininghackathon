"""
Format portfolio into submission CSV (legal_entity_id, cluster) for Level 1.

Cluster = eclass. Warm buyers get their selected eclasses from portfolio.parquet;
cold-start buyers (in customer_test but not in portfolio) get a global default
eclass from plis_training history.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portfolio", required=True, help="Path to portfolio.parquet.")
    parser.add_argument(
        "--customer-test",
        required=True,
        dest="customer_test",
        help="Path to customer_test.csv (tab-separated).",
    )
    parser.add_argument(
        "--plis-training",
        required=True,
        dest="plis_training",
        help="Path to plis_training (split) for global default eclass.",
    )
    parser.add_argument("--output", required=True, help="Path to submission CSV.")
    args = parser.parse_args()

    portfolio_path = Path(args.portfolio)
    customer_test_path = Path(args.customer_test)
    plis_path = Path(args.plis_training)
    out_path = Path(args.output)

    portfolio = pd.read_parquet(portfolio_path)
    portfolio["legal_entity_id"] = portfolio["legal_entity_id"].astype(str)
    portfolio["cluster"] = portfolio["eclass"].astype(str)

    customer_test = pd.read_csv(customer_test_path, sep="\t", dtype=str)
    if "legal_entity_id" not in customer_test.columns:
        raise ValueError(
            f"customer_test must contain 'legal_entity_id'. Got: {list(customer_test.columns)}"
        )
    customer_test["legal_entity_id"] = customer_test["legal_entity_id"].astype(str)
    all_buyers = customer_test["legal_entity_id"].unique().tolist()
    warm_in_portfolio = set(portfolio["legal_entity_id"].unique())

    # Global default eclass (most frequent in training)
    plis = pd.read_csv(plis_path, sep="\t", usecols=["eclass"], low_memory=False)
    plis["eclass"] = plis["eclass"].astype(str).str.strip().replace("nan", "")
    plis = plis[plis["eclass"] != ""]
    default_eclass = plis["eclass"].mode().iloc[0] if len(plis) else ""

    rows = []
    for lid in all_buyers:
        if lid in warm_in_portfolio:
            for _, row in portfolio[portfolio["legal_entity_id"] == lid].iterrows():
                rows.append({"legal_entity_id": lid, "cluster": row["cluster"]})
        else:
            if default_eclass:
                rows.append({"legal_entity_id": lid, "cluster": default_eclass})

    submission = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)
    print(f"Wrote {len(submission)} submission rows to {out_path}")


if __name__ == "__main__":
    main()
