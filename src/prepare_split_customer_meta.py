"""
Prepare split-specific customer metadata: copy data/02_meta/customer.csv and relabel
50 random task=none customers (with total purchase value >= 15k) as task=testing.

Total purchase value per customer = sum(quantityvalue * vk_per_item) from plis_training.
Output is written to data/03_customer/customer.csv for use by split_plis_training_validation.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MIN_PURCHASE_VALUE = 15_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--customer-meta", required=True, dest="customer_meta", help="Path to data/02_meta/customer.csv")
    parser.add_argument("--plis", required=True, help="Path to plis_training.csv")
    parser.add_argument("--output", required=True, help="Path to output customer.csv (e.g. data/03_customer/customer.csv)")
    parser.add_argument("--n-testing", type=int, default=50, dest="n_testing", help="Number of eligible customers to relabel to task=testing")
    parser.add_argument("--random-seed", type=int, default=42, dest="random_seed", help="Random seed for reproducible selection")
    args = parser.parse_args()

    if args.n_testing < 1:
        raise ValueError("n_testing must be at least 1")

    meta_path = Path(args.customer_meta)
    plis_path = Path(args.plis)
    out_path = Path(args.output)

    meta = pd.read_csv(meta_path, sep="\t", dtype=str)
    for col in ("legal_entity_id", "task"):
        if col not in meta.columns:
            raise ValueError(f"Customer metadata must contain '{col}'. Got: {list(meta.columns)}")

    plis_cols = ["legal_entity_id", "quantityvalue", "vk_per_item"]
    plis = pd.read_csv(plis_path, sep="\t", usecols=plis_cols, low_memory=False)
    for col in plis_cols:
        if col not in plis.columns:
            raise ValueError(f"Plis must contain '{col}'. Got: {list(plis.columns)}")

    q = pd.to_numeric(plis["quantityvalue"], errors="coerce").fillna(0)
    v = pd.to_numeric(plis["vk_per_item"], errors="coerce").fillna(0)
    plis = plis.assign(value=q * v)
    total_value = plis.groupby("legal_entity_id", as_index=False)["value"].sum()
    total_value["legal_entity_id"] = total_value["legal_entity_id"].astype(str)

    meta["legal_entity_id"] = meta["legal_entity_id"].astype(str)
    meta = meta.merge(total_value, on="legal_entity_id", how="left")
    meta["value"] = meta["value"].fillna(0)

    none_mask = meta["task"].str.strip().str.lower() == "none"
    eligible_mask = none_mask & (meta["value"] >= MIN_PURCHASE_VALUE)
    eligible_ids = meta.loc[eligible_mask, "legal_entity_id"].astype(str).unique()

    if len(eligible_ids) < args.n_testing:
        raise ValueError(
            f"Not enough eligible customers (task=none and total purchase value >= {MIN_PURCHASE_VALUE}): "
            f"found {len(eligible_ids)}, need {args.n_testing}"
        )

    rng = np.random.default_rng(args.random_seed)
    selected = rng.choice(eligible_ids, size=args.n_testing, replace=False)
    selected_set = set(selected.tolist())

    relabel = meta["legal_entity_id"].astype(str).isin(selected_set) & none_mask
    meta.loc[relabel, "task"] = "testing"
    meta = meta.drop(columns=["value"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {out_path}: {args.n_testing} customers relabeled to task=testing (eligible: {len(eligible_ids)})")


if __name__ == "__main__":
    main()
