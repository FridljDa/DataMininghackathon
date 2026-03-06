"""
Build data/02_meta/customer.csv from plis_training.csv.

One row per unique legal_entity_id in plis_training. Metadata columns
(estimated_number_employees, nace_code, secondary_nace_code) use the first
non-null value per customer (by orderdate). task is taken from customer_test
when present, else "none". Output has same schema as customer_test.csv.
"""

import argparse
from pathlib import Path

import pandas as pd


def first_nonnull(series: pd.Series):
    """First non-null value in series, or pd.NA if all null."""
    dropped = series.dropna()
    return dropped.iloc[0] if len(dropped) else pd.NA


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plis", required=True, help="Path to plis_training.csv")
    parser.add_argument("--customer-test", required=True, dest="customer_test", help="Path to customer_test.csv")
    parser.add_argument("--output", required=True, help="Path to output customer.csv")
    args = parser.parse_args()

    plis_path = Path(args.plis)
    customer_test_path = Path(args.customer_test)
    out_path = Path(args.output)

    meta_cols = ["estimated_number_employees", "nace_code", "secondary_nace_code"]
    plis_cols = ["legal_entity_id", "orderdate"] + meta_cols

    plis = pd.read_csv(plis_path, sep="\t", usecols=plis_cols, dtype={"legal_entity_id": "Int64"})
    plis["orderdate"] = pd.to_datetime(plis["orderdate"], format="%Y-%m-%d")
    plis = plis.sort_values(["orderdate", "legal_entity_id"])

    agg = {c: first_nonnull for c in meta_cols}
    customers = plis.groupby("legal_entity_id", as_index=False).agg(agg)

    test = pd.read_csv(customer_test_path, sep="\t", usecols=["legal_entity_id", "task"])
    customers = customers.merge(test[["legal_entity_id", "task"]], on="legal_entity_id", how="left")
    customers["task"] = customers["task"].fillna("none")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_cols = ["legal_entity_id", "estimated_number_employees", "nace_code", "secondary_nace_code", "task"]
    customers[out_cols].to_csv(out_path, sep="\t", index=False)


if __name__ == "__main__":
    main()
