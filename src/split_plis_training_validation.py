"""
Split plis_training.csv into training and testing by customer and cutoff date.

Uses customer metadata with task=testing as the test customer set. For those
customers only, rows with orderdate >= cutoff_date go to the test set; all
other rows go to the training set. Schema and delimiter are preserved.
"""

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to plis_training CSV/TSV")
    parser.add_argument("--customer-meta", required=True, dest="customer_meta", help="Path to customer metadata (must have legal_entity_id, task; task=testing marks test customers)")
    parser.add_argument("--train", required=True, help="Path to output training file")
    parser.add_argument("--test", required=True, help="Path to output test file")
    parser.add_argument("--cutoff-date", required=True, dest="cutoff_date", help="Cutoff date (YYYY-MM-DD); rows >= this from test customers go to test")
    args = parser.parse_args()

    cutoff = pd.Timestamp(args.cutoff_date)
    input_path = Path(args.input)
    customer_meta_path = Path(args.customer_meta)
    train_path = Path(args.train)
    test_path = Path(args.test)

    meta = pd.read_csv(customer_meta_path, sep="\t", dtype=str)
    for col in ("legal_entity_id", "task"):
        if col not in meta.columns:
            raise ValueError(f"Customer metadata must contain '{col}'. Got: {list(meta.columns)}")
    testing_customers = meta.loc[meta["task"].str.strip().str.lower() == "testing", "legal_entity_id"].astype(str).unique()
    selected_ids = set(testing_customers.tolist())

    # Load plis and ensure required columns
    df = pd.read_csv(input_path, sep="\t", low_memory=False)
    for col in ("legal_entity_id", "orderdate"):
        if col not in df.columns:
            raise ValueError(f"Input must contain '{col}'. Got: {list(df.columns)}")
    df["orderdate"] = pd.to_datetime(df["orderdate"], format="%Y-%m-%d")
    df["_legal_entity_id_str"] = df["legal_entity_id"].astype(str)

    # Test set: selected customers AND orderdate >= cutoff
    test_mask = df["_legal_entity_id_str"].isin(selected_ids) & (df["orderdate"] >= cutoff)
    test_df = df.loc[test_mask].drop(columns=["_legal_entity_id_str"])
    train_df = df.loc[~test_mask].drop(columns=["_legal_entity_id_str"])

    for frame in (train_df, test_df):
        frame["orderdate"] = frame["orderdate"].dt.strftime("%Y-%m-%d")

    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_path, sep="\t", index=False)
    test_df.to_csv(test_path, sep="\t", index=False)

    print(f"Training rows: {len(train_df)}, test rows: {len(test_df)} (test customers: {len(selected_ids)}, cutoff {args.cutoff_date})")


if __name__ == "__main__":
    main()
