"""
Split plis_training.csv into training and testing by cutoff date.

Rows with orderdate >= cutoff_date are eligible for the test set. A configurable
fraction of those eligible rows is sampled (with a fixed seed) into the test
output; all other rows go to the training output. Schema and delimiter are
preserved.
"""

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to plis_training CSV/TSV")
    parser.add_argument("--train", required=True, help="Path to output training file")
    parser.add_argument("--test", required=True, help="Path to output test file")
    parser.add_argument("--cutoff-date", required=True, dest="cutoff_date", help="Cutoff date (YYYY-MM-DD); rows >= this are eligible for test")
    parser.add_argument("--test-fraction", type=float, required=True, dest="test_fraction", help="Fraction of eligible rows to put in test (0–1)")
    parser.add_argument("--random-seed", type=int, required=True, dest="random_seed", help="Random seed for reproducible sampling")
    args = parser.parse_args()

    if not (0 <= args.test_fraction <= 1):
        raise ValueError("test_fraction must be between 0 and 1")

    cutoff = pd.Timestamp(args.cutoff_date)
    input_path = Path(args.input)
    train_path = Path(args.train)
    test_path = Path(args.test)

    df = pd.read_csv(input_path, sep="\t", low_memory=False)
    if "orderdate" not in df.columns:
        raise ValueError("Input must contain an 'orderdate' column")

    df["orderdate"] = pd.to_datetime(df["orderdate"], format="%Y-%m-%d")
    eligible = df["orderdate"] >= cutoff
    eligible_df = df.loc[eligible]

    test_df = eligible_df.sample(frac=args.test_fraction, random_state=args.random_seed)
    train_df = df.drop(index=test_df.index)

    for frame in (train_df, test_df):
        frame["orderdate"] = frame["orderdate"].dt.strftime("%Y-%m-%d")

    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_path, sep="\t", index=False)
    test_df.to_csv(test_path, sep="\t", index=False)


if __name__ == "__main__":
    main()
