"""
Generate trending eclass list from split training plis.

Writes CSV with a single required column: eclass (sorted by frequency desc).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plis", required=True, help="Path to split plis_training TSV.")
    parser.add_argument("--output", required=True, help="Path to output trending_classes CSV.")
    parser.add_argument(
        "--train-end",
        required=False,
        default="",
        dest="train_end",
        help="Optional max order date (YYYY-MM-DD) to restrict source rows.",
    )
    args = parser.parse_args()

    plis_path = Path(args.plis)
    out_path = Path(args.output)

    df = pd.read_csv(plis_path, sep="\t", low_memory=False)
    if "eclass" not in df.columns:
        raise ValueError(f"plis must contain 'eclass'. Got: {list(df.columns)}")

    if args.train_end:
        if "orderdate" not in df.columns:
            raise ValueError("plis must contain 'orderdate' when --train-end is provided.")
        df["orderdate"] = pd.to_datetime(df["orderdate"], format="%Y-%m-%d")
        df = df[df["orderdate"] <= pd.Timestamp(args.train_end)]

    df["eclass"] = df["eclass"].astype(str).str.strip().replace("nan", "")
    df = df[df["eclass"] != ""]

    trending = (
        df["eclass"]
        .value_counts(dropna=False)
        .rename_axis("eclass")
        .reset_index(name="n_orders")
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    trending[["eclass"]].to_csv(out_path, index=False)
    print(f"Wrote {len(trending)} trending eclasses to {out_path}")


if __name__ == "__main__":
    main()
