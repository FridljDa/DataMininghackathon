import argparse
import os

import pandas as pd


def read_csv_files(input_paths: list[str]) -> dict[str, pd.DataFrame]:
    """Read CSV files from the given paths. Paths are passed in; no implicit base path."""
    dataframes = {}
    for path in input_paths:
        name = os.path.basename(path)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, sep=None, engine="python")
                dataframes[name] = df
                print(f"Read {name}: {df.shape[0]} rows, {df.shape[1]} columns.")
            except Exception as e:
                print(f"Failed to read {name}: {e}")
        else:
            print(f"File not found: {path}")
    return dataframes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read CSV files. All paths are passed as arguments."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input CSV file paths (e.g. data/customer_test.csv data/nace_codes.csv).",
    )
    args = parser.parse_args()
    dfs = read_csv_files(args.inputs)
    for fname, dataframe in dfs.items():
        print(f"\n--- {fname} ---\n", dataframe.head())


if __name__ == "__main__":
    main()

