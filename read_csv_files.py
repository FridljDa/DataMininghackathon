import pandas as pd
import os

# List of CSV files to read (add more as needed)
csv_files = [
    'customer_test.csv',
    'nace_codes.csv',
    # 'features_per_sku.csv',  # Uncomment if available
    # 'plis_training.csv',     # Uncomment if available
]

def read_csv_files(base_path='.'):
    dataframes = {}
    for filename in csv_files:
        file_path = os.path.join(base_path, filename)
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, sep=None, engine='python')
                dataframes[filename] = df
                print(f"Read {filename}: {df.shape[0]} rows, {df.shape[1]} columns.")
            except Exception as e:
                print(f"Failed to read {filename}: {e}")
        else:
            print(f"File not found: {filename}")
    return dataframes

if __name__ == "__main__":
    # Set the base path to the script's directory
    base_path = os.path.dirname(os.path.abspath(__file__))
    dfs = read_csv_files(base_path)
    # Example: print the first few rows of each DataFrame
    for fname, df in dfs.items():
        print(f"\n--- {fname} ---\n", df.head())

