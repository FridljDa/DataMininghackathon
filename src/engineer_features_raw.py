"""
Raw feature assembly: keys from candidates + aggregates from PLIs + customer context + top-K SKU attributes.

Reads key-only candidates parquet, PLIs (for aggregates), customer metadata, and features_per_sku.
Computes pair-level aggregates (n_orders, historical_purchase_value_total, orderdate_min/max, orderdates_str)
from PLIs, merges customer context (estimated_number_employees, nace_code, secondary_nace_code), and
top-K SKU attribute columns. Writes to data/08_features_raw/{mode}/level{level}/features_raw.parquet.
Level 2 includes manufacturer in key columns. Downstream engineer_features_derived adds computed features.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

# Candidates are key-only per level
KEY_COLUMNS_L1 = ["legal_entity_id", "eclass"]
KEY_COLUMNS_L2 = ["legal_entity_id", "eclass", "manufacturer"]

# Aggregate columns we compute from PLIs (same for both levels)
AGGREGATE_COLUMNS = [
    "n_orders",
    "historical_purchase_value_total",
    "orderdate_min",
    "orderdate_max",
    "orderdates_str",
]

FEATURES_PER_SKU_COLS = ("sku", "key", "fvalue", "fvalue_set")
PLIS_REQUIRED = ("legal_entity_id", "orderdate", "eclass", "sku", "quantityvalue", "vk_per_item")
CUSTOMER_CONTEXT_COLS = ("estimated_number_employees", "nace_code", "secondary_nace_code")


def _sanitize(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]", "_", str(name))
    return re.sub(r"_+", "_", s).strip("_") or "unknown"


def _read_plis(path: Path, train_end: pd.Timestamp, require_manufacturer: bool) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    for col in PLIS_REQUIRED:
        if col not in df.columns:
            raise ValueError(f"plis must contain '{col}'. Got: {list(df.columns)}")
    if require_manufacturer and "manufacturer" not in df.columns:
        raise ValueError("plis must contain 'manufacturer' for level 2.")
    df["orderdate"] = pd.to_datetime(df["orderdate"], format="%Y-%m-%d")
    df = df[df["orderdate"] <= train_end]
    df["legal_entity_id"] = df["legal_entity_id"].astype(str)
    df["eclass"] = df["eclass"].astype(str).str.strip().replace("nan", "")
    df = df[df["eclass"] != ""]
    df["sku"] = df["sku"].astype(str).str.strip()
    if require_manufacturer:
        df["manufacturer"] = df["manufacturer"].astype(str).str.strip().replace("nan", "")
        df = df[df["manufacturer"] != ""]
    q = pd.to_numeric(df["quantityvalue"], errors="coerce").fillna(0)
    v = pd.to_numeric(df["vk_per_item"], errors="coerce").fillna(0)
    df["_spend"] = q * v
    return df


def _read_customer(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    for col in ["legal_entity_id"] + list(CUSTOMER_CONTEXT_COLS):
        if col not in df.columns:
            raise ValueError(f"customer must contain '{col}'. Got: {list(df.columns)}")
    df["legal_entity_id"] = df["legal_entity_id"].astype(str)
    return df


def _read_features_per_sku_chunked(
    path: Path,
    relevant_skus: set[str],
    top_k_keys: int,
    top_k_values_per_key: int,
    chunksize: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Read CSV in chunks; keep rows with sku in relevant_skus. Return (sku_attrs with top-K key/values only, list of column names)."""
    # Fast opt-out: allow config to disable SKU-attribute feature generation.
    if top_k_keys <= 0 or top_k_values_per_key <= 0:
        return pd.DataFrame(columns=["sku", "key", "value"]), []

    required = list(FEATURES_PER_SKU_COLS)
    chunks = []
    key_counts: dict[str, int] = {}
    key_value_counts: dict[tuple[str, str], int] = {}

    with pd.read_csv(path, sep="\t", chunksize=chunksize, low_memory=False) as reader:
        for chunk in reader:
            for c in required:
                if c not in chunk.columns:
                    raise ValueError(f"features_per_sku must contain '{c}'. Got: {list(chunk.columns)}")
            chunk = chunk[chunk["sku"].astype(str).str.strip().isin(relevant_skus)]
            if chunk.empty:
                continue
            chunk = chunk[["sku", "key", "fvalue_set"]].copy()
            chunk["key"] = chunk["key"].astype(str).str.strip()
            chunk["value"] = chunk["fvalue_set"].astype(str).str.strip().replace("nan", "")
            chunk = chunk[chunk["key"].astype(bool) & chunk["value"].astype(bool)]
            chunk = chunk.drop(columns=["fvalue_set"])
            chunks.append(chunk)
            for _, row in chunk.iterrows():
                k, v = str(row["key"]).strip(), str(row["value"]).strip()
                if k and v:
                    key_counts[k] = key_counts.get(k, 0) + 1
                    key_value_counts[(k, v)] = key_value_counts.get((k, v), 0) + 1

    if not chunks:
        return pd.DataFrame(columns=["sku", "key", "value"]), []

    sku_attrs = pd.concat(chunks, ignore_index=True)
    if sku_attrs.empty:
        return sku_attrs, []

    # Top-K keys by frequency
    top_keys = sorted(key_counts.keys(), key=lambda x: -key_counts[x])[:top_k_keys]
    if not top_keys:
        return sku_attrs, []

    # Per-key top-K values
    allowed_pairs = set()
    for k in top_keys:
        pairs_for_k = [(kk, v) for (kk, v) in key_value_counts if kk == k]
        pairs_for_k.sort(key=lambda p: -key_value_counts[p])
        for p in pairs_for_k[:top_k_values_per_key]:
            allowed_pairs.add(p)

    sku_attrs["_pair"] = list(zip(sku_attrs["key"], sku_attrs["value"]))
    sku_attrs = sku_attrs[sku_attrs["_pair"].isin(allowed_pairs)].drop(columns=["_pair"])
    column_names = ["sku_attr_" + _sanitize(k) + "_" + _sanitize(v) for (k, v) in sorted(allowed_pairs, key=lambda p: (str(p[0]), str(p[1])))]
    return sku_attrs, column_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-raw", required=True, dest="candidates_raw", help="Path to raw candidates parquet.")
    parser.add_argument("--plis", required=True, help="Path to plis_training (split) TSV.")
    parser.add_argument("--customer", required=True, help="Path to customer metadata TSV (legal_entity_id + context columns).")
    parser.add_argument("--features-per-sku", required=True, dest="features_per_sku", help="Path to features_per_sku.csv.")
    parser.add_argument("--output", required=True, help="Path to output features_raw parquet.")
    parser.add_argument("--train-end", required=True, dest="train_end", help="Last date of train period (YYYY-MM-DD).")
    parser.add_argument("--top-k-keys", type=int, default=20, dest="top_k_keys", help="Number of top attribute keys to keep.")
    parser.add_argument("--top-k-values-per-key", type=int, default=10, dest="top_k_values_per_key", help="Top values per key.")
    parser.add_argument("--chunksize", type=int, default=200_000, help="Chunk size for reading features_per_sku.")
    parser.add_argument(
        "--level",
        required=True,
        choices=("1", "2"),
        help="Level 1 or 2; level 2 requires manufacturer in candidates.",
    )
    args = parser.parse_args()

    raw_path = Path(args.candidates_raw)
    plis_path = Path(args.plis)
    customer_path = Path(args.customer)
    features_per_sku_path = Path(args.features_per_sku)
    out_path = Path(args.output)
    train_end = pd.Timestamp(args.train_end)
    level = int(args.level)
    group_cols = KEY_COLUMNS_L2 if level == 2 else KEY_COLUMNS_L1
    key_columns = KEY_COLUMNS_L2 if level == 2 else KEY_COLUMNS_L1

    candidates = pd.read_parquet(raw_path)
    for col in key_columns:
        if col not in candidates.columns:
            raise ValueError(f"candidates_raw must contain '{col}'. Got: {list(candidates.columns)}")
    candidates = candidates[group_cols].drop_duplicates().reset_index(drop=True)

    plis = _read_plis(plis_path, train_end, require_manufacturer=(level == 2))
    plis_candidate = plis.merge(candidates, on=group_cols, how="inner")
    relevant_skus = set(plis_candidate["sku"].unique())

    # Aggregates from PLIs (pair-level)
    # avg_price_per_unit = total spend / total quantity (quantity-weighted average price, leakage-free)
    agg = (
        plis_candidate.groupby(group_cols)
        .agg(
            n_orders=("_spend", "count"),
            historical_purchase_value_total=("_spend", "sum"),
            orderdate_min=("orderdate", "min"),
            orderdate_max=("orderdate", "max"),
            orderdates=("orderdate", lambda x: x.dt.to_period("M").unique().tolist()),
            avg_price_per_unit=(
                "_spend",
                lambda x: x.sum()
                / max(
                    1e-9,
                    pd.to_numeric(
                        plis_candidate.loc[x.index, "quantityvalue"], errors="coerce"
                    ).fillna(0).sum(),
                ),
            ),
        )
        .reset_index()
    )
    agg["orderdates_str"] = agg["orderdates"].apply(
        lambda periods: [f"{p.year:04d}-{p.month:02d}" for p in periods]
    )
    agg = agg.drop(columns=["orderdates"], errors="ignore")

    out = candidates.merge(agg, on=group_cols, how="left")
    out["n_orders"] = out["n_orders"].fillna(0).astype(int)
    out["historical_purchase_value_total"] = out["historical_purchase_value_total"].fillna(0.0)
    out["avg_price_per_unit"] = out["avg_price_per_unit"].fillna(0.0)
    out["orderdate_min"] = out["orderdate_min"].fillna(pd.NaT)
    out["orderdate_max"] = out["orderdate_max"].fillna(pd.NaT)
    out["orderdates_str"] = out["orderdates_str"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    # Customer context (one row per legal_entity_id)
    customer = _read_customer(customer_path)
    cust_sub = customer[["legal_entity_id"] + list(CUSTOMER_CONTEXT_COLS)].drop_duplicates(
        subset="legal_entity_id", keep="first"
    )
    out = out.merge(cust_sub, on="legal_entity_id", how="left")

    sku_attrs, attr_column_names = _read_features_per_sku_chunked(
        features_per_sku_path,
        relevant_skus,
        top_k_keys=args.top_k_keys,
        top_k_values_per_key=args.top_k_values_per_key,
        chunksize=args.chunksize,
    )

    if not attr_column_names or sku_attrs.empty:
        for col in attr_column_names:
            out[col] = 0
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_path, index=False)
        print(f"Wrote {len(out)} raw feature rows (no SKU attributes) to {out_path}")
        return

    # Entity key -> list of SKUs
    entity_skus = (
        plis_candidate[group_cols + ["sku"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    merged = entity_skus.merge(sku_attrs, on="sku", how="inner")
    counts = merged.groupby(group_cols + ["key", "value"]).size().reset_index(name="count")
    pivot = counts.pivot_table(
        index=group_cols,
        columns=["key", "value"],
        values="count",
        fill_value=0,
        aggfunc="sum",
    )
    pivot.columns = ["sku_attr_" + _sanitize(k) + "_" + _sanitize(v) for (k, v) in pivot.columns]
    pivot = pivot.reset_index()
    for col in attr_column_names:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[[c for c in group_cols + attr_column_names if c in pivot.columns]]

    out = out.merge(pivot, on=group_cols, how="left")
    for c in attr_column_names:
        if c in out.columns:
            out[c] = out[c].fillna(0).astype(int)
        else:
            out[c] = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"Wrote {len(out)} raw feature rows ({len(attr_column_names)} SKU attr columns) to {out_path}")


if __name__ == "__main__":
    main()
