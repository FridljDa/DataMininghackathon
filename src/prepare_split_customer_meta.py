"""
Prepare split-specific customer metadata: copy data/02_meta/customer.csv and relabel
task=none customers to task=testing so their purchase-value distribution matches warm.

Total purchase value per customer is computed from plis_training using only rows with
orderdate < cutoff_date (pre-cutoff only), to avoid future-information leakage in
holdout selection. Selection uses log-scale stratified bins from the warm (predict future)
distribution; customers are sampled from task=none to fill each bin proportionally.
Output is written to data/03_customer/customer.csv for use by split_plis_training_validation.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _task_normalized(series: pd.Series) -> pd.Series:
    """Normalize task to lowercase; map 'predict future' -> 'warm'."""
    s = series.str.strip().str.lower()
    return s.replace({"predict future": "warm"})


def select_testing_match_warm(
    meta: pd.DataFrame,
    n_testing: int,
    random_seed: int,
    n_bins: int = 10,
) -> set[str]:
    """
    Select n_testing task=none customer IDs so their purchase value distribution
    matches the warm (predict future) distribution, using log-scale stratified bins.
    """
    task_norm = _task_normalized(meta["task"])
    none_df = (
        meta.loc[task_norm == "none", ["legal_entity_id", "value"]]
        .copy()
        .drop_duplicates(subset=["legal_entity_id"])
    )
    warm_df = (
        meta.loc[task_norm == "warm", ["legal_entity_id", "value"]]
        .copy()
        .drop_duplicates(subset=["legal_entity_id"])
    )

    if len(warm_df) == 0:
        raise ValueError(
            "No warm (predict future) customers in metadata; cannot match distribution."
        )
    if len(none_df) < n_testing:
        raise ValueError(
            f"Not enough task=none customers to select {n_testing}; found {len(none_df)}."
        )

    none_df["lv"] = np.log1p(none_df["value"].astype(float))
    warm_df["lv"] = np.log1p(warm_df["value"].astype(float))

    edges = np.quantile(warm_df["lv"], np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 2:
        raise ValueError(
            "Warm purchase distribution has no spread; cannot define bins for matching."
        )

    none_df["bin"] = pd.cut(
        none_df["lv"],
        bins=edges,
        include_lowest=True,
        labels=range(len(edges) - 1),
    )
    warm_df["bin"] = pd.cut(
        warm_df["lv"],
        bins=edges,
        include_lowest=True,
        labels=range(len(edges) - 1),
    )

    warm_counts = warm_df["bin"].value_counts(normalize=True).sort_index()
    target = (warm_counts * n_testing).round().astype(int)
    diff = n_testing - int(target.sum())
    if diff != 0:
        frac = warm_counts * n_testing - np.floor(warm_counts * n_testing)
        order = frac.sort_values(ascending=(diff < 0)).index.tolist()
        for i, b in enumerate(order):
            if i >= abs(diff):
                break
            if diff > 0:
                target[b] = target.get(b, 0) + 1
            elif target.get(b, 0) > 0:
                target[b] = target[b] - 1

    rng = np.random.default_rng(random_seed)
    selected_ids: list[str] = []
    for b in target.index:
        k = int(target[b])
        if k <= 0:
            continue
        cand = none_df.loc[none_df["bin"] == b, "legal_entity_id"].astype(str)
        cand = cand[~cand.isin(selected_ids)]
        if len(cand) == 0:
            continue
        take = min(k, len(cand))
        chosen = rng.choice(cand.to_numpy(), size=take, replace=False)
        selected_ids.extend(chosen.tolist())

    if len(selected_ids) < n_testing:
        remaining = none_df.loc[
            ~none_df["legal_entity_id"].astype(str).isin(selected_ids),
            "legal_entity_id",
        ].astype(str)
        fill_count = n_testing - len(selected_ids)
        fill = rng.choice(remaining.to_numpy(), size=fill_count, replace=False)
        selected_ids.extend(fill.tolist())

    return set(selected_ids[:n_testing])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--customer-meta", required=True, dest="customer_meta", help="Path to data/02_meta/customer.csv")
    parser.add_argument("--plis", required=True, help="Path to plis_training.csv")
    parser.add_argument("--output", required=True, help="Path to output customer.csv (e.g. data/03_customer/customer.csv)")
    parser.add_argument("--cutoff-date", required=True, dest="cutoff_date", help="Cutoff date (YYYY-MM-DD); only plis rows with orderdate < cutoff are used for value aggregation.")
    parser.add_argument("--n-testing", type=int, default=50, dest="n_testing", help="Number of customers to relabel to task=testing (warm-matched)")
    parser.add_argument("--random-seed", type=int, default=42, dest="random_seed", help="Random seed for reproducible within-bin selection")
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

    cutoff = pd.Timestamp(args.cutoff_date)
    plis_cols = ["legal_entity_id", "orderdate", "quantityvalue", "vk_per_item"]
    plis = pd.read_csv(plis_path, sep="\t", usecols=plis_cols, low_memory=False)
    for col in plis_cols:
        if col not in plis.columns:
            raise ValueError(f"Plis must contain '{col}'. Got: {list(plis.columns)}")

    plis["orderdate"] = pd.to_datetime(plis["orderdate"], format="%Y-%m-%d")
    plis = plis[plis["orderdate"] < cutoff]

    q = pd.to_numeric(plis["quantityvalue"], errors="coerce").fillna(0)
    v = pd.to_numeric(plis["vk_per_item"], errors="coerce").fillna(0)
    plis = plis.assign(value=q * v)
    total_value = plis.groupby("legal_entity_id", as_index=False)["value"].sum()
    total_value["legal_entity_id"] = total_value["legal_entity_id"].astype(str)

    meta["legal_entity_id"] = meta["legal_entity_id"].astype(str)
    meta = meta.merge(total_value, on="legal_entity_id", how="left")
    meta["value"] = meta["value"].fillna(0)

    selected_set = select_testing_match_warm(
        meta, n_testing=args.n_testing, random_seed=args.random_seed
    )

    none_mask = _task_normalized(meta["task"]) == "none"
    relabel = meta["legal_entity_id"].astype(str).isin(selected_set) & none_mask
    meta.loc[relabel, "task"] = "testing"
    meta = meta.drop(columns=["value"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {out_path}: {args.n_testing} customers relabeled to task=testing (warm-matched)")


if __name__ == "__main__":
    main()
