"""
Format portfolio into submission CSV (legal_entity_id, cluster) for Level 1 or 2.

Level 1: cluster = eclass only (e.g. 30020903).
Level 2: cluster = eclass|manufacturer (e.g. 30020903|Bissell).
Portfolio has (legal_entity_id, eclass) only; for Level 2, manufacturer is looked up
from plis_training (most frequent per (legal_entity_id, eclass)).
Warm buyers get their selected items from portfolio; cold-start buyers get
industry-informed defaults: top-K eclasses (Level 1) or eclass|manufacturer pairs (Level 2)
per NACE-2 from plis_training, or global default if NACE is missing.

Use --buyer-source customer-test for real submission; use --buyer-source customer-split
with --customer-split for offline scoring.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

NACE_TOP_K = 3


def _plis_has_nace(plis_path: Path) -> bool:
    cols = pd.read_csv(plis_path, sep="\t", nrows=0).columns.tolist()
    return "nace_code" in cols


def _plis_has_manufacturer(plis_path: Path) -> bool:
    cols = pd.read_csv(plis_path, sep="\t", nrows=0).columns.tolist()
    return "manufacturer" in cols


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portfolio", required=True, help="Path to portfolio.parquet.")
    parser.add_argument(
        "--buyer-source",
        required=True,
        dest="buyer_source",
        choices=("customer-test", "customer-split"),
        help="customer-test = use all buyers from customer_test (online submission). "
        "customer-split = use only task=testing buyers from customer-split file (offline scoring).",
    )
    parser.add_argument(
        "--customer-test",
        dest="customer_test",
        help="Path to customer_test.csv (tab-separated). Required when --buyer-source=customer-test.",
    )
    parser.add_argument(
        "--customer-split",
        dest="customer_split",
        help="Path to split customer CSV (e.g. data/04_customer/customer.csv). Required when --buyer-source=customer-split.",
    )
    parser.add_argument(
        "--plis-training",
        required=True,
        dest="plis_training",
        help="Path to plis_training (split) for defaults and Level 2 manufacturer lookup.",
    )
    parser.add_argument(
        "--level",
        type=int,
        choices=(1, 2),
        default=1,
        help="1 = cluster is eclass only; 2 = cluster is eclass|manufacturer. Default: 1.",
    )
    parser.add_argument("--output", required=True, help="Path to submission CSV.")
    args = parser.parse_args()

    if args.buyer_source == "customer-test" and not args.customer_test:
        parser.error("--customer-test is required when --buyer-source=customer-test")
    if args.buyer_source == "customer-split" and not args.customer_split:
        parser.error("--customer-split is required when --buyer-source=customer-split")

    portfolio_path = Path(args.portfolio)
    plis_path = Path(args.plis_training)
    out_path = Path(args.output)
    level = args.level

    portfolio = pd.read_parquet(portfolio_path)
    portfolio["legal_entity_id"] = portfolio["legal_entity_id"].astype(str)
    portfolio["eclass"] = portfolio["eclass"].astype(str).str.strip()
    warm_in_portfolio = set(portfolio["legal_entity_id"].unique())

    if level == 1:
        portfolio["cluster"] = portfolio["eclass"]
    else:
        # Level 2: look up most frequent manufacturer per (legal_entity_id, eclass) from plis
        plis_full = pd.read_csv(plis_path, sep="\t", usecols=["legal_entity_id", "eclass", "manufacturer"], low_memory=False)
        plis_full["legal_entity_id"] = plis_full["legal_entity_id"].astype(str)
        plis_full["eclass"] = plis_full["eclass"].astype(str).str.strip().replace("nan", "")
        plis_full["manufacturer"] = plis_full["manufacturer"].astype(str).str.strip().replace("nan", "")
        plis_full = plis_full[(plis_full["eclass"] != "") & (plis_full["manufacturer"] != "")]
        if len(plis_full) > 0:
            top_manu = (
                plis_full.groupby(["legal_entity_id", "eclass", "manufacturer"], as_index=False)
                .size()
                .sort_values("size", ascending=False)
                .drop_duplicates(subset=["legal_entity_id", "eclass"], keep="first")
                [["legal_entity_id", "eclass", "manufacturer"]]
            )
            portfolio = portfolio.merge(top_manu, on=["legal_entity_id", "eclass"], how="left")
        else:
            portfolio["manufacturer"] = ""
        portfolio["manufacturer"] = portfolio["manufacturer"].fillna("").astype(str)
        portfolio["cluster"] = portfolio["eclass"] + "|" + portfolio["manufacturer"]
        portfolio = portfolio[portfolio["cluster"].str.contains(r"\S+\|\S+", regex=True, na=False)]

    if args.buyer_source == "customer-test":
        customer_path = Path(args.customer_test)
        customers = pd.read_csv(customer_path, sep="\t", dtype=str)
        if "legal_entity_id" not in customers.columns:
            raise ValueError(
                f"customer_test must contain 'legal_entity_id'. Got: {list(customers.columns)}"
            )
        customers["legal_entity_id"] = customers["legal_entity_id"].astype(str)
        all_buyers = customers["legal_entity_id"].unique().tolist()
    else:
        customer_path = Path(args.customer_split)
        customers = pd.read_csv(customer_path, sep="\t", dtype=str)
        if "legal_entity_id" not in customers.columns or "task" not in customers.columns:
            raise ValueError(
                f"customer-split file must contain 'legal_entity_id' and 'task'. Got: {list(customers.columns)}"
            )
        customers["legal_entity_id"] = customers["legal_entity_id"].astype(str)
        task_norm = customers["task"].str.strip().str.lower()
        all_buyers = (
            customers.loc[task_norm == "testing", "legal_entity_id"].unique().tolist()
        )

    # Defaults: global mode + per-NACE-2 top-K by purchase frequency
    plis_cols = ["eclass"]
    if _plis_has_nace(plis_path):
        plis_cols.append("nace_code")
    if level == 2 and _plis_has_manufacturer(plis_path):
        plis_cols.append("manufacturer")
    plis = pd.read_csv(plis_path, sep="\t", usecols=plis_cols, low_memory=False)
    plis["eclass"] = plis["eclass"].astype(str).str.strip().replace("nan", "")
    plis = plis[plis["eclass"] != ""]
    if level == 2 and "manufacturer" in plis.columns:
        plis["manufacturer"] = plis["manufacturer"].astype(str).str.strip().replace("nan", "")
        plis = plis[plis["manufacturer"] != ""]
        plis["cluster"] = plis["eclass"] + "|" + plis["manufacturer"]
    else:
        plis["cluster"] = plis["eclass"]

    default_eclass = plis["eclass"].mode().iloc[0] if len(plis) else ""
    default_clusters: list[str] = []
    if level == 2 and "cluster" in plis.columns and plis["cluster"].str.contains(r"\S+\|\S+", regex=True).any():
        default_clusters = plis["cluster"].value_counts().head(NACE_TOP_K).index.tolist()
    else:
        default_clusters = [default_eclass] if default_eclass else []

    nace_to_eclasses: dict[str, list[str]] = {}
    nace_to_clusters: dict[str, list[str]] = {}
    if "nace_code" in plis.columns:
        plis["nace_2"] = plis["nace_code"].astype(str).str.strip().str[:2].replace("nan", "")
        plis = plis[plis["nace_2"] != ""]
        if len(plis) > 0:
            if level == 1:
                freq = (
                    plis.groupby(["nace_2", "eclass"], as_index=False)
                    .size()
                    .sort_values(["nace_2", "size"], ascending=[True, False])
                )
                for n2, grp in freq.groupby("nace_2"):
                    top = grp.head(NACE_TOP_K)["eclass"].tolist()
                    if top:
                        nace_to_eclasses[n2] = top
            else:
                freq = (
                    plis.groupby(["nace_2", "cluster"], as_index=False)
                    .size()
                    .sort_values(["nace_2", "size"], ascending=[True, False])
                )
                for n2, grp in freq.groupby("nace_2"):
                    top = grp.head(NACE_TOP_K)["cluster"].tolist()
                    if top:
                        nace_to_clusters[n2] = top

    def eclasses_for_cold_buyer(nace_code: str) -> list[str]:
        if not nace_code or pd.isna(nace_code):
            return [default_eclass] if default_eclass else []
        n2 = str(nace_code).strip()[:2]
        if n2 in nace_to_eclasses:
            return nace_to_eclasses[n2]
        return [default_eclass] if default_eclass else []

    def clusters_for_cold_buyer(nace_code: str) -> list[str]:
        if not nace_code or pd.isna(nace_code):
            return default_clusters
        n2 = str(nace_code).strip()[:2]
        if n2 in nace_to_clusters:
            return nace_to_clusters[n2]
        return default_clusters

    rows = []
    cold_buyers_need_nace = args.buyer_source == "customer-test"
    customer_nace = None
    if cold_buyers_need_nace and "nace_code" in customers.columns:
        customer_nace = customers.set_index("legal_entity_id")["nace_code"].astype(str)

    for lid in all_buyers:
        if lid in warm_in_portfolio:
            for _, row in portfolio[portfolio["legal_entity_id"] == lid].iterrows():
                rows.append({"legal_entity_id": lid, "cluster": row["cluster"]})
        else:
            if level == 1:
                if cold_buyers_need_nace and customer_nace is not None and lid in customer_nace.index:
                    items = eclasses_for_cold_buyer(customer_nace.loc[lid])
                else:
                    items = [default_eclass] if default_eclass else []
            else:
                if cold_buyers_need_nace and customer_nace is not None and lid in customer_nace.index:
                    items = clusters_for_cold_buyer(customer_nace.loc[lid])
                else:
                    items = default_clusters
            for c in items:
                if c:
                    rows.append({"legal_entity_id": lid, "cluster": c})

    submission = pd.DataFrame(rows).drop_duplicates(subset=["legal_entity_id", "cluster"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)
    print(f"Wrote {len(submission)} submission rows to {out_path}")


if __name__ == "__main__":
    main()
