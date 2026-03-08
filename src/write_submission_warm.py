"""
Format portfolio into submission CSV (legal_entity_id, cluster) for Level 1 or 2.

Level 1: cluster = eclass only (e.g. 30020903).
Level 2: cluster = eclass|manufacturer (e.g. 30020903|Bissell).
Portfolio has (legal_entity_id, eclass) only; for Level 2, manufacturer is looked up
from plis_training (most frequent per (legal_entity_id, eclass)).
Warm buyers get their selected items from portfolio; cold-start buyers get
industry-informed defaults: top-K eclasses (Level 1) or eclass|manufacturer pairs (Level 2)
via NACE hierarchical collaborative filtering (when --nace-codes is set) or simple
NACE-2 frequency fallback otherwise.

When --scores is provided (path to scores.parquet from the modelling step), cold-start
rankings are derived from the model's own score_base values aggregated over warm buyers
in the same NACE industry (cross-task borrowing). This is strictly more informative than
raw PLI frequency because the model score already encodes recency, regularity, spend and
gap patterns. Falls back to PLI-frequency rankings when no positive-scored warm buyers
exist for a given NACE group.

Use --buyer-source customer-test for real submission; use --buyer-source customer-split
with --customer-split for offline scoring.

When --mode is warm_only or cold_only, only the corresponding rows are written (for use
in split DAG rules); merge_submission_parts then concatenates them into the final CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

# Hierarchy levels for cold-start lookup (most specific first)
NACE_LEVELS = ("nace_4", "nace_3", "nace_2", "section", "global")


def _plis_has_nace(plis_path: Path) -> bool:
    cols = pd.read_csv(plis_path, sep="\t", nrows=0).columns.tolist()
    return "nace_code" in cols


def _plis_has_manufacturer(plis_path: Path) -> bool:
    cols = pd.read_csv(plis_path, sep="\t", nrows=0).columns.tolist()
    return "manufacturer" in cols


def _resolve_nace_hierarchy(nace_codes_df: pd.DataFrame, code: str) -> dict[str, str]:
    """Return hierarchy keys for a NACE code: nace_4, nace_3, nace_2, section (and global is implicit)."""
    code = str(code).strip() if code and not pd.isna(code) else ""
    if not code:
        return {level: "" for level in NACE_LEVELS}
    row = nace_codes_df[nace_codes_df["nace_code"].astype(str).str.strip() == code]
    if len(row) == 0:
        return {
            "nace_4": code if len(code) == 4 else "",
            "nace_3": code[:3] if len(code) >= 3 else "",
            "nace_2": code[:2] if len(code) >= 2 else "",
            "section": "",
            "global": "",
        }
    r = row.iloc[0]
    n2 = str(r["nace_2digits"]).strip() if pd.notna(r["nace_2digits"]) else code[:2]
    n3 = str(r["nace_3digits"]).strip() if pd.notna(r["nace_3digits"]) else (code[:3] if len(code) >= 3 else "")
    sect = str(r["toplevel_section"]).strip() if pd.notna(r["toplevel_section"]) else ""
    n4 = code if len(code) == 4 else ""
    return {"nace_4": n4, "nace_3": n3, "nace_2": n2, "section": sect, "global": ""}


def _build_cold_start_rankings(
    plis: pd.DataFrame,
    nace_codes_df: pd.DataFrame,
    level: int,
    warm_ids: set[str],
    cold_start_top_k: int = 3,
    min_buyers_in_nace: int = 2,
) -> tuple[dict[tuple[str, str], list[str]], dict[tuple[str, str], list[str]]]:
    """
    Build per-NACE-level top-K eclass and cluster rankings from warm-buyer plis.
    Keys are (level_name, level_value) e.g. ("nace_2", "86").
    NACE-level rankings use lift = P(eclass|nace_group) / P(eclass|global); global fallback uses n_buyers.
    Returns (rankings_eclass, rankings_cluster) for Level 1 and Level 2.
    """
    plis_warm = plis[plis["legal_entity_id"].astype(str).isin(warm_ids)].copy()
    if len(plis_warm) == 0:
        return {}, {}

    plis_warm["_spend"] = (
        pd.to_numeric(plis_warm["quantityvalue"], errors="coerce").fillna(0)
        * pd.to_numeric(plis_warm["vk_per_item"], errors="coerce").fillna(0)
    )
    plis_warm = plis_warm[plis_warm["eclass"] != ""]
    if level == 2 and "manufacturer" in plis_warm.columns:
        plis_warm["manufacturer"] = plis_warm["manufacturer"].astype(str).str.strip().replace("nan", "")
        plis_warm = plis_warm[plis_warm["manufacturer"] != ""]
        plis_warm["cluster"] = plis_warm["eclass"] + "|" + plis_warm["manufacturer"]

    has_nace = "nace_code" in plis_warm.columns
    if has_nace:
        plis_warm["nace_code"] = plis_warm["nace_code"].astype(str).str.strip().replace("nan", "")
        nace_lookup = nace_codes_df[["nace_code", "nace_2digits", "nace_3digits", "toplevel_section"]].copy()
        nace_lookup["nace_code"] = nace_lookup["nace_code"].astype(str).str.strip()
        plis_warm = plis_warm.merge(nace_lookup, left_on="nace_code", right_on="nace_code", how="left")
        plis_warm["nace_2"] = plis_warm["nace_2digits"].fillna(plis_warm["nace_code"].str[:2])
        plis_warm["nace_3"] = plis_warm["nace_3digits"].fillna(plis_warm["nace_code"].str[:3])
        plis_warm["nace_4"] = plis_warm["nace_code"].where(plis_warm["nace_code"].str.len() == 4, "")
        plis_warm["section"] = plis_warm["toplevel_section"].fillna("")

    total_warm_buyers = plis_warm["legal_entity_id"].nunique()

    def global_rates_for(value_col: str) -> pd.Series:
        n_buyers_per = plis_warm.groupby(value_col, dropna=False)["legal_entity_id"].nunique()
        return (n_buyers_per / total_warm_buyers).rename("global_rate")

    def agg_and_rank(level_name: Optional[str], value_col: str) -> dict[tuple[str, str], list[str]]:
        key_cols = [level_name] if level_name and level_name in plis_warm.columns else []
        rankings: dict[tuple[str, str], list[str]] = {}
        if not key_cols:
            # Global fallback: rank by n_buyers (no lift)
            agg = (
                plis_warm.groupby(value_col, dropna=False)
                .agg(n_buyers=("legal_entity_id", "nunique"))
                .reset_index()
            )
            global_top = agg.nlargest(cold_start_top_k, "n_buyers")[value_col].tolist()
            if global_top:
                rankings[("global", "")] = global_top
            return rankings
        agg = (
            plis_warm.groupby(key_cols + [value_col], dropna=False)
            .agg(n_buyers=("legal_entity_id", "nunique"))
            .reset_index()
        )
        nace_value_buyers = plis_warm.groupby(key_cols[0])["legal_entity_id"].nunique()
        agg["nace_buyers"] = agg[key_cols[0]].map(nace_value_buyers)
        agg["group_rate"] = agg["n_buyers"] / agg["nace_buyers"]
        global_rate = global_rates_for(value_col)
        agg = agg.merge(global_rate.rename("global_rate"), left_on=value_col, right_index=True, how="left")
        agg["global_rate"] = agg["global_rate"].fillna(1.0 / total_warm_buyers)
        agg["lift"] = agg["group_rate"] / agg["global_rate"]
        agg = agg[agg["n_buyers"] >= min_buyers_in_nace]
        for grp_key, grp in agg.groupby(key_cols[0]):
            if grp_key == "" or (isinstance(grp_key, float) and pd.isna(grp_key)):
                continue
            level_val = str(grp_key).strip()
            top = grp.nlargest(cold_start_top_k, "lift")[value_col].tolist()
            if top and level_name:
                rankings[(level_name, level_val)] = top
        return rankings

    rankings_eclass = {}
    rankings_cluster = {}
    if has_nace:
        for level_name in ("nace_4", "nace_3", "nace_2", "section"):
            if level_name not in plis_warm.columns:
                continue
            r_e = agg_and_rank(level_name, "eclass")
            rankings_eclass.update(r_e)
            if level == 2 and "cluster" in plis_warm.columns:
                r_c = agg_and_rank(level_name, "cluster")
                rankings_cluster.update(r_c)
    if ("global", "") not in rankings_eclass:
        rankings_eclass.update(agg_and_rank(None, "eclass"))
    if level == 2 and ("global", "") not in rankings_cluster and "cluster" in plis_warm.columns:
        rankings_cluster.update(agg_and_rank(None, "cluster"))

    return rankings_eclass, rankings_cluster


def _build_cold_start_rankings_from_scores(
    scores: pd.DataFrame,
    nace_codes_df: pd.DataFrame,
    level: int,
    cold_start_top_k: int = 3,
) -> tuple[dict[tuple[str, str], list[str]], dict[tuple[str, str], list[str]]]:
    """
    Build per-NACE-level top-K eclass/cluster rankings from warm-buyer model scores
    (cross-task borrowing). Ranks eclasses by sum of positive score_base values across
    warm buyers in the same NACE group, using the same rankings dict shape as
    _build_cold_start_rankings so the rest of the cold-start path is unchanged.

    Requires scores to contain: legal_entity_id, eclass, score_base, nace_2.
    For level 2, also requires manufacturer (or cluster) column.
    Only rows with score_base > 0 are used (model-predicted recurring needs).
    """
    required = {"legal_entity_id", "eclass", "score_base", "nace_2"}
    if not required.issubset(scores.columns):
        return {}, {}

    positive = scores[scores["score_base"] > 0].copy()
    if positive.empty:
        return {}, {}

    positive["nace_2"] = positive["nace_2"].astype(str).str.strip()
    positive["nace_code"] = positive["nace_code"].astype(str).str.strip() if "nace_code" in positive.columns else positive["nace_2"]

    has_nace_ref = len(nace_codes_df) > 0

    def _nace_keys(code: str) -> dict[str, str]:
        if has_nace_ref:
            return _resolve_nace_hierarchy(nace_codes_df, code)
        n2 = code[:2] if len(code) >= 2 else code
        n3 = code[:3] if len(code) >= 3 else ""
        return {"nace_4": code if len(code) == 4 else "", "nace_3": n3, "nace_2": n2, "section": "", "global": ""}

    if level == 2 and "manufacturer" in positive.columns:
        positive["manufacturer"] = positive["manufacturer"].astype(str).str.strip().replace("nan", "")
        positive = positive[positive["manufacturer"] != ""]
        positive["cluster"] = positive["eclass"] + "|" + positive["manufacturer"]

    rankings_eclass: dict[tuple[str, str], list[str]] = {}
    rankings_cluster: dict[tuple[str, str], list[str]] = {}

    # Aggregate score_base per (nace_code, eclass) — sum rewards eclasses that many buyers scored highly
    eclass_agg = (
        positive.groupby(["nace_code", "eclass"])["score_base"]
        .sum()
        .reset_index(name="score_sum")
    )

    for nace_code, grp in eclass_agg.groupby("nace_code"):
        top_eclasses = grp.nlargest(cold_start_top_k, "score_sum")["eclass"].tolist()
        if not top_eclasses:
            continue
        keys = _nace_keys(str(nace_code))
        for level_name in ("nace_4", "nace_3", "nace_2"):
            val = keys.get(level_name, "")
            if val:
                key = (level_name, val)
                if key not in rankings_eclass:
                    rankings_eclass[key] = top_eclasses
        if keys.get("section"):
            key = ("section", keys["section"])
            if key not in rankings_eclass:
                rankings_eclass[key] = top_eclasses

    if ("global", "") not in rankings_eclass:
        global_top = (
            positive.groupby("eclass")["score_base"]
            .sum()
            .nlargest(cold_start_top_k)
            .index.tolist()
        )
        if global_top:
            rankings_eclass[("global", "")] = global_top

    if level == 2 and "cluster" in positive.columns:
        cluster_agg = (
            positive.groupby(["nace_code", "cluster"])["score_base"]
            .sum()
            .reset_index(name="score_sum")
        )
        for nace_code, grp in cluster_agg.groupby("nace_code"):
            top_clusters = grp.nlargest(cold_start_top_k, "score_sum")["cluster"].tolist()
            if not top_clusters:
                continue
            keys = _nace_keys(str(nace_code))
            for level_name in ("nace_4", "nace_3", "nace_2"):
                val = keys.get(level_name, "")
                if val:
                    key = (level_name, val)
                    if key not in rankings_cluster:
                        rankings_cluster[key] = top_clusters
            if keys.get("section"):
                key = ("section", keys["section"])
                if key not in rankings_cluster:
                    rankings_cluster[key] = top_clusters

        if ("global", "") not in rankings_cluster:
            global_top_c = (
                positive.groupby("cluster")["score_base"]
                .sum()
                .nlargest(cold_start_top_k)
                .index.tolist()
            )
            if global_top_c:
                rankings_cluster[("global", "")] = global_top_c

    return rankings_eclass, rankings_cluster


def _hierarchical_lookup(
    nace_code: str,
    secondary_nace: Optional[str],
    nace_codes_df: pd.DataFrame,
    rankings: dict[tuple[str, str], list[str]],
    top_k: int,
) -> list[str]:
    """Return up to top_k unique items by looking up hierarchy (nace_4 -> nace_3 -> nace_2 -> section -> global), then secondary_nace if needed."""
    if not rankings:
        return []
    hierarchy = _resolve_nace_hierarchy(nace_codes_df, nace_code)
    seen: set[str] = set()
    for level in NACE_LEVELS:
        if level == "global":
            key = ("global", "")
        else:
            val = hierarchy.get(level, "")
            if not val:
                continue
            key = (level, val)
        if key in rankings:
            for item in rankings[key]:
                if item and item not in seen:
                    seen.add(item)
                    if len(seen) >= top_k:
                        return list(seen)
    if len(seen) < top_k and secondary_nace:
        sec_h = _resolve_nace_hierarchy(nace_codes_df, secondary_nace)
        for level in NACE_LEVELS:
            if level == "global":
                key = ("global", "")
            else:
                val = sec_h.get(level, "")
                if not val:
                    continue
                key = (level, val)
            if key in rankings:
                for item in rankings[key]:
                    if item and item not in seen:
                        seen.add(item)
                        if len(seen) >= top_k:
                            return list(seen)
    return list(seen)


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
        "--nace-codes",
        dest="nace_codes",
        default=None,
        help="Path to nace_codes.csv (enables hierarchical NACE fallback for cold-start). If omitted, uses 2-digit NACE frequency fallback.",
    )
    parser.add_argument(
        "--level",
        type=int,
        choices=(1, 2),
        default=1,
        help="1 = cluster is eclass only; 2 = cluster is eclass|manufacturer. Default: 1.",
    )
    parser.add_argument(
        "--cold-start-top-k",
        type=int,
        default=10,
        dest="cold_start_top_k",
        help="Max number of eclasses/clusters to recommend per cold-start buyer (default: 10).",
    )
    parser.add_argument(
        "--min-buyers-in-nace",
        type=int,
        default=2,
        dest="min_buyers_in_nace",
        help="Min distinct buyers in NACE group for an eclass to be ranked (default: 2).",
    )
    parser.add_argument(
        "--scores",
        dest="scores",
        default=None,
        help="Path to scores.parquet from the modelling step (enables cross-task borrowing: "
        "cold-start rankings derived from model score_base aggregated over warm buyers in "
        "the same NACE industry, rather than raw PLI frequency).",
    )
    parser.add_argument(
        "--mode",
        choices=("all", "warm_only", "cold_only"),
        default="all",
        help="all = warm + cold rows (default). warm_only = only task-based warm buyers. cold_only = only cold-start buyers.",
    )
    parser.add_argument(
        "--warm-fallback-portfolio",
        dest="warm_fallback_portfolio",
        default=None,
        help="Optional path to a fallback portfolio.parquet. Warm buyers with zero rows in the primary portfolio get recommendations from this portfolio instead of cold-start.",
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

    # Optional warm fallback: for task-based warm buyers with zero primary rows, use this portfolio.
    fallback_portfolio: Optional[pd.DataFrame] = None
    if args.warm_fallback_portfolio and Path(args.warm_fallback_portfolio).exists():
        fallback_portfolio = pd.read_parquet(args.warm_fallback_portfolio)
        fallback_portfolio["legal_entity_id"] = fallback_portfolio["legal_entity_id"].astype(str)

    if level == 1:
        portfolio["cluster"] = portfolio["eclass"]
    else:
        # Level 2: use manufacturer from portfolio if present, else look up from plis
        if "manufacturer" in portfolio.columns:
            portfolio["manufacturer"] = portfolio["manufacturer"].astype(str).str.strip().replace("nan", "")
        else:
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

    if fallback_portfolio is not None:
        if level == 1:
            fallback_portfolio = fallback_portfolio.copy()
            fallback_portfolio["eclass"] = fallback_portfolio["eclass"].astype(str).str.strip()
            fallback_portfolio["cluster"] = fallback_portfolio["eclass"]
        else:
            fallback_portfolio = fallback_portfolio.copy()
            fallback_portfolio["eclass"] = fallback_portfolio["eclass"].astype(str).str.strip()
            if "manufacturer" not in fallback_portfolio.columns and plis_path.exists():
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
                    fallback_portfolio = fallback_portfolio.merge(top_manu, on=["legal_entity_id", "eclass"], how="left")
                else:
                    fallback_portfolio["manufacturer"] = ""
            fallback_portfolio["manufacturer"] = fallback_portfolio.get("manufacturer", pd.Series(dtype=object)).fillna("").astype(str)
            fallback_portfolio["cluster"] = fallback_portfolio["eclass"] + "|" + fallback_portfolio["manufacturer"]
            fallback_portfolio = fallback_portfolio[fallback_portfolio["cluster"].str.contains(r"\S+\|\S+", regex=True, na=False)]
        fallback_by_buyer = {lid: grp for lid, grp in fallback_portfolio.groupby("legal_entity_id")}
    else:
        fallback_by_buyer = {}

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

    task_norm = customers["task"].str.strip().str.lower() if "task" in customers.columns else pd.Series(dtype=object)
    warm_ids: set[str] = set()
    if "task" in customers.columns:
        warm_ids = set(
            customers.loc[
                task_norm.isin(["predict future", "testing"]),
                "legal_entity_id",
            ]
            .astype(str)
            .tolist()
        )

    need_cold = args.mode in ("all", "cold_only")
    use_nace_hierarchy = bool(need_cold and args.nace_codes and Path(args.nace_codes).exists())
    nace_codes_df = pd.DataFrame()
    if use_nace_hierarchy:
        nace_codes_df = pd.read_csv(Path(args.nace_codes), sep="\t", dtype=str, low_memory=False)
    rankings_eclass: dict[tuple[str, str], list[str]] = {}
    rankings_cluster: dict[tuple[str, str], list[str]] = {}
    default_eclass = ""
    default_clusters: list[str] = []
    plis: Optional[pd.DataFrame] = None

    use_scores_rankings = False
    if need_cold and args.scores and Path(args.scores).exists():
        scores_df = pd.read_parquet(args.scores)
        rankings_eclass, rankings_cluster = _build_cold_start_rankings_from_scores(
            scores_df,
            nace_codes_df,
            level,
            cold_start_top_k=args.cold_start_top_k,
        )
        if rankings_eclass:
            use_scores_rankings = True
            if ("global", "") in rankings_eclass and rankings_eclass[("global", "")]:
                default_eclass = rankings_eclass[("global", "")][0]
            if level == 2 and ("global", "") in rankings_cluster and rankings_cluster[("global", "")]:
                default_clusters = rankings_cluster[("global", "")]
            else:
                default_clusters = [default_eclass] if default_eclass else []

    if need_cold and not use_scores_rankings:
        if use_nace_hierarchy:
            plis_cols = ["legal_entity_id", "eclass", "quantityvalue", "vk_per_item"]
            if _plis_has_nace(plis_path):
                plis_cols.append("nace_code")
            if level == 2 and _plis_has_manufacturer(plis_path):
                plis_cols.append("manufacturer")
            plis_cold = pd.read_csv(plis_path, sep="\t", usecols=plis_cols, low_memory=False)
            plis_cold["legal_entity_id"] = plis_cold["legal_entity_id"].astype(str)
            rankings_eclass, rankings_cluster = _build_cold_start_rankings(
                plis_cold,
                nace_codes_df,
                level,
                warm_ids,
                cold_start_top_k=args.cold_start_top_k,
                min_buyers_in_nace=args.min_buyers_in_nace,
            )
            if ("global", "") in rankings_eclass and rankings_eclass[("global", "")]:
                default_eclass = rankings_eclass[("global", "")][0]
            if level == 2 and ("global", "") in rankings_cluster and rankings_cluster[("global", "")]:
                default_clusters = rankings_cluster[("global", "")]
            else:
                default_clusters = [default_eclass] if default_eclass else []
        else:
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
            if level == 2 and "cluster" in plis.columns and plis["cluster"].str.contains(r"\S+\|\S+", regex=True).any():
                default_clusters = plis["cluster"].value_counts().head(args.cold_start_top_k).index.tolist()
            else:
                default_clusters = [default_eclass] if default_eclass else []

    nace_to_eclasses: dict[str, list[str]] = {}
    nace_to_clusters: dict[str, list[str]] = {}
    if need_cold and use_scores_rankings and not use_nace_hierarchy:
        for (level_name, val), eclasses in rankings_eclass.items():
            if level_name == "nace_2" and val:
                nace_to_eclasses[val] = eclasses
        if level == 2:
            for (level_name, val), clusters in rankings_cluster.items():
                if level_name == "nace_2" and val:
                    nace_to_clusters[val] = clusters
    elif need_cold and not use_nace_hierarchy and _plis_has_nace(plis_path) and plis is not None:
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
                    top = grp.head(args.cold_start_top_k)["eclass"].tolist()
                    if top:
                        nace_to_eclasses[n2] = top
            else:
                freq = (
                    plis.groupby(["nace_2", "cluster"], as_index=False)
                    .size()
                    .sort_values(["nace_2", "size"], ascending=[True, False])
                )
                for n2, grp in freq.groupby("nace_2"):
                    top = grp.head(args.cold_start_top_k)["cluster"].tolist()
                    if top:
                        nace_to_clusters[n2] = top

    def eclasses_for_cold_buyer(
        nace_code: str,
        secondary_nace: Optional[str] = None,
    ) -> list[str]:
        if use_nace_hierarchy and len(nace_codes_df) > 0:
            out = _hierarchical_lookup(
                nace_code, secondary_nace, nace_codes_df, rankings_eclass, args.cold_start_top_k
            )
            return out if out else ([default_eclass] if default_eclass else [])
        if not nace_code or pd.isna(nace_code):
            return [default_eclass] if default_eclass else []
        n2 = str(nace_code).strip()[:2]
        if n2 in nace_to_eclasses:
            return nace_to_eclasses[n2]
        return [default_eclass] if default_eclass else []

    def clusters_for_cold_buyer(
        nace_code: str,
        secondary_nace: Optional[str] = None,
    ) -> list[str]:
        if use_nace_hierarchy and len(nace_codes_df) > 0:
            out = _hierarchical_lookup(
                nace_code, secondary_nace, nace_codes_df, rankings_cluster, args.cold_start_top_k
            )
            return out if out else default_clusters
        if not nace_code or pd.isna(nace_code):
            return default_clusters
        n2 = str(nace_code).strip()[:2]
        if n2 in nace_to_clusters:
            return nace_to_clusters[n2]
        return default_clusters

    rows = []
    cold_buyers_need_nace = args.buyer_source == "customer-test"
    customer_nace = None
    customer_secondary_nace = None
    if cold_buyers_need_nace and "nace_code" in customers.columns:
        customer_nace = customers.set_index("legal_entity_id")["nace_code"].astype(str)
    if cold_buyers_need_nace and "secondary_nace_code" in customers.columns:
        customer_secondary_nace = customers.set_index("legal_entity_id")["secondary_nace_code"]

    for lid in all_buyers:
        if lid in warm_in_portfolio:
            if args.mode == "cold_only":
                continue
            for _, row in portfolio[portfolio["legal_entity_id"] == lid].iterrows():
                rows.append({"legal_entity_id": lid, "cluster": row["cluster"]})
        else:
            if args.mode == "warm_only":
                continue
            nace_val = customer_nace.loc[lid] if customer_nace is not None and lid in customer_nace.index else ""
            sec_nace = (
                customer_secondary_nace.loc[lid]
                if customer_secondary_nace is not None and lid in customer_secondary_nace.index
                else None
            )
            if sec_nace is not None and (pd.isna(sec_nace) or str(sec_nace).strip() == ""):
                sec_nace = None
            if level == 1:
                items = eclasses_for_cold_buyer(nace_val, sec_nace)
            else:
                items = clusters_for_cold_buyer(nace_val, sec_nace)
            for c in items:
                if c:
                    rows.append({"legal_entity_id": lid, "cluster": c})

    submission = pd.DataFrame(rows).drop_duplicates(subset=["legal_entity_id", "cluster"])
    if submission.empty:
        submission = pd.DataFrame(columns=["legal_entity_id", "cluster"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)
    print(f"Wrote {len(submission)} submission rows to {out_path}")


if __name__ == "__main__":
    main()
