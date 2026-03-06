"""
Offline scoring: compare submission (legal_entity_id, cluster) to holdout plis_testing.

Uses Level-2 semantics: cluster = eclass|manufacturer. Ground truth pairs and spend
are derived from plis_testing; savings = savings_rate * spend on hit pairs,
fees = fixed_fee_eur * num_predictions, total_score = total_savings - total_fees.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _normalize_cluster(eclass: str | float, manufacturer: str | float) -> str:
    e = eclass if pd.isna(eclass) else str(eclass).strip()
    if e and e != "nan":
        try:
            e = str(int(float(e)))
        except (ValueError, TypeError):
            e = str(e).strip()
    else:
        e = ""
    m = manufacturer if pd.isna(manufacturer) else str(manufacturer).strip()
    if m == "nan":
        m = ""
    return f"{e}|{m}"


def _read_plis(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    for col in ("legal_entity_id", "orderdate", "eclass", "manufacturer", "quantityvalue", "vk_per_item"):
        if col not in df.columns:
            raise ValueError(f"plis_testing must contain '{col}'. Got: {list(df.columns)}")
    return df


def _build_truth_spend(plis: pd.DataFrame) -> tuple[dict[tuple[str, str], float], float]:
    """Aggregate spend per (legal_entity_id, cluster) from plis. Cluster = eclass|manufacturer."""
    plis = plis.copy()
    plis["cluster"] = plis.apply(
        lambda r: _normalize_cluster(r["eclass"], r["manufacturer"]), axis=1
    )
    plis["legal_entity_id"] = plis["legal_entity_id"].astype(str)
    q = pd.to_numeric(plis["quantityvalue"], errors="coerce").fillna(0)
    v = pd.to_numeric(plis["vk_per_item"], errors="coerce").fillna(0)
    plis["_spend"] = q * v
    # Only count pairs with non-empty cluster (both eclass and manufacturer)
    plis = plis[plis["cluster"].str.contains(r"\S+\|\S+", regex=True, na=False)]
    agg = plis.groupby(["legal_entity_id", "cluster"], as_index=False)["_spend"].sum()
    pair_spend = {}
    total_spend = 0.0
    for _, row in agg.iterrows():
        key = (row["legal_entity_id"], row["cluster"])
        pair_spend[key] = float(row["_spend"])
        total_spend += row["_spend"]
    return pair_spend, total_spend


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score submission against plis_testing holdout; write summary and details."
    )
    parser.add_argument("--submission", required=True, help="Submission CSV (legal_entity_id, cluster).")
    parser.add_argument("--plis-testing", required=True, dest="plis_testing", help="Holdout PLI TSV.")
    parser.add_argument("--summary", required=True, help="Output path for score_summary.csv.")
    parser.add_argument("--details", required=True, help="Output path for score_details.parquet.")
    parser.add_argument("--savings-rate", type=float, required=True, dest="savings_rate")
    parser.add_argument("--fixed-fee-eur", type=float, required=True, dest="fixed_fee_eur")
    parser.add_argument("--scoring-months", type=int, default=1, dest="scoring_months", help="Reserved for future use.")
    args = parser.parse_args()

    submission_path = Path(args.submission)
    plis_path = Path(args.plis_testing)
    summary_path = Path(args.summary)
    details_path = Path(args.details)

    sub = pd.read_csv(submission_path)
    for col in ("legal_entity_id", "cluster"):
        if col not in sub.columns:
            raise ValueError(f"Submission must contain '{col}'. Got: {list(sub.columns)}")
    sub["legal_entity_id"] = sub["legal_entity_id"].astype(str)
    sub["cluster"] = sub["cluster"].astype(str).str.strip()

    plis = _read_plis(plis_path)
    pair_spend, total_ground_spend = _build_truth_spend(plis)

    num_predictions = len(sub)
    hit_pairs: set[tuple[str, str]] = set()
    for _, row in sub.iterrows():
        key = (row["legal_entity_id"], row["cluster"])
        if key in pair_spend:
            hit_pairs.add(key)

    num_hits = len(hit_pairs)
    hit_spend_sum = sum(pair_spend[k] for k in hit_pairs)
    total_savings = args.savings_rate * hit_spend_sum
    total_fees = args.fixed_fee_eur * num_predictions
    total_score = total_savings - total_fees
    spend_capture_rate = hit_spend_sum / total_ground_spend if total_ground_spend > 0 else 0.0

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(
        [
            {
                "total_score": total_score,
                "total_savings": total_savings,
                "total_fees": total_fees,
                "num_hits": num_hits,
                "num_predictions": num_predictions,
                "spend_capture_rate": spend_capture_rate,
                "total_ground_spend": total_ground_spend,
            }
        ]
    )
    summary_df.to_csv(summary_path, index=False)

    details_list = []
    for _, row in sub.iterrows():
        key = (row["legal_entity_id"], row["cluster"])
        hit = key in pair_spend
        matched_spend = pair_spend.get(key, 0.0)
        details_list.append(
            {
                "legal_entity_id": row["legal_entity_id"],
                "cluster": row["cluster"],
                "hit": hit,
                "matched_spend": matched_spend,
            }
        )
    details_df = pd.DataFrame(details_list)
    details_path.parent.mkdir(parents=True, exist_ok=True)
    details_df.to_parquet(details_path, index=False)

    print(
        f"total_score={total_score:.2f} total_savings={total_savings:.2f} total_fees={total_fees:.2f} "
        f"num_hits={num_hits} num_predictions={num_predictions} spend_capture_rate={spend_capture_rate:.4f}"
    )


if __name__ == "__main__":
    main()
