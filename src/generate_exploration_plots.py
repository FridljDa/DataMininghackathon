"""
Generate EDA plots from raw CSVs and customer metadata into data/04_plots.

Reads plis_training (tab), customer_test (tab), les_cs (comma), customer (tab).
Produces: seasonal purchase volume, violin of purchase value by task,
task distribution, cs vs task consistency heatmap, top-N NACE by task.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Output filenames must match config.yaml plots.files
PLOT_FILES = [
    "seasonal_purchase_volume.png",
    "purchase_value_by_task_violin.png",
    "task_distribution.png",
    "cs_task_consistency_heatmap.png",
    "nace_by_task_topn.png",
]


def task_group(series: pd.Series) -> pd.Series:
    """Map task to normalized task_group: none, warm, cold_start."""
    return series.str.strip().str.lower().replace(
        {"predict future": "warm", "cold start": "cold_start"}
    ).fillna("none")


def load_inputs(plis_path: Path, customer_test_path: Path, les_cs_path: Path, customer_meta_path: Path):
    """Load and minimally prepare dataframes."""
    plis_cols = ["orderdate", "legal_entity_id", "quantityvalue", "vk_per_item"]
    plis = pd.read_csv(plis_path, sep="\t", usecols=plis_cols)
    plis["orderdate"] = pd.to_datetime(plis["orderdate"], format="%Y-%m-%d")
    plis["value"] = plis["quantityvalue"] * plis["vk_per_item"]

    customer_test = pd.read_csv(customer_test_path, sep="\t", usecols=["legal_entity_id", "task"])
    les_cs = pd.read_csv(les_cs_path, sep=",", usecols=["legal_entity_id", "cs"])
    customer_meta = pd.read_csv(customer_meta_path, sep="\t", usecols=["legal_entity_id", "task", "nace_code"])

    customer_meta["task_group"] = task_group(customer_meta["task"])
    return plis, customer_test, les_cs, customer_meta


def plot_seasonal_purchase_volume(plis: pd.DataFrame, out_path: Path) -> None:
    """Monthly total quantity (purchase volume) over time."""
    monthly = plis.set_index("orderdate").resample("ME")["quantityvalue"].sum()
    fig, ax = plt.subplots(figsize=(10, 4))
    monthly.plot(ax=ax)
    ax.set_title("Seasonal purchase volume (monthly total quantity)")
    ax.set_ylabel("Quantity")
    ax.set_xlabel("Month")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()


def plot_purchase_value_by_task_violin(plis: pd.DataFrame, customer_meta: pd.DataFrame, out_path: Path) -> None:
    """Violin of per-customer total purchase value by task_group (none, warm, cold_start)."""
    cust_value = plis.groupby("legal_entity_id")["value"].sum().reset_index()
    cust_value = cust_value.merge(
        customer_meta[["legal_entity_id", "task_group"]],
        on="legal_entity_id",
        how="inner",
    )
    # Clip extreme values for readable violin (log scale or cap)
    cust_value["value_clip"] = cust_value["value"].clip(upper=cust_value["value"].quantile(0.99))
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.violinplot(data=cust_value, x="task_group", y="value_clip", ax=ax)
    ax.set_title("Purchase value by customer task (clipped at 99th percentile)")
    ax.set_ylabel("Total purchase value")
    ax.set_xlabel("Task")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()


def plot_task_distribution(customer_meta: pd.DataFrame, out_path: Path) -> None:
    """Bar chart of customer counts by task_group."""
    counts = customer_meta["task_group"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Task distribution")
    ax.set_ylabel("Number of customers")
    ax.set_xlabel("Task")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()


def plot_cs_task_consistency_heatmap(les_cs: pd.DataFrame, customer_meta: pd.DataFrame, out_path: Path) -> None:
    """Heatmap of cs (0/1) vs task_group to verify alignment."""
    joint = les_cs.merge(
        customer_meta[["legal_entity_id", "task_group"]],
        on="legal_entity_id",
        how="inner",
    )
    ct = pd.crosstab(joint["cs"].astype(str), joint["task_group"])
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(ct, annot=True, fmt="d", ax=ax, cmap="Blues")
    ax.set_title("cs vs task consistency (customer counts)")
    ax.set_xlabel("Task")
    ax.set_ylabel("cs")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()


def plot_nace_by_task_topn(customer_meta: pd.DataFrame, out_path: Path, top_n: int = 15) -> None:
    """Top-N NACE codes by count, stacked or grouped by task_group."""
    # Top NACE overall, then show counts by task within those
    meta = customer_meta.dropna(subset=["nace_code"])
    nace_counts = meta["nace_code"].value_counts()
    top_naces = nace_counts.head(top_n).index.tolist()
    subset = meta[meta["nace_code"].isin(top_naces)]
    ct = pd.crosstab(subset["nace_code"].astype(str), subset["task_group"])
    ct = ct.reindex([str(n) for n in top_naces])
    fig, ax = plt.subplots(figsize=(10, 5))
    ct.plot(kind="barh", stacked=True, ax=ax)
    ax.set_title(f"Top-{top_n} NACE codes by task")
    ax.set_ylabel("NACE code")
    ax.set_xlabel("Number of customers")
    ax.legend(title="Task")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plis", required=True, help="Path to plis_training.csv")
    parser.add_argument("--customer-test", required=True, dest="customer_test", help="Path to customer_test.csv")
    parser.add_argument("--les-cs", required=True, dest="les_cs", help="Path to les_cs.csv")
    parser.add_argument("--customer-meta", required=True, dest="customer_meta", help="Path to data/02_meta/customer.csv")
    parser.add_argument("--output-dir", required=True, dest="output_dir", help="Directory for plot files (e.g. data/04_plots)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plis, customer_test, les_cs, customer_meta = load_inputs(
        Path(args.plis),
        Path(args.customer_test),
        Path(args.les_cs),
        Path(args.customer_meta),
    )

    plot_seasonal_purchase_volume(plis, out_dir / "seasonal_purchase_volume.png")
    plot_purchase_value_by_task_violin(plis, customer_meta, out_dir / "purchase_value_by_task_violin.png")
    plot_task_distribution(customer_meta, out_dir / "task_distribution.png")
    plot_cs_task_consistency_heatmap(les_cs, customer_meta, out_dir / "cs_task_consistency_heatmap.png")
    plot_nace_by_task_topn(customer_meta, out_dir / "nace_by_task_topn.png")


if __name__ == "__main__":
    main()
