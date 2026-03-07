"""Unit tests for candidate generation: lookback thresholds and level-specific behavior."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
import pytest


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=30, check=False)


@pytest.fixture
def project_root() -> Path:
    """Project root (parent of src/ and data/)."""
    return Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Level 1: threshold filtering
# ---------------------------------------------------------------------------


def test_level1_min_order_frequency_and_min_lookback_spend_filter(
    tmp_path: Path, project_root: Path
) -> None:
    """With min_order_frequency=2 and min_lookback_spend=100, only pairs passing both appear in seen; trending union still adds hot x trending."""
    train_end = "2025-06-30"
    # b1,e1: 2 orders in lookback, 150 EUR -> pass. b1,e2: 1 order, 50 EUR -> fail. b2,e1: 2 orders, 200 EUR -> pass.
    plis = tmp_path / "plis.tsv"
    plis.write_text(
        "orderdate\tlegal_entity_id\teclass\tquantityvalue\tvk_per_item\n"
        "2025-01-01\tb1\te1\t1\t75\n"
        "2025-02-01\tb1\te1\t1\t75\n"
        "2025-03-01\tb1\te2\t1\t50\n"
        "2025-04-01\tb2\te1\t2\t100\n"
        "2025-05-01\tb2\te1\t2\t100\n",
        encoding="utf-8",
    )
    customer = tmp_path / "customer.tsv"
    customer.write_text(
        "legal_entity_id\ttask\n"
        "b1\tpredict future\n"
        "b2\tpredict future\n",
        encoding="utf-8",
    )
    trending = tmp_path / "trending.csv"
    trending.write_text("eclass\ne1\ne2\ne3\n", encoding="utf-8")
    out = tmp_path / "candidates.parquet"

    result = _run(
        [
            "uv", "run", "src/generate_candidates.py",
            "--plis", str(plis),
            "--customer", str(customer),
            "--trending-classes", str(trending),
            "--output", str(out),
            "--train-end", train_end,
            "--level", "1",
            "--lookback-months", "12",
            "--min-order-frequency", "2",
            "--min-lookback-spend", "100",
        ],
        cwd=project_root,
    )
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"

    df = pd.read_parquet(out)
    assert set(df.columns) == {"legal_entity_id", "eclass"}
    # Seen passing: (b1,e1), (b2,e1). Trending cross: b1,b2 x e1,e2,e3 -> 6. Union dedup: (b1,e1),(b1,e2),(b1,e3),(b2,e1),(b2,e2),(b2,e3) = 6
    assert len(df) == 6
    assert set(df["legal_entity_id"]) == {"b1", "b2"}
    assert set(df["eclass"]) == {"e1", "e2", "e3"}


def test_level1_permissive_thresholds_preserve_broad_inclusion(
    tmp_path: Path, project_root: Path
) -> None:
    """min_order_frequency=1 and min_lookback_spend=0 keep all pairs with at least one order in lookback."""
    train_end = "2025-06-30"
    plis = tmp_path / "plis.tsv"
    plis.write_text(
        "orderdate\tlegal_entity_id\teclass\tquantityvalue\tvk_per_item\n"
        "2025-01-01\tb1\te1\t1\t10\n"
        "2025-02-01\tb1\te2\t1\t1\n"
        "2025-03-01\tb2\te1\t1\t1\n",
        encoding="utf-8",
    )
    customer = tmp_path / "customer.tsv"
    customer.write_text(
        "legal_entity_id\ttask\n"
        "b1\tpredict future\n"
        "b2\tpredict future\n",
        encoding="utf-8",
    )
    trending = tmp_path / "trending.csv"
    trending.write_text("eclass\ne1\ne2\n", encoding="utf-8")
    out = tmp_path / "candidates.parquet"

    result = _run(
        [
            "uv", "run", "src/generate_candidates.py",
            "--plis", str(plis),
            "--customer", str(customer),
            "--trending-classes", str(trending),
            "--output", str(out),
            "--train-end", train_end,
            "--level", "1",
            "--lookback-months", "12",
            "--min-order-frequency", "1",
            "--min-lookback-spend", "0",
        ],
        cwd=project_root,
    )
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"

    df = pd.read_parquet(out)
    # Seen: (b1,e1),(b1,e2),(b2,e1). Trending: b1,b2 x e1,e2. Union: (b1,e1),(b1,e2),(b2,e1),(b2,e2) = 4
    assert len(df) == 4
    assert set(df["legal_entity_id"]) == {"b1", "b2"}
    assert set(df["eclass"]) == {"e1", "e2"}


def test_level1_lookback_window_boundary(tmp_path: Path, project_root: Path) -> None:
    """Only orders inside [train_end - lookback_months, train_end] count toward eligibility."""
    train_end = "2025-06-30"
    # Order at 2024-01-01 (18 months back) and 2025-05-01 (2 months back). lookback_months=12 -> only 2025-05-01 in window -> 1 order for (b1,e1).
    plis = tmp_path / "plis.tsv"
    plis.write_text(
        "orderdate\tlegal_entity_id\teclass\tquantityvalue\tvk_per_item\n"
        "2024-01-01\tb1\te1\t1\t100\n"
        "2025-05-01\tb1\te1\t1\t100\n",
        encoding="utf-8",
    )
    customer = tmp_path / "customer.tsv"
    customer.write_text(
        "legal_entity_id\ttask\n"
        "b1\tpredict future\n",
        encoding="utf-8",
    )
    trending = tmp_path / "trending.csv"
    trending.write_text("eclass\ne1\n", encoding="utf-8")
    out = tmp_path / "candidates.parquet"

    result = _run(
        [
            "uv", "run", "src/generate_candidates.py",
            "--plis", str(plis),
            "--customer", str(customer),
            "--trending-classes", str(trending),
            "--output", str(out),
            "--train-end", train_end,
            "--level", "1",
            "--lookback-months", "12",
            "--min-order-frequency", "2",
            "--min-lookback-spend", "0",
        ],
        cwd=project_root,
    )
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"

    df = pd.read_parquet(out)
    # (b1,e1) has only 1 order in 12-month lookback -> excluded from seen. Union = trending cross only: (b1,e1) -> 1 row
    assert len(df) == 1
    assert df.iloc[0]["legal_entity_id"] == "b1"
    assert df.iloc[0]["eclass"] == "e1"


# ---------------------------------------------------------------------------
# Level 2: threshold filtering, no trending cross
# ---------------------------------------------------------------------------


def test_level2_min_order_frequency_and_min_lookback_spend_filter(
    tmp_path: Path, project_root: Path
) -> None:
    """Level 2 filters (legal_entity_id, eclass, manufacturer) by lookback thresholds; no trending cross."""
    train_end = "2025-06-30"
    plis = tmp_path / "plis.tsv"
    plis.write_text(
        "orderdate\tlegal_entity_id\teclass\tmanufacturer\tquantityvalue\tvk_per_item\n"
        "2025-01-01\tb1\te1\tm1\t1\t60\n"
        "2025-02-01\tb1\te1\tm1\t1\t60\n"
        "2025-03-01\tb1\te1\tm2\t1\t30\n"
        "2025-04-01\tb1\te2\tm1\t1\t150\n",
        encoding="utf-8",
    )
    customer = tmp_path / "customer.tsv"
    customer.write_text(
        "legal_entity_id\ttask\n"
        "b1\tpredict future\n",
        encoding="utf-8",
    )
    trending = tmp_path / "trending.csv"
    trending.write_text("eclass\ne1\n", encoding="utf-8")
    out = tmp_path / "candidates.parquet"

    result = _run(
        [
            "uv", "run", "src/generate_candidates.py",
            "--plis", str(plis),
            "--customer", str(customer),
            "--trending-classes", str(trending),
            "--output", str(out),
            "--train-end", train_end,
            "--level", "2",
            "--lookback-months", "12",
            "--min-order-frequency", "2",
            "--min-lookback-spend", "100",
        ],
        cwd=project_root,
    )
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"

    df = pd.read_parquet(out)
    assert set(df.columns) == {"legal_entity_id", "eclass", "manufacturer"}
    # (b1,e1,m1): 2 orders, 120 EUR -> pass. (b1,e1,m2): 1 order -> fail. (b1,e2,m1): 1 order -> fail.
    assert len(df) == 1
    assert df.iloc[0]["legal_entity_id"] == "b1"
    assert df.iloc[0]["eclass"] == "e1"
    assert df.iloc[0]["manufacturer"] == "m1"


def test_level2_permissive_thresholds_all_triplets_included(
    tmp_path: Path, project_root: Path
) -> None:
    """Level 2 with min_order_frequency=1 and min_lookback_spend=0 includes all (buyer, eclass, manufacturer) with ≥1 order in lookback."""
    train_end = "2025-06-30"
    plis = tmp_path / "plis.tsv"
    plis.write_text(
        "orderdate\tlegal_entity_id\teclass\tmanufacturer\tquantityvalue\tvk_per_item\n"
        "2025-01-01\tb1\te1\tm1\t1\t1\n"
        "2025-02-01\tb1\te1\tm2\t1\t1\n"
        "2025-03-01\tb2\te1\tm1\t1\t1\n",
        encoding="utf-8",
    )
    customer = tmp_path / "customer.tsv"
    customer.write_text(
        "legal_entity_id\ttask\n"
        "b1\tpredict future\n"
        "b2\tpredict future\n",
        encoding="utf-8",
    )
    trending = tmp_path / "trending.csv"
    trending.write_text("eclass\ne1\n", encoding="utf-8")
    out = tmp_path / "candidates.parquet"

    result = _run(
        [
            "uv", "run", "src/generate_candidates.py",
            "--plis", str(plis),
            "--customer", str(customer),
            "--trending-classes", str(trending),
            "--output", str(out),
            "--train-end", train_end,
            "--level", "2",
            "--lookback-months", "12",
            "--min-order-frequency", "1",
            "--min-lookback-spend", "0",
        ],
        cwd=project_root,
    )
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"

    df = pd.read_parquet(out)
    assert len(df) == 3
    rows = set(tuple(r) for _, r in df.iterrows())
    assert rows == {("b1", "e1", "m1"), ("b1", "e1", "m2"), ("b2", "e1", "m1")}
