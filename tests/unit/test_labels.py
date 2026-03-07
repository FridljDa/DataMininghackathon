"""Unit tests for attach_validation_labels: distinct set_id counting, label threshold, level 2, missing set_id."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from modelling.common.labels import attach_validation_labels


@pytest.fixture
def plis_with_set_id(tmp_path: Path) -> Path:
    """PLIS TSV in val window with set_id; same set_id = one order."""
    p = tmp_path / "plis.tsv"
    # b1,e1: set_id S1 (3 rows) -> 1 order, spend 30. b1,e2: S2,S3 (2 rows each) -> 2 orders, spend 20+30=50.
    # b2,e1: S4 (1 row) -> 1 order, spend 10.
    p.write_text(
        "orderdate\tlegal_entity_id\teclass\tset_id\tquantityvalue\tvk_per_item\n"
        "2025-02-01\tb1\te1\tS1\t1\t5\n"
        "2025-02-02\tb1\te1\tS1\t1\t10\n"
        "2025-02-03\tb1\te1\tS1\t1\t15\n"
        "2025-03-01\tb1\te2\tS2\t1\t10\n"
        "2025-03-02\tb1\te2\tS2\t1\t10\n"
        "2025-03-10\tb1\te2\tS3\t1\t15\n"
        "2025-03-11\tb1\te2\tS3\t1\t15\n"
        "2025-04-01\tb2\te1\tS4\t1\t10\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def plis_no_set_id(tmp_path: Path) -> Path:
    """PLIS TSV without set_id column."""
    p = tmp_path / "plis_no_set_id.tsv"
    p.write_text(
        "orderdate\tlegal_entity_id\teclass\tquantityvalue\tvk_per_item\n"
        "2025-02-01\tb1\te1\t1\t10\n",
        encoding="utf-8",
    )
    return p


def test_n_orders_val_distinct_set_id(
    plis_with_set_id: Path,
) -> None:
    """Repeated rows with same set_id count as one order."""
    df = pd.DataFrame([
        {"legal_entity_id": "b1", "eclass": "e1"},
        {"legal_entity_id": "b1", "eclass": "e2"},
        {"legal_entity_id": "b2", "eclass": "e1"},
    ])
    out = attach_validation_labels(
        df,
        plis_with_set_id,
        val_start=pd.Timestamp("2025-01-01"),
        val_end=pd.Timestamp("2025-06-30"),
        n_min_label=1,
        level=1,
    )
    out = out.sort_values(["legal_entity_id", "eclass"]).reset_index(drop=True)
    assert list(out["n_orders_val"]) == [1, 2, 1]  # b1,e1: 1 order; b1,e2: 2 orders; b2,e1: 1 order
    assert list(out["s_val"]) == [30.0, 50.0, 10.0]  # spend unchanged (sum of rows)
    assert list(out["label"]) == [1, 1, 1]


def test_label_threshold_n_min_label(
    plis_with_set_id: Path,
) -> None:
    """label = 1 iff n_orders_val >= n_min_label."""
    df = pd.DataFrame([
        {"legal_entity_id": "b1", "eclass": "e1"},   # 1 order
        {"legal_entity_id": "b1", "eclass": "e2"},   # 2 orders
        {"legal_entity_id": "b2", "eclass": "e1"},   # 1 order
    ])
    out1 = attach_validation_labels(
        df, plis_with_set_id,
        pd.Timestamp("2025-01-01"), pd.Timestamp("2025-06-30"),
        n_min_label=1, level=1,
    )
    out2 = attach_validation_labels(
        df, plis_with_set_id,
        pd.Timestamp("2025-01-01"), pd.Timestamp("2025-06-30"),
        n_min_label=2, level=1,
    )
    out1 = out1.sort_values(["legal_entity_id", "eclass"]).reset_index(drop=True)
    out2 = out2.sort_values(["legal_entity_id", "eclass"]).reset_index(drop=True)
    assert list(out1["label"]) == [1, 1, 1]
    assert list(out2["label"]) == [0, 1, 0]  # only b1,e2 has >= 2 orders


def test_level2_manufacturer_grouping(tmp_path: Path) -> None:
    """Level 2 keys include manufacturer; n_orders_val is distinct set_id per (b, e, m)."""
    p = tmp_path / "plis_l2.tsv"
    p.write_text(
        "orderdate\tlegal_entity_id\teclass\tmanufacturer\tset_id\tquantityvalue\tvk_per_item\n"
        "2025-02-01\tb1\te1\tM1\tS1\t1\t10\n"
        "2025-02-02\tb1\te1\tM1\tS1\t1\t10\n"
        "2025-02-03\tb1\te1\tM2\tS2\t1\t20\n",
        encoding="utf-8",
    )
    df = pd.DataFrame([
        {"legal_entity_id": "b1", "eclass": "e1", "manufacturer": "M1"},
        {"legal_entity_id": "b1", "eclass": "e1", "manufacturer": "M2"},
    ])
    out = attach_validation_labels(
        df, p,
        pd.Timestamp("2025-01-01"), pd.Timestamp("2025-06-30"),
        n_min_label=1, level=2,
    )
    out = out.sort_values(["legal_entity_id", "eclass", "manufacturer"]).reset_index(drop=True)
    assert list(out["n_orders_val"]) == [1, 1]
    assert list(out["label"]) == [1, 1]


def test_missing_set_id_raises(plis_no_set_id: Path) -> None:
    """Missing set_id in plis raises ValueError."""
    df = pd.DataFrame([{"legal_entity_id": "b1", "eclass": "e1"}])
    with pytest.raises(ValueError, match="set_id"):
        attach_validation_labels(
            df, plis_no_set_id,
            pd.Timestamp("2025-01-01"), pd.Timestamp("2025-06-30"),
            n_min_label=1, level=1,
        )
