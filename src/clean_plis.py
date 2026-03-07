"""
Clean raw plis_training data using column rules from config.yaml.

The cleaning contract is read from a YAML block such as cleaning.plis_training.
Supported operations:
- keep only configured columns
- normalize string fields (strip, empty_as_null)
- parse dates and numerics
- drop rows on missing/invalid required fields
- set invalid optional values to null
- drop exact duplicates on a configured subset

Input and output are tab-separated to match the current repo pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


VALID_TYPES = ("string", "date", "float", "int")
VALID_ACTIONS = ("drop_row", "keep_null", "set_null")


def _load_config(path: Path, key_path: str) -> dict:
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    for key in key_path.strip().split("."):
        data = (data or {}).get(key, {})
    return data if isinstance(data, dict) else {}


def _normalize_text(series: pd.Series, *, strip: bool, empty_as_null: bool) -> pd.Series:
    out = series.astype("string")
    if strip:
        out = out.str.strip()
    if empty_as_null:
        out = out.mask(out.eq(""), pd.NA)
    return out


def _prepare_scalar_input(series: pd.Series, *, strip: bool) -> pd.Series:
    if pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype):
        return _normalize_text(series, strip=strip, empty_as_null=True)
    return series


def _missing_action(rules: dict) -> str:
    default = "drop_row" if rules.get("required") else "keep_null"
    action = rules.get("on_missing", default)
    if action not in VALID_ACTIONS:
        raise ValueError(f"on_missing must be one of {VALID_ACTIONS}, got {action!r}")
    return action


def _invalid_action(rules: dict) -> str:
    default = "drop_row" if rules.get("required") else "set_null"
    action = rules.get("on_invalid", default)
    if action not in VALID_ACTIONS:
        raise ValueError(f"on_invalid must be one of {VALID_ACTIONS}, got {action!r}")
    return action


def _apply_numeric_bounds(series: pd.Series, rules: dict) -> pd.Series:
    invalid = pd.Series(False, index=series.index)

    if "min_exclusive" in rules:
        invalid |= series.notna() & ~(series > rules["min_exclusive"])
    if "min_inclusive" in rules:
        invalid |= series.notna() & ~(series >= rules["min_inclusive"])
    if "max_exclusive" in rules:
        invalid |= series.notna() & ~(series < rules["max_exclusive"])
    if "max_inclusive" in rules:
        invalid |= series.notna() & ~(series <= rules["max_inclusive"])

    return invalid


def _validate_contract(contract: dict, available_columns: list[str]) -> tuple[list[str], dict, dict]:
    keep_columns = contract.get("keep_columns") or []
    if not isinstance(keep_columns, list) or not keep_columns:
        raise ValueError("cleaning contract must define a non-empty keep_columns list")
    if len(keep_columns) != len(set(keep_columns)):
        raise ValueError(f"keep_columns contains duplicates: {keep_columns}")

    missing = [col for col in keep_columns if col not in available_columns]
    if missing:
        raise ValueError(
            f"Configured keep_columns are missing from input: {missing}. "
            f"Available: {available_columns}"
        )

    columns_cfg = contract.get("columns") or {}
    if not isinstance(columns_cfg, dict):
        raise ValueError("cleaning contract 'columns' must be a mapping")
    unknown_cfg = [col for col in columns_cfg if col not in keep_columns]
    if unknown_cfg:
        raise ValueError(
            f"cleaning contract references columns not present in keep_columns: {unknown_cfg}"
        )

    for col, rules in columns_cfg.items():
        if not isinstance(rules, dict):
            raise ValueError(f"Column rules for {col!r} must be a mapping")
        col_type = rules.get("type")
        if col_type not in VALID_TYPES:
            raise ValueError(f"Column {col!r}: type must be one of {VALID_TYPES}, got {col_type!r}")
        _missing_action(rules)
        if col_type in {"date", "float", "int"}:
            _invalid_action(rules)

    deduplicate = contract.get("deduplicate") or {}
    if deduplicate:
        subset = deduplicate.get("subset") or []
        if not subset:
            raise ValueError("deduplicate.subset must be a non-empty list when deduplicate is configured")
        unknown_subset = [col for col in subset if col not in keep_columns]
        if unknown_subset:
            raise ValueError(
                f"deduplicate.subset references columns not present in keep_columns: {unknown_subset}"
            )
        keep = deduplicate.get("keep", "first")
        if keep not in ("first", "last", False):
            raise ValueError("deduplicate.keep must be 'first', 'last', or false")

    return keep_columns, columns_cfg, deduplicate


def _read_dtype_map(columns_cfg: dict) -> dict[str, str]:
    dtype_map: dict[str, str] = {}
    for col, rules in columns_cfg.items():
        if rules.get("type") in {"string", "date"}:
            dtype_map[col] = "string"
    return dtype_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to raw plis_training file (tab-separated).")
    parser.add_argument("--output", required=True, help="Path to cleaned output file (tab-separated).")
    parser.add_argument("--config", required=True, help="Path to config YAML.")
    parser.add_argument(
        "--config-key",
        default="cleaning.plis_training",
        dest="config_key",
        help="Dot path to cleaning block in config.yaml.",
    )
    parser.add_argument("--sep", default="\t", help="Field delimiter for input and output. Defaults to tab.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    config_path = Path(args.config)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input not found: {input_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    contract = _load_config(config_path, args.config_key)
    if not contract:
        raise ValueError(f"Config key not found or empty: {args.config_key}")

    available_columns = pd.read_csv(
        input_path, sep=args.sep, nrows=0, encoding="utf-8-sig"
    ).columns.tolist()
    keep_columns, columns_cfg, deduplicate = _validate_contract(contract, available_columns)
    dtype_map = _read_dtype_map(columns_cfg)

    df = pd.read_csv(
        input_path,
        sep=args.sep,
        usecols=keep_columns,
        low_memory=False,
        encoding="utf-8-sig",
        dtype=dtype_map,
    )
    n_in = len(df)
    keep_mask = pd.Series(True, index=df.index)

    dropped_missing: list[tuple[str, int]] = []
    dropped_invalid: list[tuple[str, int]] = []
    nulled_invalid: list[tuple[str, int]] = []

    for col in keep_columns:
        rules = columns_cfg.get(col)
        if rules is None:
            continue

        col_type = rules["type"]
        strip = bool(rules.get("strip", False))

        if col_type == "string":
            cleaned = _normalize_text(
                df[col],
                strip=strip,
                empty_as_null=bool(rules.get("empty_as_null", False)),
            )
            missing_mask = cleaned.isna()
            n_missing = int(missing_mask.sum())
            if n_missing:
                action = _missing_action(rules)
                if action == "drop_row":
                    keep_mask &= ~missing_mask
                    dropped_missing.append((col, n_missing))
            df[col] = cleaned
            continue

        prepared = _prepare_scalar_input(df[col], strip=strip)
        missing_mask = prepared.isna()

        if col_type == "date":
            cleaned = pd.to_datetime(
                prepared,
                format=rules.get("format"),
                errors="coerce",
            )
            parse_invalid = prepared.notna() & cleaned.isna()
            invalid_mask = parse_invalid
        else:
            cleaned = pd.to_numeric(prepared, errors="coerce")
            parse_invalid = prepared.notna() & cleaned.isna()
            invalid_mask = parse_invalid | _apply_numeric_bounds(cleaned, rules)
            if col_type == "int":
                non_integer = cleaned.notna() & (cleaned % 1 != 0)
                invalid_mask |= non_integer
                cleaned = cleaned.round().astype("Int64")

        n_missing = int(missing_mask.sum())
        if n_missing:
            action = _missing_action(rules)
            if action == "drop_row":
                keep_mask &= ~missing_mask
                dropped_missing.append((col, n_missing))

        n_invalid = int(invalid_mask.sum())
        if n_invalid:
            action = _invalid_action(rules)
            if action == "drop_row":
                keep_mask &= ~invalid_mask
                dropped_invalid.append((col, n_invalid))
            else:
                cleaned = cleaned.mask(invalid_mask)
                nulled_invalid.append((col, n_invalid))

        df[col] = cleaned

    df = df.loc[keep_mask, keep_columns].copy()

    for col, rules in columns_cfg.items():
        if rules["type"] == "date":
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime(rules.get("format", "%Y-%m-%d"))

    duplicates_dropped = 0
    if deduplicate:
        before = len(df)
        df = df.drop_duplicates(
            subset=deduplicate["subset"],
            keep=deduplicate.get("keep", "first"),
        ).copy()
        duplicates_dropped = before - len(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep=args.sep, index=False)

    print(f"Cleaned plis_training: rows {n_in} -> {len(df)}")
    for col, count in dropped_missing:
        print(f"  drop_row missing {col}: {count}")
    for col, count in dropped_invalid:
        print(f"  drop_row invalid {col}: {count}")
    for col, count in nulled_invalid:
        print(f"  set_null invalid {col}: {count}")
    if duplicates_dropped:
        print(f"  drop_duplicates: {duplicates_dropped}")
    print(f"Wrote cleaned file to {output_path}")


if __name__ == "__main__":
    main()
