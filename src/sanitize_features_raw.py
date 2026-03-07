"""
Apply per-feature missing-value policies to features_raw (drop_row, ignore, fill).

Reads config (YAML) from --config and key path --config-key, and features_raw parquet.
Outputs sanitized parquet. Policies: drop_row (drop rows where null), ignore, fill
(strategy: constant | median | mean | mode; value for constant).
Unspecified features use default_action (ignore). Config must not reference unknown columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


VALID_ACTIONS = ("drop_row", "ignore", "fill")
FILL_STRATEGIES = ("constant", "median", "mean", "mode")


def _load_config(path: Path, key_path: str) -> dict:
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    for key in key_path.strip().split("."):
        data = (data or {}).get(key, {})
    return data if isinstance(data, dict) else {}


def _get_policy(config: dict, column: str) -> str:
    features = config.get("features") or {}
    default = config.get("default_action", "ignore")
    entry = features.get(column)
    if entry is None:
        return default
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        return entry.get("action", default)
    return default


def _get_fill_params(config: dict, column: str) -> dict:
    features = config.get("features") or {}
    entry = features.get(column)
    if not isinstance(entry, dict):
        return {}
    return {
        "strategy": entry.get("strategy", "constant"),
        "value": entry.get("value"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-raw", required=True, dest="features_raw", help="Path to features_raw parquet.")
    parser.add_argument("--output", required=True, help="Path to output sanitized parquet.")
    parser.add_argument("--config", required=True, help="Path to config YAML.")
    parser.add_argument("--config-key", required=True, dest="config_key", help="Dot path to sanitation block (e.g. modelling.missing_value_sanitation).")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    config = _load_config(config_path, args.config_key)

    raw_path = Path(args.features_raw)
    out_path = Path(args.output)
    df = pd.read_parquet(raw_path)
    n_in = len(df)

    # Config must not reference columns that are not in the dataframe
    features_cfg = config.get("features") or {}
    unknown = [c for c in features_cfg if c not in df.columns]
    if unknown:
        raise ValueError(
            f"missing_value_sanitation.features references columns not in features_raw: {unknown}. "
            f"Available: {list(df.columns)}"
        )

    default_action = config.get("default_action", "ignore")
    if default_action not in VALID_ACTIONS:
        raise ValueError(f"default_action must be one of {VALID_ACTIONS}, got {default_action!r}")

    stats = {"drop_row": [], "fill": [], "ignore": []}

    for col in df.columns:
        policy = _get_policy(config, col)
        if policy not in VALID_ACTIONS:
            raise ValueError(f"feature {col!r} action must be one of {VALID_ACTIONS}, got {policy!r}")

        if policy == "ignore":
            stats["ignore"].append(col)
            continue

        if policy == "drop_row":
            null_count = df[col].isna().sum()
            if null_count > 0:
                df = df[df[col].notna()].copy()
                stats["drop_row"].append((col, int(null_count)))
            continue

        if policy == "fill":
            null_count = df[col].isna().sum()
            if null_count == 0:
                stats["ignore"].append(col)
                continue
            params = _get_fill_params(config, col)
            strategy = params.get("strategy", "constant")

            if strategy == "constant":
                value = params.get("value")
                if value is None and df[col].dtype.kind in "iufc":
                    value = 0
                if value is None:
                    raise ValueError(f"feature {col!r}: fill strategy 'constant' requires 'value' in config")
                df[col] = df[col].fillna(value)
                stats["fill"].append((col, strategy, int(null_count)))
                continue

            if strategy in ("median", "mean"):
                if df[col].dtype.kind not in "iufc":
                    raise ValueError(
                        f"feature {col!r}: fill strategy {strategy!r} requires numeric column, got {df[col].dtype}"
                    )
                fill_val = df[col].median() if strategy == "median" else df[col].mean()
                df[col] = df[col].fillna(fill_val)
                stats["fill"].append((col, strategy, int(null_count)))
                continue

            if strategy == "mode":
                mode_vals = df[col].dropna().mode()
                fill_val = mode_vals.iloc[0] if len(mode_vals) else None
                if fill_val is None:
                    raise ValueError(f"feature {col!r}: no mode available (all null?)")
                df[col] = df[col].fillna(fill_val)
                stats["fill"].append((col, strategy, int(null_count)))
                continue

            raise ValueError(f"feature {col!r}: fill strategy must be one of {FILL_STRATEGIES}, got {strategy!r}")

    n_out = len(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    print(f"Sanitized features_raw: rows {n_in} -> {n_out}")
    for col, dropped in stats["drop_row"]:
        print(f"  drop_row {col}: dropped {dropped} rows")
    for col, strat, filled in stats["fill"]:
        print(f"  fill {col} ({strat}): filled {filled} nulls")
    print(f"Wrote {n_out} rows to {out_path}")


if __name__ == "__main__":
    main()
