#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PARAM_TO_CONFIG_PATH: dict[str, tuple[str, ...]] = {
    "config_top_k_per_buyer": ("modelling", "selection", "top_k_per_buyer"),
    "config_cold_start_top_k": ("submission", "cold_start_top_k"),
    "guardrail_min_orders": ("modelling", "selection", "guardrails", "min_orders"),
    "guardrail_min_months": ("modelling", "selection", "guardrails", "min_months"),
    "guardrail_high_spend": ("modelling", "selection", "guardrails", "high_spend"),
    "guardrail_min_avg_monthly_spend": (
        "modelling",
        "selection",
        "guardrails",
        "min_avg_monthly_spend",
    ),
}


@dataclass
class EffectRow:
    level: int
    param: str
    value: Any
    count: int
    mean_score: float
    median_score: float
    max_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Suggest next config.yaml parameters from score/tuning CSVs."
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--scores-dir", type=Path, default=Path("data/15_scores"))
    parser.add_argument("--best-dir", type=Path, default=Path("data/16_scores_best"))
    parser.add_argument(
        "--tuning-dir", type=Path, default=Path("data/17_submission_tuning")
    )
    parser.add_argument("--level", choices=("1", "2", "all"), default="all")
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--max-suggestions", type=int, default=6)
    return parser.parse_args()


def to_number_or_str(value: str) -> Any:
    s = value.strip()
    if s == "":
        return s
    try:
        if "." in s or "e" in s.lower():
            n = float(s)
            if n.is_integer():
                return int(n)
            return n
        return int(s)
    except ValueError:
        return s


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_param_effects(tuning_dir: Path) -> list[EffectRow]:
    out: list[EffectRow] = []
    for level in (1, 2):
        p = tuning_dir / f"param_effects_level{level}.csv"
        if not p.exists():
            continue
        for row in read_csv_rows(p):
            param = row.get("param", "")
            if param not in PARAM_TO_CONFIG_PATH:
                continue
            out.append(
                EffectRow(
                    level=level,
                    param=param,
                    value=to_number_or_str(row.get("value", "")),
                    count=int(float(row.get("count", "0") or 0)),
                    mean_score=float(row.get("mean_score", "0") or 0),
                    median_score=float(row.get("median_score", "0") or 0),
                    max_score=float(row.get("max_score", "0") or 0),
                )
            )
    return out


def read_score_files(base_dir: Path) -> int:
    # Side-effect-free validation that score CSVs exist and are parseable.
    file_count = 0
    for p in base_dir.rglob("score_summary_live.csv"):
        _ = read_csv_rows(p)
        file_count += 1
    return file_count


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {config_path}, got {type(data).__name__}")
    return data


def get_path_value(data: dict[str, Any], path: tuple[str, ...]) -> tuple[bool, Any]:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return False, None
        cur = cur[key]
    return True, cur


def set_path_value(data: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cur: dict[str, Any] = data
    for key in path[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[path[-1]] = value


def numeric_equal(a: Any, b: Any) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-12
    except (TypeError, ValueError):
        return a == b


def pick_best_for_level(
    rows: list[EffectRow], current_value: Any, min_count: int
) -> tuple[EffectRow | None, EffectRow | None]:
    current_row = next((r for r in rows if numeric_equal(r.value, current_value)), None)
    eligible = [r for r in rows if r.count >= min_count]
    if not eligible:
        eligible = rows
    if not eligible:
        return None, current_row
    best = sorted(eligible, key=lambda r: (r.max_score, r.mean_score, r.count), reverse=True)[0]
    return best, current_row


def best_per_param(
    effects: list[EffectRow],
    config: dict[str, Any],
    target_levels: set[int],
    min_count: int,
) -> list[tuple[tuple[str, ...], Any, float, str]]:
    # Returns: (config_path, suggested_value, support_score, reason)
    grouped: dict[str, list[EffectRow]] = {}
    for row in effects:
        if row.level in target_levels:
            grouped.setdefault(row.param, []).append(row)

    suggestions: list[tuple[tuple[str, ...], Any, float, str]] = []
    for param, rows in grouped.items():
        config_path = PARAM_TO_CONFIG_PATH[param]
        exists, current_value = get_path_value(config, config_path)
        if not exists:
            # existing_only mode: skip unknown keys
            continue

        by_level: dict[int, list[EffectRow]] = {}
        for r in rows:
            by_level.setdefault(r.level, []).append(r)

        votes: dict[str, dict[str, Any]] = {}
        current_refs: list[str] = []
        for level, level_rows in by_level.items():
            best, cur = pick_best_for_level(level_rows, current_value, min_count)
            if best is None:
                continue

            cur_ref = cur.max_score if cur else 0.0
            lift = best.max_score - cur_ref
            vkey = str(best.value)
            if vkey not in votes:
                votes[vkey] = {"value": best.value, "support": 0.0, "lines": []}

            # Prefer values that both improve best score and have more observations.
            support = lift + 0.1 * best.count
            votes[vkey]["support"] += support
            votes[vkey]["lines"].append(
                f"level{level}: value={best.value} max={best.max_score:.2f} "
                f"count={best.count} lift_vs_current={lift:.2f}"
            )
            if cur:
                current_refs.append(
                    f"level{level}: current={current_value} max={cur.max_score:.2f} count={cur.count}"
                )

        if not votes:
            continue

        winner = sorted(votes.values(), key=lambda v: v["support"], reverse=True)[0]
        suggested_value = winner["value"]
        if numeric_equal(suggested_value, current_value):
            continue

        support_score = float(winner["support"])
        reason_parts = []
        if current_refs:
            reason_parts.extend(current_refs)
        reason_parts.extend(winner["lines"])
        reason = " | ".join(reason_parts)
        suggestions.append((config_path, suggested_value, support_score, reason))

    suggestions.sort(key=lambda x: x[2], reverse=True)
    return suggestions


def build_patch(
    suggestions: list[tuple[tuple[str, ...], Any, float, str]],
    max_suggestions: int,
) -> tuple[dict[str, Any], list[str]]:
    patch: dict[str, Any] = {}
    notes: list[str] = []
    for path, value, _, reason in suggestions[:max_suggestions]:
        set_path_value(patch, path, value)
        notes.append(f"{'.'.join(path)} -> {value} :: {reason}")
    return patch, notes


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    # Validate expected data sources are present and readable.
    score_runs_count = read_score_files(args.scores_dir)
    best_runs_count = read_score_files(args.best_dir)

    effects = read_param_effects(args.tuning_dir)
    if args.level == "all":
        levels = {1, 2}
    else:
        levels = {int(args.level)}

    suggestions = best_per_param(
        effects=effects,
        config=config,
        target_levels=levels,
        min_count=max(1, args.min_count),
    )
    patch, notes = build_patch(suggestions, max_suggestions=max(1, args.max_suggestions))

    print("# Suggested config.yaml patch")
    if patch:
        print(yaml.safe_dump(patch, sort_keys=False).rstrip())
    else:
        print("{}")

    print()
    print("# Evidence")
    print(f"- parsed score files: data/15_scores={score_runs_count}, data/16_scores_best={best_runs_count}")
    print(f"- parsed tuning effect rows: {len(effects)}")
    print(f"- scope: level={args.level}, min_count={args.min_count}")
    if notes:
        for n in notes:
            print(f"- {n}")
    else:
        print("- No robust value change exceeded current settings for mapped keys.")


if __name__ == "__main__":
    main()
