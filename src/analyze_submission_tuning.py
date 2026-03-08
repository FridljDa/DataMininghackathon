"""
Submission tuning diagnostics: run metrics, param effects, submission shape, and plots.

Reads data/14_submission (current submissions), data/15_scores/online/runs/level{level},
and data/16_scores_best/.../best_run to produce CSVs and plots for hyperparameter choice.
Handles missing or empty run/submission data gracefully.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_PLOTTING = True
except ImportError:
    _HAS_PLOTTING = False

# Score summary columns we carry through
SCORE_COLS = [
    "total_score", "total_savings", "total_fees", "num_hits", "num_predictions",
    "spend_capture_rate", "total_ground_spend",
]
# Config keys to flatten into run_metrics for param_effects
CONFIG_KEYS = [
    "train_end", "lookback_months", "score_threshold", "top_k_per_buyer",
    "cold_start_top_k", "selected_features",
]
GUARDRAIL_KEYS = ["min_orders", "min_months", "high_spend", "min_avg_monthly_spend"]

# Task labels in customer_test: normalized to warm / cold / unknown
WARM_TASKS = {"predict future", "testing"}
COLD_TASKS = {"cold start"}


def _load_customer_task_map(customer_test_path: Path) -> dict[str, str]:
    """
    Load customer_test TSV and return legal_entity_id -> "warm" | "cold" | "unknown".
    Warm: task in ("predict future", "testing"); cold: "cold start"; else "unknown".
    """
    out: dict[str, str] = {}
    if not customer_test_path.is_file():
        return out
    try:
        df = pd.read_csv(customer_test_path, sep="\t", dtype=str, usecols=["legal_entity_id", "task"])
    except Exception:
        return out
    for _, row in df.iterrows():
        lid = str(row["legal_entity_id"])
        t = (row["task"] or "").strip().lower() if pd.notna(row["task"]) else ""
        if t in WARM_TASKS:
            out[lid] = "warm"
        elif t in COLD_TASKS:
            out[lid] = "cold"
        else:
            out[lid] = "unknown"
    return out


def _load_run(run_dir: Path) -> dict | None:
    """Load one run directory: score row + metadata. Return None if invalid."""
    live_path = run_dir / "score_summary_live.csv"
    summary_path = run_dir / "score_summary.csv"
    path_to_use = live_path if live_path.is_file() else summary_path
    meta_path = run_dir / "metadata.json"
    if not path_to_use.is_file():
        return None
    with path_to_use.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader, None)
    if not row or "total_score" not in row:
        return None
    try:
        float(row["total_score"])
    except (ValueError, TypeError):
        return None
    meta: dict = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"score_row": row, "meta": meta, "run_id": run_dir.name, "run_dir": run_dir}


def _is_run_dir(path: Path) -> bool:
    """True if path looks like a run dir (has score summary or .archived)."""
    return (path / "score_summary_live.csv").is_file() or (path / "score_summary.csv").is_file() or (path / ".archived").is_file()


def _gather_runs(runs_dir: Path) -> list[dict]:
    """Gather all valid runs from runs_dir. Supports flat (level/run_id) and nested (level/approach/run_id) layouts."""
    runs: list[dict] = []
    if not runs_dir.is_dir():
        return runs
    for path in runs_dir.iterdir():
        if not path.is_dir():
            continue
        if _is_run_dir(path):
            rec = _load_run(path)
            if rec is not None:
                runs.append(rec)
        else:
            for run_path in path.iterdir():
                if run_path.is_dir() and _is_run_dir(run_path):
                    rec = _load_run(run_path)
                    if rec is not None:
                        runs.append(rec)
    return runs


def _run_metrics_rows(runs: list[dict]) -> list[dict]:
    """Build one row per run: score cols + approach, created_at, run_id + flattened config."""
    rows = []
    for r in runs:
        row = {k: r["score_row"].get(k) for k in SCORE_COLS}
        row["run_id"] = r["run_id"]
        row["approach"] = r["meta"].get("approach", "")
        row["created_at"] = r["meta"].get("created_at", "")
        cfg = r["meta"].get("config") or {}
        for key in CONFIG_KEYS:
            val = cfg.get(key)
            if key == "selected_features" and isinstance(val, list):
                val = ",".join(val) if val else ""
            row[f"config_{key}"] = val
        gr = cfg.get("guardrails") or {}
        for key in GUARDRAIL_KEYS:
            row[f"guardrail_{key}"] = gr.get(key)
        rows.append(row)
    return rows


def _run_record_for_jsonl(r: dict, level: int) -> dict:
    """Build one JSONL record: score metrics, run_id, created_at, approach, level, params (full config)."""
    row = r["score_row"]
    meta = r["meta"]

    def _num(key: str):
        val = row.get(key)
        if val is None or (isinstance(val, str) and val.strip() == ""):
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    record: dict = {
        "total_score": _num("total_score"),
        "total_savings": _num("total_savings"),
        "total_fees": _num("total_fees"),
        "num_hits": _num("num_hits"),
        "num_predictions": _num("num_predictions"),
        "spend_capture_rate": _num("spend_capture_rate"),
        "run_id": r["run_id"],
        "created_at": meta.get("created_at") or None,
        "approach": meta.get("approach") or None,
        "level": level,
        "params": dict(meta.get("config") or {}),
    }
    return record


def _param_effects_df(run_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Group by each parameter and aggregate count, mean/median/max total_score."""
    if run_metrics_df.empty:
        return pd.DataFrame(columns=["param", "value", "count", "mean_score", "median_score", "max_score"])
    numeric_score = pd.to_numeric(run_metrics_df["total_score"], errors="coerce")
    run_metrics_df = run_metrics_df.assign(_score=numeric_score)
    param_cols = [c for c in run_metrics_df.columns if c.startswith("config_") or c.startswith("guardrail_")]
    effects = []
    for col in param_cols:
        for val, grp in run_metrics_df.groupby(col, dropna=False):
            v = val if pd.notna(val) and str(val).strip() != "" else "_empty"
            effects.append({
                "param": col,
                "value": v,
                "count": len(grp),
                "mean_score": grp["_score"].mean(),
                "median_score": grp["_score"].median(),
                "max_score": grp["_score"].max(),
            })
    out = pd.DataFrame(effects)
    if not out.empty:
        out = out.drop(columns=[], errors="ignore")
    return out


def _submission_shape(
    submission_path: Path,
    approach: str,
    task_map: dict[str, str] | None = None,
) -> dict:
    """Compute shape metrics for one submission CSV. Optionally add warm/cold buyer counts from task_map."""
    _warm_cold_none = {
        "n_warm_buyers_submitted": None,
        "n_cold_buyers_submitted": None,
        "n_unknown_task_buyers_submitted": None,
        "warm_buyer_share": None,
        "cold_buyer_share": None,
        "avg_predictions_per__warm_buyer": None,
        "avg_predictions_per__cold_buyer": None,
    }
    out = {"approach": approach, "submission_path": str(submission_path)}
    if not submission_path.is_file():
        out["n_predictions"] = None
        out["n_buyers"] = None
        out["avg_predictions_per_buyer"] = None
        out["median_predictions_per_buyer"] = None
        out["duplicate_rate"] = None
        out["top_cluster_share"] = None
        out.update(_warm_cold_none)
        return out
    try:
        df = pd.read_csv(submission_path)
    except Exception:
        out["n_predictions"] = None
        out["n_buyers"] = None
        out["avg_predictions_per_buyer"] = None
        out["median_predictions_per_buyer"] = None
        out["duplicate_rate"] = None
        out["top_cluster_share"] = None
        out.update(_warm_cold_none)
        return out
    if "legal_entity_id" not in df.columns or "cluster" not in df.columns:
        out["n_predictions"] = len(df)
        out["n_buyers"] = None
        out["avg_predictions_per_buyer"] = None
        out["median_predictions_per_buyer"] = None
        out["duplicate_rate"] = None
        out["top_cluster_share"] = None
        out.update(_warm_cold_none)
        return out
    df["legal_entity_id"] = df["legal_entity_id"].astype(str)
    n_raw = len(df)
    dedup = df.drop_duplicates(subset=["legal_entity_id", "cluster"])
    n_predictions = len(dedup)
    duplicate_rate = (n_raw - n_predictions) / n_raw if n_raw else 0.0
    n_buyers = dedup["legal_entity_id"].nunique()
    per_buyer = dedup.groupby("legal_entity_id").size()
    out["n_predictions"] = n_predictions
    out["n_buyers"] = int(n_buyers)
    out["avg_predictions_per_buyer"] = float(per_buyer.mean()) if len(per_buyer) else None
    out["median_predictions_per_buyer"] = float(per_buyer.median()) if len(per_buyer) else None
    out["duplicate_rate"] = duplicate_rate
    cluster_counts = dedup["cluster"].value_counts()
    top_share = cluster_counts.iloc[0] / n_predictions if n_predictions and len(cluster_counts) else None
    out["top_cluster_share"] = float(top_share) if top_share is not None else None

    if task_map is not None and n_buyers:
        unique_buyers = dedup["legal_entity_id"].unique()
        n_warm = sum(1 for lid in unique_buyers if task_map.get(lid) == "warm")
        n_cold = sum(1 for lid in unique_buyers if task_map.get(lid) == "cold")
        n_unknown = n_buyers - n_warm - n_cold
        out["n_warm_buyers_submitted"] = int(n_warm)
        out["n_cold_buyers_submitted"] = int(n_cold)
        out["n_unknown_task_buyers_submitted"] = int(n_unknown)
        out["warm_buyer_share"] = n_warm / n_buyers
        out["cold_buyer_share"] = n_cold / n_buyers
        warm_ids = [lid for lid in unique_buyers if task_map.get(lid) == "warm"]
        cold_ids = [lid for lid in unique_buyers if task_map.get(lid) == "cold"]
        warm_counts = per_buyer.reindex(warm_ids).dropna()
        cold_counts = per_buyer.reindex(cold_ids).dropna()
        out["avg_predictions_per__warm_buyer"] = float(warm_counts.mean()) if len(warm_counts) else None
        out["avg_predictions_per__cold_buyer"] = float(cold_counts.mean()) if len(cold_counts) else None
    else:
        out.update(_warm_cold_none)
    return out


def _infer_approach_from_path(path: Path) -> str:
    """e.g. data/14_submission/online/lgbm_two_stage/level1/submission.csv -> lgbm_two_stage."""
    parts = path.parts
    for i, p in enumerate(parts):
        if p == "online" and i + 1 < len(parts):
            return parts[i + 1]
    return path.parent.name or "unknown"


def _best_run_identifiers(best_run_dir: Path) -> tuple[str | None, float | None, str | None]:
    """Return (run_id, total_score, created_at) from best_run dir by reading summary + metadata, or (None, None, None)."""
    if not best_run_dir.is_dir():
        return None, None, None
    live_path = best_run_dir / "score_summary_live.csv"
    summary_path = best_run_dir / "score_summary.csv"
    path_to_use = live_path if live_path.is_file() else summary_path
    if not path_to_use.is_file():
        return None, None, None
    try:
        with path_to_use.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row = next(reader, None)
        if not row or "total_score" not in row:
            return None, None, None
        total_score = float(row["total_score"])
    except (ValueError, TypeError, OSError):
        return None, None, None
    created_at = None
    run_id = None
    meta_path = best_run_dir / "metadata.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            created_at = (meta.get("created_at") or "").strip() or None
            run_id = (meta.get("run_id") or "").strip() or None
        except (json.JSONDecodeError, OSError):
            pass
    return run_id, total_score, created_at


def _write_placeholder_csv(path: Path, message: str, columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if columns:
            w.writerow(columns)
        w.writerow([message])


def _write_placeholder_plot(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not _HAS_PLOTTING:
        path.with_suffix(".txt").write_text(f"No matplotlib: {message}", encoding="utf-8")
        return
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=10, wrap=True)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, required=True, help="Level (1 or 2)")
    parser.add_argument("--output-dir", required=True, dest="output_dir", help="Output directory (e.g. data/17_submission_tuning)")
    parser.add_argument("--runs-dir", required=True, dest="runs_dir", help="Archived runs dir (e.g. data/15_scores/online/runs/level1)")
    parser.add_argument("--best-run-dir", required=True, dest="best_run_dir", help="Best run copy dir (e.g. data/16_scores_best/online/level1/best_run)")
    parser.add_argument("--customer-test", required=True, dest="customer_test", help="Path to customer_test.csv (for warm/cold buyer counts)")
    parser.add_argument("--submissions", nargs="*", default=[], help="Paths to submission CSVs (online, this level)")
    args = parser.parse_args()

    level = args.level
    out_dir = Path(args.output_dir)
    runs_dir = Path(args.runs_dir)
    best_run_dir = Path(args.best_run_dir)
    customer_test_path = Path(args.customer_test)
    submission_paths = [Path(p) for p in args.submissions]
    task_map = _load_customer_task_map(customer_test_path)

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Run metrics ----
    runs = _gather_runs(runs_dir)
    run_metrics_path = out_dir / f"run_metrics_level{level}.csv"
    if not runs:
        _write_placeholder_csv(
            run_metrics_path,
            "No archived runs found; run archive_score_run after submitting.",
            columns=["run_id", "approach", "created_at"] + SCORE_COLS,
        )
        run_metrics_df = pd.DataFrame()
    else:
        rows = _run_metrics_rows(runs)
        run_metrics_df = pd.DataFrame(rows)
        run_metrics_df.to_csv(run_metrics_path, index=False)
    print(f"Wrote {run_metrics_path} ({len(run_metrics_df)} runs)")

    # ---- Run records JSONL (one object per run: metrics + full params) ----
    run_records_path = out_dir / f"run_records_level{level}.jsonl"
    with run_records_path.open("w", encoding="utf-8") as f:
        for r in runs:
            record = _run_record_for_jsonl(r, level)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {run_records_path} ({len(runs)} runs)")

    # ---- Param effects ----
    param_effects_path = out_dir / f"param_effects_level{level}.csv"
    if run_metrics_df.empty:
        _write_placeholder_csv(
            param_effects_path,
            "No run metrics; param effects require archived runs.",
            columns=["param", "value", "count", "mean_score", "median_score", "max_score"],
        )
    else:
        param_effects_df = _param_effects_df(run_metrics_df)
        param_effects_df.to_csv(param_effects_path, index=False)
        print(f"Wrote {param_effects_path}")

    # ---- Current submission shape ----
    shape_rows = []
    for p in submission_paths:
        approach = _infer_approach_from_path(p)
        shape_rows.append(_submission_shape(p, approach, task_map=task_map))
    shape_path = out_dir / f"current_submission_shape_level{level}.csv"
    pd.DataFrame(shape_rows).to_csv(shape_path, index=False)
    print(f"Wrote {shape_path} ({len(shape_rows)} submissions)")

    # ---- Plots: resolve best run for highlighting ----
    best_run_id, best_score, best_created = _best_run_identifiers(best_run_dir)

    def _is_best_run(df: pd.DataFrame) -> pd.Series:
        if df.empty:
            return pd.Series(dtype=bool)
        if best_run_id:
            return df["run_id"] == best_run_id
        if best_score is not None and best_created:
            return (pd.to_numeric(df["total_score"], errors="coerce") == best_score) & (df["created_at"].astype(str).str.strip() == best_created)
        return pd.Series(False, index=df.index)

    def _plot_score_vs_predictions() -> None:
        path = out_dir / f"score_vs_predictions_level{level}.png"
        if run_metrics_df.empty:
            _write_placeholder_plot(path, "No run data; submit and archive runs to see score vs num_predictions.")
            return
        if not _HAS_PLOTTING:
            _write_placeholder_plot(path, "Matplotlib not available.")
            return
        df = run_metrics_df.copy()
        df["total_score"] = pd.to_numeric(df["total_score"], errors="coerce")
        df["num_predictions"] = pd.to_numeric(df["num_predictions"], errors="coerce")
        df = df.dropna(subset=["total_score", "num_predictions"])
        if df.empty:
            _write_placeholder_plot(path, "No valid score/predictions in run metrics.")
            return
        fig, ax = plt.subplots(figsize=(7, 5))
        is_best = _is_best_run(df)
        if is_best.any():
            best_df = df[is_best]
            ax.scatter(best_df["num_predictions"], best_df["total_score"], c="green", s=80, label="Best run", zorder=3)
        ax.scatter(df["num_predictions"], df["total_score"], alpha=0.6, s=30, label="Runs")
        ax.set_xlabel("num_predictions")
        ax.set_ylabel("total_score")
        ax.set_title(f"Level {level}: Score vs predictions (fee pressure)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"Wrote {path}")

    def _plot_score_vs_capture() -> None:
        path = out_dir / f"score_vs_capture_level{level}.png"
        if run_metrics_df.empty:
            _write_placeholder_plot(path, "No run data; submit and archive runs to see score vs spend_capture_rate.")
            return
        if not _HAS_PLOTTING:
            _write_placeholder_plot(path, "Matplotlib not available.")
            return
        df = run_metrics_df.copy()
        df["total_score"] = pd.to_numeric(df["total_score"], errors="coerce")
        df["spend_capture_rate"] = pd.to_numeric(df["spend_capture_rate"], errors="coerce")
        df = df.dropna(subset=["total_score", "spend_capture_rate"])
        if df.empty:
            _write_placeholder_plot(path, "No valid score/capture in run metrics.")
            return
        fig, ax = plt.subplots(figsize=(7, 5))
        is_best = _is_best_run(df)
        if is_best.any():
            best_df = df[is_best]
            ax.scatter(best_df["spend_capture_rate"], best_df["total_score"], c="green", s=80, label="Best run", zorder=3)
        ax.scatter(df["spend_capture_rate"], df["total_score"], alpha=0.6, s=30, label="Runs")
        ax.set_xlabel("spend_capture_rate")
        ax.set_ylabel("total_score")
        ax.set_title(f"Level {level}: Score vs spend capture")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"Wrote {path}")

    def _plot_run_timeline() -> None:
        path = out_dir / f"run_timeline_level{level}.png"
        if run_metrics_df.empty:
            _write_placeholder_plot(path, "No run data; submit and archive runs to see timeline.")
            return
        if not _HAS_PLOTTING:
            _write_placeholder_plot(path, "Matplotlib not available.")
            return
        df = run_metrics_df.copy()
        df["total_score"] = pd.to_numeric(df["total_score"], errors="coerce")
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        df = df.dropna(subset=["created_at", "total_score"]).sort_values("created_at")
        if df.empty:
            _write_placeholder_plot(path, "No valid created_at/score in run metrics.")
            return
        fig, ax = plt.subplots(figsize=(9, 4))
        is_best = _is_best_run(df)
        ax.scatter(df["created_at"], df["total_score"], alpha=0.6, s=30, label="Runs")
        if is_best.any():
            best_df = df[is_best]
            ax.scatter(best_df["created_at"], best_df["total_score"], c="green", s=80, label="Best run", zorder=3)
        ax.set_xlabel("created_at")
        ax.set_ylabel("total_score")
        ax.set_title(f"Level {level}: Run timeline")
        ax.legend()
        plt.xticks(rotation=20)
        fig.tight_layout()
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"Wrote {path}")

    def _plot_submission_size_vs_score() -> None:
        path = out_dir / f"submission_size_vs_score_level{level}.png"
        if not _HAS_PLOTTING:
            _write_placeholder_plot(path, "Matplotlib not available.")
            return
        fig, ax = plt.subplots(figsize=(7, 5))
        # Current submissions: n_predictions (x) – we don't have score for "current" unless we overlay from run_metrics by approach
        shape_df = pd.DataFrame(shape_rows)
        if not shape_df.empty and "n_predictions" in shape_df.columns:
            valid = shape_df["n_predictions"].notna()
            if valid.any():
                ax.scatter(
                    shape_df.loc[valid, "n_predictions"],
                    [0] * valid.sum(),
                    marker="s",
                    s=60,
                    label="Current submissions (score unknown)",
                    alpha=0.8,
                )
        if not run_metrics_df.empty:
            df = run_metrics_df.copy()
            df["total_score"] = pd.to_numeric(df["total_score"], errors="coerce")
            df["num_predictions"] = pd.to_numeric(df["num_predictions"], errors="coerce")
            df = df.dropna(subset=["total_score", "num_predictions"])
            if not df.empty:
                is_best = _is_best_run(df)
                if is_best.any():
                    best_df = df[is_best]
                    ax.scatter(best_df["num_predictions"], best_df["total_score"], c="green", s=80, label="Best run", zorder=3)
                ax.scatter(df["num_predictions"], df["total_score"], alpha=0.6, s=30, label="Archived runs")
        ax.set_xlabel("num_predictions / n_predictions")
        ax.set_ylabel("total_score")
        ax.set_title(f"Level {level}: Submission size vs score")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"Wrote {path}")

    _plot_score_vs_predictions()
    _plot_score_vs_capture()
    _plot_run_timeline()
    _plot_submission_size_vs_score()


if __name__ == "__main__":
    main()
