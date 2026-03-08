# Submission tuning: threshold and portfolio sweep

This doc defines the first **negative-threshold sweep** and how to run it using the existing archive and tuning scripts.

## Sweep matrix

Run selection-only experiments (scores already produced) with:

| Parameter | Values |
|-----------|--------|
| `score_threshold` | `0.0`, `-0.01`, `-0.05`, `-0.10`, `-0.20` |
| `top_k_per_buyer` | `150`, `300`, `400` |
| Level | `1`, `2` (evaluate separately) |
| Approach focus | `lgbm_two_stage`; compare vs `phase3_repro` and `pass_through` |

Guardrails can stay at current defaults or use the tighter set (`min_orders: 2`, `min_months: 2`, `high_spend: 200`) for level 1 if desired.

## Metrics to compare

- `total_score`, `total_fees`, `num_hits`, `spend_capture_rate`
- `n_predictions`, `avg_predictions_per_buyer`
- Warm vs cold submission shape from `data/17_submission_tuning/current_submission_shape_level{1,2}.csv`

## Workflow (reuse existing pipeline)

1. **Ensure scores exist**  
   Run the DAG up to and including `train_approach` so that `data/12_predictions/{mode}/{approach}/level{level}/scores.parquet` exist for the approaches you care about.

2. **Per trial**  
   Set level-specific selection in `config.yaml` under `modelling.selection.by_level`:
   - For level 1: `modelling.selection.by_level.1` with `score_threshold`, `top_k_per_buyer`, and optionally `guardrails`.
   - For level 2: `modelling.selection.by_level.2` with the same keys.

   Then run from portfolio through submission and (for online) submit + archive:
   ```bash
   snakemake data/13_portfolio/online/lgbm_two_stage/level1/portfolio.parquet \
             data/14_submission/online/lgbm_two_stage/level1/submission.csv \
             data/14_submission/online/lgbm_two_stage/level2/submission.csv
   ```
   Submit to the portal, then run the rules that archive the run (e.g. `archive_score_run` inputs).

3. **After multiple trials**  
   Recompute tuning summaries and param effects:
   ```bash
   uv run src/analyze_submission_tuning.py \
     --level 1 --output-dir data/17_submission_tuning \
     --runs-dir data/15_scores/online/runs/level1 \
     --best-run-dir data/16_scores_best/online/level1/best_run \
     --customer-test data/02_raw/customer_test.csv \
     --submissions data/14_submission/online/*/level1/submission.csv
   ```
   Repeat for `--level 2` with the corresponding paths.

4. **Inspect**  
   Use `data/17_submission_tuning/param_effects_level{1,2}.csv`, `run_metrics_level{1,2}.csv`, and the plots to pick the best risk-adjusted settings per level. For tuning history and parameter-suggestion workflows (e.g. the submission-param-suggester skill), `run_records_level{1,2}.jsonl` provides one JSON object per archived run with fields: `total_score`, `total_savings`, `total_fees`, `num_hits`, `num_predictions`, `spend_capture_rate`, `run_id`, `created_at`, `approach`, `level`, and `params` (the full `metadata.config` snapshot for that run, including nested `guardrails` and any future keys).

## Optional: sweep runner script

From the repo root you can run a small script that iterates over the matrix, patches `config.yaml` with `by_level` overrides, and runs Snakemake from `select_portfolio` through merged submission (no re-train, no submit/archive):

```bash
uv run scripts/run_threshold_sweep.py --dry-run   # print trials only
uv run scripts/run_threshold_sweep.py             # run each trial (portfolio + submission)
```

You still need to submit and archive each produced submission manually (or via your CI) so that `analyze_submission_tuning` sees the archived runs.

## Hybrid portfolio (if threshold tuning is too sparse)

If `lgbm_two_stage` still under-submits after the threshold sweep, use the **hybrid** strategy: primary portfolio from `lgbm_two_stage`, backfill from `phase3_repro` up to `target_per_buyer` per buyer.

1. Add `hybrid_lgbm_phase3` to `modelling.enabled_approaches` in `config.yaml`.
2. Run the pipeline as usual. The Snakefile will:
   - Build portfolios for `lgbm_two_stage` and `phase3_repro` (and other approaches).
   - For `hybrid_lgbm_phase3`, run `merge_portfolio_hybrid` (no training step).
   - Produce submissions for hybrid using that merged portfolio; cold-start uses `phase3_repro` scores for ranking.
3. Tune `target_per_buyer` (default 400) in the `portfolio_hybrid` rule if needed.
