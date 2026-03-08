---
name: submission-param-suggester
description: Analyzes submission tuning and score history CSV files, then recommends next config.yaml parameter values as a YAML patch. Use when tuning submission size/score trade-offs, selecting next hyperparameter trials, or when the user asks what to try next in config.yaml.
---

# Submission Parameter Suggester

## Purpose

Suggest next parameter values to try in `config.yaml` by reading:
- `data/17_submission_tuning/run_records_level*.jsonl` (preferred when present; one JSON object per run with full `params`)
- `data/17_submission_tuning/param_effects_level*.csv` and `run_metrics_level*.csv` (fallback)
- `data/16_scores_best/**/*.csv`
- `data/15_scores/**/*.csv`

Output is a YAML patch snippet for existing keys only.

## Quick Start

Run:

```bash
python .cursor/skills/submission-param-suggester/scripts/suggest_next_params.py \
  --config config.yaml \
  --scores-dir data/15_scores \
  --best-dir data/16_scores_best \
  --tuning-dir data/17_submission_tuning
```

## Behavior

1. Reads tuning data: when present, `run_records_level*.jsonl` (full params per run); otherwise `param_effects_level*.csv` and `run_metrics_level*.csv`.
2. Reads `score_summary_live.csv` files from score history directories.
3. Maps observed parameters to existing `config.yaml` keys:
   - `config_top_k_per_buyer` -> `modelling.selection.top_k_per_buyer`
   - `config_cold_start_top_k` -> `submission.cold_start_top_k`
   - `guardrail_min_orders` -> `modelling.selection.guardrails.min_orders`
   - `guardrail_min_months` -> `modelling.selection.guardrails.min_months`
   - `guardrail_high_spend` -> `modelling.selection.guardrails.high_spend`
   - `guardrail_min_avg_monthly_spend` -> `modelling.selection.guardrails.min_avg_monthly_spend`
4. Recommends values with supporting evidence from multiple levels where possible.
5. Prints YAML patch text only for keys that already exist in `config.yaml`.

## Options

- `--level 1|2|all` (default `all`): restrict evidence to one level.
- `--min-count N` (default `2`): minimum observations per parameter value.
- `--max-suggestions N` (default `6`): cap number of key updates in output.

## Output Contract

- Primary output: YAML patch snippet for `config.yaml`.
- Secondary output: short evidence notes for each suggested key.
- If no robust improvement is found, emits an empty patch and explains why.

