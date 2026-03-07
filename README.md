# Challenge2 Pipeline

This repository uses Snakemake as the single entrypoint for the data pipeline.

## Run The Pipeline

From the repository root:

```bash
uv sync
uv run snakemake --cores 1
```

That command runs the default `all` target in `Snakefile`. The pipeline runs every approach listed in `modelling.enabled_approaches` (e.g. baseline, lgbm_two_stage, pass_through), producing scores, portfolio, submission, and scores per approach; see `docs/modelling.md` for the candidate set and selection policy.

To force generate everything, run 
```bash
uv run snakemake --cores 1
```

## Score run history

Each scoring run is archived per approach so you never lose prior results and can see which commit produced which score.

- **Where:** Online runs under `data/14_scores/online/runs/<approach>/`, offline under `data/14_scores/offline/runs/<approach>/`.
- **Run folder format:** `runs/<approach>/<run_id>/` with `run_id = <UTC timestamp>_<short git sha>` and an optional `_dirty` suffix when the working tree had uncommitted changes (e.g. `20250307_143022_abc1234_dirty`).
- **Contents:** Each run folder contains `score_summary.csv`, `score_details.parquet`, and `metadata.json` (commit, branch, dirty, created_at).
- **Index:** `data/14_scores/online/run_index_<approach>.csv` and `data/14_scores/offline/run_index_<approach>.csv` list every run for that approach with columns `run_id`, `commit_sha`, `branch`, `dirty`, `created_at`, `run_dir` for quick commit→score lookup.

The default pipeline builds and archives scores for all enabled approaches (online and offline). To request a single offline output for one approach:

```bash
uv run snakemake data/14_scores/offline/baseline/score_summary.csv data/14_scores/offline/baseline/runs/.last_archived --cores 1
```

To see which commit achieved a given score, open the run folder’s `metadata.json` or look up the run in the corresponding `run_index_<approach>.csv`.

## Notes

- Input/output paths and directory layout are defined in the Snakefile; `config.yaml` holds tunable parameters, raw input file refs, and portal credentials.
- Raw input files expected by the current workflow:
  - `data/01_raw/plis_training.csv`
  - `data/01_raw/customer_test.csv`
  - `data/01_raw/les_cs.csv`

  - `data/02_raw/plis_training.csv`
  - `data/02_raw/customer_test.csv`
  - `data/02_raw/les_cs.csv`

## Submit Predictions

The pipeline produces one submission per enabled approach under `data/13_submission/online/<approach>/submission.csv`. The default `snakemake` target uploads each of these to the Unite evaluator (challenge 2). To upload a specific approach manually:

```bash
# Challenge 1 (parquet) — if your pipeline produces parquet
uv run src/submit.py --challenge 1 --file data/13_submission/online/lgbm_two_stage/submission.parquet

# Challenge 2 (csv, default level 2)
uv run src/submit.py --challenge 2 --file data/13_submission/online/lgbm_two_stage/submission.csv

# Challenge 2 with explicit level
uv run src/submit.py --challenge 2 --file data/13_submission/online/lgbm_two_stage/submission.csv --level 1
```

Set `portal_credentials.team` and `portal_credentials.password` in `config.yaml`; the script logs in to the evaluator portal, uploads the file, and waits for the scoring result.

## Troubleshooting

- **LightGBM / libomp on macOS:** If `train_lgbm` fails with `Library not loaded: @rpath/libomp.dylib`, install OpenMP: `brew install libomp`.
