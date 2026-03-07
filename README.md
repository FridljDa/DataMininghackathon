# Challenge2 Pipeline

This repository uses Snakemake as the single entrypoint for the data pipeline.

## Run The Pipeline

From the repository root:

```bash
uv sync
uv run snakemake --cores 1
```

That command runs the default `all` target in `Snakefile`. The default modelling path uses the two-stage EU model (recurrence classifier + value regressor) and candidate set/selection policy described in `docs/modelling.md`.

To force generate everything, run 
```bash
uv run snakemake --cores 1
```

## Score run history

Each scoring run is archived so you never lose prior results and can see which commit produced which score.

- **Where:** Online runs under `data/12_scores/online/runs/`, offline under `data/12_scores/offline/runs/`.
- **Run folder format:** `runs/<run_id>/` with `run_id = <UTC timestamp>_<short git sha>` and an optional `_dirty` suffix when the working tree had uncommitted changes (e.g. `20250307_143022_abc1234_dirty`).
- **Contents:** Each run folder contains `score_summary.csv`, `score_details.parquet`, and `metadata.json` (commit, branch, dirty, created_at).
- **Index:** `data/12_scores/online/run_index.csv` and `data/12_scores/offline/run_index.csv` list every run with columns `run_id`, `commit_sha`, `branch`, `dirty`, `created_at`, `run_dir` for quick commit→score lookup.

The default pipeline archives the **online** score after scoring. To score and archive the **offline** pipeline, request the offline outputs:

```bash
uv run snakemake data/12_scores/offline/score_summary.csv data/12_scores/offline/runs/.last_archived --cores 1
```

To see which commit achieved a given score, open the run folder’s `metadata.json` or look up the run in the corresponding `run_index.csv`.

## Notes

- Input/output paths are configured in `config.yaml`.
- Raw input files expected by the current workflow:
  - `data/01_raw/plis_training.csv`
  - `data/01_raw/customer_test.csv`
  - `data/01_raw/les_cs.csv`

  - `data/02_raw/plis_training.csv`
  - `data/02_raw/customer_test.csv`
  - `data/02_raw/les_cs.csv`

## Submit Predictions

Use `src/submit.py` to upload predictions and see scores:

```bash
# Challenge 1 (parquet)
uv run src/submit.py --challenge 1 --file data/11_submission/online/submission.parquet

# Challenge 2 (csv, default level 2)
uv run src/submit.py --challenge 2 --file data/11_submission/online/submission.csv

# Challenge 2 with explicit level
uv run src/submit.py --challenge 2 --file data/11_submission/online/submission.csv --level 1
```

Set `portal_credentials.team` and `portal_credentials.password` in `config.yaml`; the script logs in to the evaluator portal, uploads the file, and waits for the scoring result.

## Troubleshooting

- **LightGBM / libomp on macOS:** If `train_lgbm` fails with `Library not loaded: @rpath/libomp.dylib`, install OpenMP: `brew install libomp`.
