# Challenge2 Pipeline

This repository uses Snakemake as the single entrypoint for the data pipeline.

## Run The Pipeline

From the repository root:

```bash
uv sync
uv run snakemake --cores 1
```

That command runs the default `all` target in `Snakefile`.

To force generate everything, run 
```bash
uv run snakemake --cores 1
```

## Notes

- Input/output paths are configured in `config.yaml`.
- Raw input files expected by the current workflow:
  - `data/01_raw/plis_training.csv`
  - `data/01_raw/customer_test.csv`
  - `data/01_raw/les_cs.csv`

## Env
Create a file `.env` at the root of the project with credentials:
```sh
TEAM=...
PASSWORD=...
```
  - `data/02_raw/plis_training.csv`
  - `data/02_raw/customer_test.csv`
  - `data/02_raw/les_cs.csv`

## Submit Predictions

Use `src/submit.py` to upload predictions and see scores:

```bash
# Challenge 1 (parquet)
uv run src/submit.py --challenge 1 --file data/10_submission/online/submission.parquet

# Challenge 2 (csv, default level 2)
uv run src/submit.py --challenge 2 --file data/10_submission/online/submission.csv

# Challenge 2 with explicit level
uv run src/submit.py --challenge 2 --file data/10_submission/online/submission.csv --level 1
```

The script reads `TEAM` and `PASSWORD` from the `.env` file above, logs in to the evaluator portal, uploads the file, and waits for the scoring result.
