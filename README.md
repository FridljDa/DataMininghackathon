# Challenge2 Pipeline

This repository uses Snakemake as the single entrypoint for the data pipeline.

## Run The Pipeline

From the repository root:

```bash
uv sync
uv run snakemake --cores 1
```

That command runs the default `all` target in `Snakefile` and produces:

- `data/03_meta/customer.csv`
- `data/06_plots/` (EDA: seasonal volume, violin by task, task distribution, cs/task heatmap, NACE by task)
- `data/10_submission/submission.csv`

## Notes

- Input/output paths are configured in `config.yaml`.
- Raw input files expected by the current workflow:
  - `data/02_raw/plis_training.csv`
  - `data/02_raw/customer_test.csv`
  - `data/02_raw/les_cs.csv`
