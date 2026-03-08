# Sweep override configs

Override files in this directory are merged on top of the repo root `config.yaml` when running the threshold sweep script. Each file should contain only the keys you want to override, using the same nested structure as the base config.

The current set is a **focused level-1 experiment**: hybrid (`hybrid_lgbm_phase3` = lgbm_two_stage primary + phase3_repro backfill) with a small threshold sweep (2 runs). Overrides set `modelling.enabled_approaches` so that both approaches and the hybrid merge are built for the same `run_id`.

## Format

- **Optional** top-level key `_sweep` (reserved; stripped before merge):
  - `level`: `"1"` or `"2"` — which level this trial runs for.
  - `approach`: e.g. `lgbm_two_stage` — which modelling approach.

- All other keys are deep-merged onto the loaded base config. Use the same paths as in `config.yaml` (e.g. `modelling.selection.by_level."1"`, `submission.by_level."1"`).

## Usage

From repo root:

```bash
uv run scripts/run_level2_threshold_sweep.py --dry-run
uv run scripts/run_level2_threshold_sweep.py --cores 1
```

The script discovers override files by name (e.g. `config_1.yaml`, `config_2.yaml`) in sorted order. Use `--sweeps-dir` to point to another directory.
