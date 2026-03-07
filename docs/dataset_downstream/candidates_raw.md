# `data/07_candidates/{mode}/level{level}/candidates_raw.parquet`

## Purpose

Intermediate candidate table for Level 1 (E-Class) or Level 2 (E-Class + manufacturer) recommendation modelling.

Each row is one candidate key that passed eligibility filters in the lookback window. Level 1: (`legal_entity_id`, `eclass`); Level 2: (`legal_entity_id`, `eclass`, `manufacturer`). Output contains **key columns only**; aggregates are computed later by `src/engineer_features_raw.py`, which writes `data/08_features_raw/{mode}/level{level}/features_raw.parquet`. Derived feature engineering then produces `data/09_features_derived/{mode}/level{level}/features_all.parquet`.

Practical definition from `src/generate_candidates.py`:

- Restrict PLIs to training period (`orderdate <= train_end`) and warm/testing buyers only (`task in ["predict future", "testing"]`).
- Lookback window: `[train_end - lookback_months, train_end]` (inclusive). Per entity key, compute in that window:
  - `n_orders_in_lookback`
  - `lookback_spend` = sum(quantityvalue × vk_per_item)
- Keep only keys with `n_orders_in_lookback >= min_order_frequency` and `lookback_spend >= min_lookback_spend` (from `modelling.candidates` in `config.yaml`).
- **Level 1:** *Seen* keys (passing the above) are unioned with the trending cross (hot buyers × `data/07_candidates/trending_classes.csv` eclasses), then deduplicated.
- **Level 2:** Output is the set of seen keys only (no trending cross).

## File Format

- Format: Parquet
- Path pattern: `data/07_candidates/{mode}/level{level}/candidates_raw.parquet` (e.g. `online/level1/`, `offline/level2/`).
- Columns: **key only** — Level 1: `legal_entity_id`, `eclass`; Level 2: `legal_entity_id`, `eclass`, `manufacturer`.

## Columns

- `legal_entity_id`: buyer/customer identifier.
- `eclass`: E-Class product category candidate for the buyer.
- `manufacturer`: (Level 2 only) manufacturer part of the candidate key.

## Modeling Notes

- This is a warm/startable candidate universe, not final model features and not final predictions.
- Candidate eligibility is controlled by `config.yaml` under `modelling.candidates`:
  - `lookback_months`: length of lookback window (months) ending at `train_end`
  - `min_order_frequency`: minimum number of orders in lookback for the key to be included
  - `min_lookback_spend`: minimum EUR spend in lookback for the key to be included
- These values are passed from the Snakefile into `src/generate_candidates.py` and applied before writing this parquet.
- Main downstream: `src/engineer_features_raw.py` joins these keys with PLIs/customer/features_per_sku to build `features_raw` (with aggregates and SKU attributes); `src/engineer_features_derived.py` then adds derived features. SKU-attribute columns in `features_raw` are named `sku_attr_<key>_<value>` (sanitized). To use them in modelling, add the desired names to `modelling.features.selected` (names are data-dependent).
- Why this dataset exists: shrinks the modelling space to keys with minimal historical signal in the lookback window before expensive feature engineering and training.

## Pairwise Relationships

- Each row is a unique candidate key: (`legal_entity_id`, `eclass`) for Level 1; (`legal_entity_id`, `eclass`, `manufacturer`) for Level 2.
- `legal_entity_id` and `eclass` (and `manufacturer` at Level 2) are many-to-many globally.

