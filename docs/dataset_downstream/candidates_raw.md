# `data/07_candidates/{mode}/candidates_raw.parquet`

## Purpose

Intermediate candidate table for Level 1 (E-Class) recommendation modelling.

Each row is one buyer-candidate pair: (`legal_entity_id`, `eclass`) that passed candidate eligibility filters in the lookback window.  
This is the first downstream dataset after split preparation and is the direct input to the feature pipeline: `src/engineer_features_raw.py` writes non-derived columns to `data/08_features_raw/{mode}/features_raw.parquet`, then `src/engineer_features_derived.py` adds derived features and writes `data/09_features_derived/{mode}/features_all.parquet`.

Practical definition from `src/generate_candidates.py`:

- Restrict PLIs to training period (`orderdate <= train_end`) and warm/testing buyers only (`task in ["predict future", "testing"]`).
- In lookback window `L` months:
  - keep pair when `n_orders_in_L >= eta`
  - and `s_lookback >= tau`
- For kept pairs, store full-train aggregates (`n_orders`, `historical_purchase_value_total`, min/max order date, order months).

## File Format

- Format: Parquet
- Path pattern: `data/07_candidates/online/candidates_raw.parquet` and `data/07_candidates/offline/candidates_raw.parquet`
- Current columns: 8
- Current rows:
  - online: 25,237
  - offline: 47,975

## Top 3 Rows (Raw Sample)

Sample from `data/07_candidates/online/candidates_raw.parquet`:

| legal_entity_id | eclass | n_orders | historical_purchase_value_total | orderdate_min | orderdate_max | t_last | orderdates_str |
|---:|---:|---:|---:|---|---|---|---|
| 41165867 | 19010107 | 1 | 551.31 | 2024-10-30 00:00:00 | 2024-10-30 00:00:00 | 2024-10-30 00:00:00 | ['2024-10'] |
| 41165867 | 19010108 | 1 | 391.67 | 2024-12-12 00:00:00 | 2024-12-12 00:00:00 | 2024-12-12 00:00:00 | ['2024-12'] |
| 41165867 | 19019090 | 2 | 184.15 | 2024-09-10 00:00:00 | 2024-11-07 00:00:00 | 2024-11-07 00:00:00 | ['2024-09', '2024-11'] |

## Columns

- `legal_entity_id`: buyer/customer identifier.
- `eclass`: Level 1 product category candidate for the buyer.
- `n_orders`: number of line-level purchase records for this (`legal_entity_id`, `eclass`) in train period (`orderdate <= train_end`).
- `historical_purchase_value_total`: total spend proxy in train period, computed as `sum(quantityvalue * vk_per_item)`.
- `orderdate_min`: earliest observed purchase date for this buyer-eclass in train period.
- `orderdate_max`: latest observed purchase date for this buyer-eclass in train period.
- `t_last`: last observed purchase date (currently equal to `orderdate_max`; kept as explicit recency anchor for downstream features).
- `orderdates_str`: unique order months encoded as `YYYY-MM` strings for parquet-safe serialization (list-like field).

## Modeling Notes

- This is a warm/startable candidate universe, not final model features and not final predictions.
- Candidate inclusion is controlled by `config.yaml` modelling parameters:
  - `train_end`: `2024-12-31`
  - `lookback_months` (`L`): `18`
  - `min_order_frequency` (`eta`): `1`
  - `min_lookback_spend` (`tau`): `100.0`
- `historical_purchase_value_total >= 100` appears in outputs because `tau = 100` and eligibility is spend-filtered in lookback.
- Date coverage in current files:
  - global `orderdate_min`: `2023-01-01`
  - global `orderdate_max`: `2024-12-31`
- Main downstream dependency:
  - consumed by `src/engineer_features_raw.py` (pass-through to `data/08_features_raw`), then `src/engineer_features_derived.py` reconstructs periods from `orderdates_str` and derives recurrence/recency/trend/economic features into `data/09_features_derived`.
- Why this dataset exists:
  - shrinks the modelling space to buyer-category pairs with minimal historical signal before expensive feature engineering and training.

## Pairwise Relationships

Sanity-check basis: current generated files (full parquet, online + offline).

- (`legal_entity_id`, `eclass`) is unique per row (candidate key).
- `legal_entity_id` <-> `eclass`: many-to-many (`N:M`) globally.
- `t_last` == `orderdate_max` for all rows by construction in generator.
- Online snapshot:
  - buyers: 47
  - unique eclasses: 2,911
- Offline snapshot:
  - buyers: 97
  - unique eclasses: 3,387

