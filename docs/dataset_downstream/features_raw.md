# `data/08_features_raw/{mode}/level{level}/features_raw.parquet`

## Purpose

Raw feature table produced by `src/engineer_features_raw.py`: one row per candidate (entity key) with pass-through columns from candidates plus top-K SKU attribute columns from `data/02_raw/features_per_sku.csv`.

- **Level 1**: key = (`legal_entity_id`, `eclass`).
- **Level 2**: key = (`legal_entity_id`, `eclass`, `manufacturer`).

Downstream: `sanitize_features_raw` then `engineer_features_derived` (adds derived features to produce `features_all`).

## Columns

- **Key and history** (same as candidates): `legal_entity_id`, `eclass` (level 2: `manufacturer`), `n_orders`, `historical_purchase_value_total`, `orderdate_min`, `orderdate_max`, `orderdates_str`.
- **SKU attributes**: `sku_attr_<key>_<value>` (sanitized key/value from `features_per_sku`). Each value is the count of SKUs in that candidate’s purchase history that have that (key, value). Missing candidates get 0. Column set is data-dependent (top K keys and top K values per key from config).

## Configuration

- `config.inputs.features_per_sku`: path to the attribute CSV.
- `config.modelling.features_per_sku`: `top_k_keys`, `top_k_values_per_key`, `chunksize`.

To use SKU-attribute columns in modelling, add their names to `modelling.features.selected` (names appear after the first run or in feature analysis outputs).
