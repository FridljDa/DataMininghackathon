# `data/01_raw/features_per_sku.csv`

## Purpose
Product-attribute table for SKU enrichment. This file maps each SKU to one or more normalized feature key/value pairs.
According to organizer info, SKUs here are the SKUs that are part of `plis_training.csv`.

## File Format
- Delimiter: tab (`\t`)
- Header: yes
- Rows: 18,115,013 data rows (18,115,014 lines including header)
- Size: very large (approximately 1.8 GB)

## Columns
- `safe_synonym`: normalized product synonym/group label.
- `sku`: product SKU identifier.
- `key`: feature name (attribute key).
- `fvalue`: feature value for that key.
- `fvalue_set`: canonicalized feature value set representation.

## Modeling Notes
- Treat this as a long-form key-value feature store (multiple rows per SKU).
- Expect high cardinality and sparse coverage across keys; prefer aggregation or feature hashing/encoding rather than wide one-hot expansion at full scale.
- Useful for level-3 style abstraction and clustering where product characteristics matter beyond `eclass`.
- Coverage is tied to training SKUs, so expect missing feature rows for SKUs that only appear in held-out/evaluation PLIs.
