# `data/01_raw/features_per_sku.csv`

## Purpose
Product-attribute table for SKU enrichment. This file maps each SKU to one or more normalized feature key/value pairs.
According to organizer info, SKUs here are the SKUs that are part of `plis_training.csv`.

## File Format
- Delimiter: tab (`\t`)
- Header: yes
- Rows: 18,115,013 data rows (18,115,014 lines including header)
- Size: very large (approximately 1.8 GB)

## Top 3 Rows (Raw Sample)

| safe_synonym | sku | key | fvalue | fvalue_set |
|---|---|---|---|---|
| 1000_10_zusatzzeichen_ | 114-7428 | Folie-_verkehrsschilder_folie | RA0 | RA0 |
| 1000_10_zusatzzeichen_ | 164-74054 | Folie-_verkehrsschilder_folie | RA1 | RA1 |
| 1000_10_zusatzzeichen_ | 5921-A120.20.405 | Folie-_verkehrsschilder_folie | RA1 | RA1 |

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

## Pairwise Relationships
Sanity-check basis: sample only (first 200k rows), not full-file verified.

- `sku` <-> `safe_synonym`: many-to-many (`N:M`) in sample.
- `sku` <-> `key`: many-to-many (`N:M`) in sample.
- `key` <-> `fvalue`: many-to-many (`N:M`) in sample.
- `key` <-> `fvalue_set`: many-to-many (`N:M`) in sample.

Treat these as heuristic cardinalities for modeling decisions; strict constraints should be verified with a dedicated full pass if required.
