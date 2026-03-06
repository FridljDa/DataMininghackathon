# `data/01_raw/customer_test.csv`

## Purpose

Metadata table for the challenge customer set (legal entities in `les_cs.csv`). Each row represents one customer to score.

## File Format

- Delimiter: tab (`\t`)
- Header: yes
- Rows: 100 data rows (101 lines including header)

## Top 3 Rows (Raw Sample)


| legal_entity_id | estimated_number_employees | nace_code | secondary_nace_code | task       |
| --------------- | -------------------------- | --------- | ------------------- | ---------- |
| 41361768        | 6546                       | 861       |                     | cold start |
| 41525307        | 630                        | 3511      |                     | cold start |
| 60218513        | 1919                       | 1089      |                     | cold start |


## Columns

- `legal_entity_id`: customer (legal entity) identifier.
- `estimated_number_employees`: approximate buyer company size; may be missing.
- `nace_code`: primary industry code (NACE).
- `secondary_nace_code`: optional secondary industry code (NACE); often empty.
- `task`: scenario flag for the customer.
  - Observed values in this file:
    - `cold start`
    - `predict future`

## Modeling Notes

- `task` should be used to branch strategy:
  - `cold start` <-> `cs = 0` in `les_cs.csv`
  - `predict future` <-> `cs = 1` in `les_cs.csv`
- This file is metadata-only (no PLIs). Link to transactional behavior via `legal_entity_id` in `plis_training.csv`.
- Join `nace_code` and `secondary_nace_code` with `nace_codes.csv` for hierarchical industry features.

## Pairwise Relationships

Sanity-check basis: exact (full file).

- `legal_entity_id` -> `task`: many-to-one (`N:1`).
- `legal_entity_id` -> `nace_code`: many-to-one (`N:1`).
- `task` <-> `nace_code`: many-to-many (`N:M`).
- Cross-file (`customer_test.csv` vs `les_cs.csv`):
  - `legal_entity_id` sets align 1:1 (same 100 entities).
  - `task` <-> `cs` is 1:1 at value level:
    - `cold start` <-> `0`
    - `predict future` <-> `1`

