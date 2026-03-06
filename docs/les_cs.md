# `data/01_raw/les_cs.csv`

## Purpose
Customer split-definition file for evaluation setup. Each row marks one legal entity (`legal_entity_id`) as either cold-start or predict-future scenario.

## File Format
- Delimiter: comma (`,`)
- Header: yes
- Rows: 101 data rows (102 lines including header)

## Top 3 Rows (Raw Sample)

| legal_entity_id | cs |
|---|---|
| 41361768 | 0 |
| 41525307 | 0 |
| 60218513 | 0 |

## Columns
- `legal_entity_id`: customer (legal entity) identifier.
- `cs`: scenario flag.
  - `0`: cold start customer.
  - `1`: predict-future (warm) customer.

## How This Drives the Train/Test Split
- `cs = 0`:
  - `plis_training.csv` excludes all PLIs for this customer.
  - Held-out evaluation side contains all PLIs for this customer.
- `cs = 1`:
  - `plis_training.csv` contains PLIs only up to `2025-07-01` (exclusive of later dates).
  - Held-out evaluation side contains PLIs after `2025-07-01`.

## Consistency Notes
- `customer_test.csv` should align to this population of legal entities.
- `task` in `customer_test.csv` maps naturally to this flag:
  - `cold start` <-> `cs = 0`
  - `predict future` <-> `cs = 1`
- This file is the canonical source for determining cold-start vs predict-future behavior.
- `plis_training.csv` includes many legal entities outside this file; those non-challenge entities may have rows after `2025-07-01`.
- For split interpretation on scored buyers, always filter to entities in `les_cs.csv` / `customer_test.csv`.

## Pairwise Relationships
Sanity-check basis: exact (full file).

- `legal_entity_id` -> `cs`: many-to-one (`N:1`).
- `cs` -> `legal_entity_id`: one-to-many (`1:N`) by definition.
- Cross-file with `customer_test.csv`:
  - `legal_entity_id` sets are 1:1 aligned.
  - `cs` <-> `task` is 1:1 at value level (`0` <-> `cold start`, `1` <-> `predict future`).
