# `data/01_raw/les_cs.csv`

## Purpose
Customer split-definition file for evaluation setup. Each row marks one legal entity (`legal_entity_id`) as either cold-start or predict-future scenario.

## File Format
- Delimiter: comma (`,`)
- Header: yes
- Rows: 101 data rows (102 lines including header)

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
