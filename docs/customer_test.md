# `data/01_raw/customer_test.csv`

## Purpose
Test-set buyer metadata used for inference. Each row represents one buyer (`legal_entity_id`) to score in the challenge.

## File Format
- Delimiter: tab (`\t`)
- Header: yes
- Rows: 100 data rows (101 lines including header)

## Columns
- `legal_entity_id`: buyer/company identifier.
- `estimated_number_employees`: approximate buyer company size; may be missing.
- `nace_code`: primary industry code (NACE).
- `secondary_nace_code`: optional secondary industry code (NACE); often empty.
- `task`: scenario flag for the buyer.
  - Observed values in this file:
    - `cold start`
    - `predict future`

## Modeling Notes
- `task` should be used to branch strategy between low-history vs history-based prediction behavior.
- Join `nace_code` and `secondary_nace_code` with `nace_codes.csv` for hierarchical industry features.
