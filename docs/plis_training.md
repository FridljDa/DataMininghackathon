# `data/01_raw/plis_training.csv`

## Purpose
Historical transaction-level training data. Each row is a purchased item line, used to learn recurring demand patterns and build candidate Core Demand portfolios.
Organizer definition: this is the DE PLI universe after applying the challenge split logic from `les_cs.csv`.

## File Format
- Delimiter: tab (`\t`)
- Header: yes
- Rows: 8,373,695 data rows (8,373,696 lines including header)
- Size: large (approximately 796 MB)

## Top 3 Rows (Raw Sample)

| orderdate | legal_entity_id | set_id | sku | eclass | manufacturer | quantityvalue | vk_per_item | estimated_number_employees | nace_code | secondary_nace_code |
|---|---|---|---|---|---|---|---|---|---|---|
| 2023-01-11 | 10063702 | 20240502211517-189794 | 5497-2606083 | 27141104 | WAGO Kontakttechnik | 50 | 0.29 | 2 | 3513 | 2712 |
| 2023-01-11 | 10063702 | 20240502211517-17767700 | 5497-6404351 | 27141104 | WAGO Kontakttechnik | 25 | 0.464 | 2 | 3513 | 2712 |
| 2023-01-11 | 10063702 | 20240502211517-13291417 | 721-01914478 | 27242202 | Siemens | 1 | 293.01 | 2 | 3513 | 2712 |

## Columns
- `orderdate`: transaction date (`YYYY-MM-DD`).
- `legal_entity_id`: buyer/company identifier.
- `set_id`: order/set identifier used to tie multiple row-level PLIs into one purchase event (basket/order session).  
  Practical interpretation: if several rows share the same `set_id`, they were bought together as part of the same checkout/order context, even when SKUs differ.  
  Use in modeling: aggregate per (`legal_entity_id`, `set_id`) to build basket-level features (basket size, category mix, co-purchase patterns), and avoid counting each line item as an independent event.
- `sku`: product SKU identifier.
- `eclass`: E-Class category identifier.
- `manufacturer`: product brand/manufacturer name.
- `quantityvalue`: purchased quantity.
- `vk_per_item`: unit price/value per item.
- `estimated_number_employees`: approximate buyer size; may be missing.
- `nace_code`: primary buyer industry code (NACE).
- `secondary_nace_code`: optional secondary NACE code.

## Modeling Notes
- Split logic with `les_cs.csv` (cutoff `2025-07-01`):
  - For `cs = 0` customers: all PLIs are removed from this training file.
  - For `cs = 1` customers: only PLIs before the cutoff are included.
- Observed date coverage (full file): `2023-01-01` to `2025-12-31`.
- Scope clarification for scored challenge entities (`les_cs.csv` / `customer_test.csv`):
  - For warm (`cs = 1`) entities, rows in `plis_training.csv` end at `2025-06-30`.
  - Data after `2025-07-01` for these warm entities is held out for evaluation.
- The full file also contains many non-challenge entities; those rows can extend beyond the warm cutoff and should not be used to infer the scored split horizon.
- Core grain is line-item level, so aggregate to buyer-time and buyer-category views for stable recurring-demand signals.
- Main hierarchy options for prediction levels:
  - Level 1 candidate key: `eclass`
  - Level 2 candidate key: (`eclass`, `manufacturer`)
- Combine with NACE and company-size metadata for cold-start similarity features.

## Pairwise Relationships
Sanity-check basis: sample only (first 200k rows), not full-file verified.

- `set_id` <-> `legal_entity_id`: many-to-many (`N:M`) in sample.
- `set_id` <-> `orderdate`: many-to-many (`N:M`) in sample.
- `sku` <-> `eclass`: many-to-many (`N:M`) in sample.
- `sku` <-> `manufacturer`: many-to-many (`N:M`) in sample.
- `eclass` <-> `manufacturer`: many-to-many (`N:M`) in sample.

Use these as practical assumptions for feature engineering; re-check exact cardinality if a strict constraint is needed in production logic.
