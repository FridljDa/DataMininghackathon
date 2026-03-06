# `data/01_raw/plis_training.csv`

## Purpose
Historical transaction-level training data. Each row is a purchased item line, used to learn recurring demand patterns and build candidate Core Demand portfolios.

## File Format
- Delimiter: tab (`\t`)
- Header: yes
- Rows: 8,373,695 data rows (8,373,696 lines including header)
- Size: large (approximately 796 MB)

## Columns
- `orderdate`: transaction date (`YYYY-MM-DD`).
- `legal_entity_id`: buyer/company identifier.
- `set_id`: order/set identifier (groups line items within a purchase event).
- `sku`: product SKU identifier.
- `eclass`: E-Class category identifier.
- `manufacturer`: product brand/manufacturer name.
- `quantityvalue`: purchased quantity.
- `vk_per_item`: unit price/value per item.
- `estimated_number_employees`: approximate buyer size; may be missing.
- `nace_code`: primary buyer industry code (NACE).
- `secondary_nace_code`: optional secondary NACE code.

## Modeling Notes
- Core grain is line-item level, so aggregate to buyer-time and buyer-category views for stable recurring-demand signals.
- Main hierarchy options for prediction levels:
  - Level 1 candidate key: `eclass`
  - Level 2 candidate key: (`eclass`, `manufacturer`)
- Combine with NACE and company-size metadata for cold-start similarity features.
