# `data/01_raw/nace_codes.csv`

## Purpose
Reference lookup for NACE industry taxonomy. Use this file to map NACE codes to human-readable descriptions and hierarchy levels.

## File Format
- Delimiter: tab (`\t`)
- Header: yes
- Rows: 975 data rows (976 lines including header)

## Columns
- `nace_code`: NACE code for the row (can be different granularities, e.g. 2/3/4 digits).
- `n_nace_description`: description of `nace_code`.
- `toplevel_section`: top-level NACE section letter (for example `A`, `B`, `C`).
- `toplevel_section_description`: description of top-level section.
- `nace_2digits`: associated 2-digit NACE division code.
- `nace_2digits_description`: description of 2-digit division.
- `nace_3digits`: associated 3-digit NACE group code.
- `nace_3digits_description`: description of 3-digit group.

## Modeling Notes
- Join by `nace_code` from buyer/order tables to derive robust industry features.
- `nace_code -> n_nace_description` is one-to-one in this file.
- `n_nace_description -> nace_code` is not one-to-one (same label can appear at multiple code granularities).
