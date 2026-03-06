# Challenge Glossary

- `LE` (legal entity): customer/company identifier.
- `PLI` (product line item): one ordered product entry with quantity (line in a receipt/order set).
- `SKU` (stock keeping unit): product identifier.
- `features`: product attributes (for example dimensions, weight, material).
- `NACE`: company activity classification (industry code taxonomy).
- `cs`: cold-start split flag in `les_cs.csv`.
  - `0`: cold-start customer (no customer history in training).
  - `1`: predict-future customer (history before cutoff in training, future held out).

## Split Semantics (Organizer Definition)
- `plis_training.csv`:
  - contains all DE PLIs except:
    - all PLIs of customers with `cs = 0`
    - PLIs after `2025-07-01` for customers with `cs = 1`
- Held-out evaluation side:
  - contains:
    - all PLIs of customers with `cs = 0`
    - PLIs after `2025-07-01` for customers with `cs = 1`
