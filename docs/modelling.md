# Modelling Approach — Core Demand Prediction (Level 1)

## 0. Global Sets & Notation

Given from data/problem setup:

- $ \mathcal{B} $: all buyers
- $ \mathcal{B}_{\text{warm}} \subseteq \mathcal{B} $: warm buyers (`cs = 1`)
- $ \mathcal{B}_{\text{cold}} = \mathcal{B} \setminus \mathcal{B}_{\text{warm}} $: cold buyers (`cs = 0`)
- $ \mathcal{E} $: all E-Class identifiers (eclasses)
- $ \text{history}(b) \subseteq \mathcal{E} $: eclasses buyer $ b $ purchased in observed history

Defined by our modelling choices:

- $ \mathcal{C}_b \subseteq \mathcal{E} $: candidate set we construct for buyer $ b $ (Section 4)
- $ \hat{S}_b \subseteq \mathcal{C}_b $: final predicted set selected by the policy (Section 7)

Core symbols:
- $ T $: training cutoff timestamp
- $ L $: lookback window
- $ F $: fixed fee per predicted element (monthly)
- $ b $ indexes buyers, $ e $ indexes eclasses

---

## 1. Problem Framing

For each scored warm buyer $ b \in \mathcal{B}_{\text{warm}} $ we must select a portfolio $ \hat{S}_b \subseteq \mathcal{E} $ of E-Class identifiers that maximises **net economic benefit**:

$$
\text{Score} = \sum_{b \in \mathcal{B}_{\text{warm}}} \left[ \sum_{e \in \hat{S}_b} \text{Savings}(b, e) - |\hat{S}_b| \cdot F \right]
$$

where $ F = €10 $ is the fixed monthly fee per predicted element and savings are realised only when prediction $ e $ matches actual future purchases.

Savings per hit are a function of future spend (exact formula is a black box — savings scale non-linearly with spend, roughly with its square root, combined with demand frequency). The scoring function may change; do not hardcode a fixed savings rate.

Selection is precision-first: every element that is not a true recurring need pays a fee with zero return.

---

## 2. Data & Split

Raw inputs:
- `data/02_raw/plis_training.csv` (all historical PLIs)
- `data/02_raw/customer_test.csv` (buyers to predict for online submission)

Pipeline split used by modelling/scoring:

1. Build base customer metadata at `data/03_meta/customer.csv`.
2. Create split metadata at `data/04_customer/customer.csv`:
   - relabel `test_customers_count = 50` buyers from `task=none` to `task=testing`
   - selection is stratified to match the warm-buyer purchase-value distribution
   - matching uses pre-cutoff spend only (`orderdate < 2025-07-01`) to avoid leakage
3. Split PLIs using `cutoff_date = 2025-07-01` into:
   - `data/05_training_validation/plis_training.csv`
   - `data/05_training_validation/plis_testing.csv`
   - rule: rows for `task=testing` buyers with `orderdate >= cutoff_date` go to `plis_testing`; all other rows stay in `plis_training`

How these split files are used:
- **Online pipeline:** train on split `plis_training`, predict for `customer_test`, score externally on hidden data.
- **Offline pipeline:** train on split `plis_training`, predict only split `task=testing` buyers, score against split `plis_testing`.

Focus for this modelling document: warm-buying recurrence logic for Level 1 (`cluster = eclass`), primarily evaluated through the offline split.

Available columns used from `plis_training`:

- `orderdate`, `legal_entity_id`, `eclass`, `manufacturer`
- `quantityvalue`, `vk_per_item` (spend proxy: $ s = \text{quantityvalue} \times \text{vk\_per\_item} $)
- `nace_code`, `estimated_number_employees` (buyer context)

---

## 3. Offline Validation Setup

Within `data/05_training_validation/plis_training.csv`, use a fixed temporal train/validation window from config:

- **Train period:** `2023-01-01` — `2024-12-31`
- **Validation period:** `2025-01-01` — `2025-06-30`

For each candidate pair $ (b, e) $, validation targets are computed from PLIs in the validation window.

Label a pair $ (b, e) $ as **positive** if validation orders meet:

$$
\text{label}(b, e) = \mathbf{1}\left[\text{orders in val}(b, e) \geq n_{\min}\right]
$$

with current default $ n_{\min} = 1 $ (`modelling.windows.validation.n_min_label`).

Validation spend is aggregated as:

$$
s_{\text{val}}(b,e) = \sum_{\text{val rows}} \text{quantityvalue} \times \text{vk\_per\_item}
$$

Default target remains binary recurrence: bought again vs not bought again.

> **Evaluator behavior note:** Predictions are set-based per buyer and cluster. Duplicate rows for the same $ (b, e) $ are sanitized and counted once by the organizer scorer.

Evaluate using an approximation of the euro score on validation:

$$
\widehat{\text{Score}}_{\text{val}} = \sum_{b \in \mathcal{B}_{\text{warm}}} \left[ \sum_{e \in \hat{S}_b} r \cdot s_{\text{val}}(b,e) \cdot \mathbf{1}[\text{label}=1] - |\hat{S}_b| \cdot F \right]
$$

> **Important:** The validation window is 6 months but the hidden evaluation covers approximately 1 month. When using $ \widehat{\text{Score}}_{\text{val}} $ to tune thresholds (EU threshold, $ K $, etc.), divide spend by 6 to obtain a per-month estimate. Without this normalisation the local score will be ~6× the real evaluation score, causing the EU threshold to be set too low (too many elements included).

---

## 4. Candidate Generation

For each warm buyer, restrict the candidate set to reduce fee leakage.

We construct $ \mathcal{C}_b $ as:

$$
\mathcal{C}_b = \{ e \in \text{history}(b) \mid n_{\text{orders}}(b,e, L) \ge \eta \;\land\; s_{\text{lookback}}(b,e,L) \ge \tau \}
$$

Interpretation:
- $ \text{history}(b) \subseteq \mathcal{E} $ is the set of eclasses buyer $ b $ has ever purchased
- $ n_{\text{orders}}(b,e, L) $ is the number of orders of eclass $ e $ by buyer $ b $ within the lookback window $ L $
- $ s_{\text{lookback}}(b,e,L) = \sum_{\text{rows in } L} \text{quantityvalue} \times \text{vk\_per\_item} $: total spend on eclass $ e $ by buyer $ b $ in the lookback window (EUR)
- $ T $ is the training cutoff timestamp
- $ L $ is the lookback window (default: 18 months)
- $ \eta $ is the minimum order frequency threshold (default: 1)
- $ \tau $ is the minimum lookback spend threshold in EUR (default: 100)
- So $ \mathcal{C}_b $ keeps only eclasses from $ b $'s history that were bought at least $ \eta $ times and with at least $ \tau $ EUR spend in the lookback window
- Therefore, by construction, $ \mathcal{C}_b \subseteq \text{history}(b) \subseteq \mathcal{E} $

---

## 5. Feature Engineering

For each candidate pair $ (b, e) $ with $ b \in \mathcal{B}_{\text{warm}} $ and $ e \in \mathcal{C}_b $ in the train period, derive a flexible feature set from these families:

### Core feature families
- Frequency/intensity signals (how often and how consistently a pair is purchased).
- Recency and inter-purchase timing signals (time since last activity, gap behavior).
- Economic/value signals (spend magnitude, spend concentration, value stability).
- Calendar and seasonality signals (month/quarter/year effects and cyclical timing).
- Momentum/trend signals (recent acceleration/deceleration versus prior periods).
- Buyer-context signals (size/sector metadata and related static attributes).

### Tenure-normalized (average monthly) features
Raw counts and spend are not comparable when buyers joined at different times. The pipeline adds quotient features that normalize by observed tenure so late joiners are comparable to long-tenure buyers:
- **Buyer/pair tenure:** `buyer_tenure_months`, `pair_tenure_months`, `effective_lookback_months` (denominators; no leakage, clipped ≥ 1).
- **Average monthly:** `avg_monthly_orders_buyer_tenure`, `avg_monthly_orders_pair_tenure`, `avg_monthly_spend_buyer_tenure`, `avg_monthly_spend_pair_tenure`, `avg_monthly_orders_in_lookback`, `avg_monthly_spend_in_lookback`.
Selection guardrails can optionally use `avg_monthly_spend_buyer_tenure` (`modelling.selection.guardrails.min_avg_monthly_spend`). The phase3_repro approach can score using monthlyized lookback rates via `modelling.approaches.phase3_repro.use_monthly_lookback_rates`.

### Design principles
- Keep the concrete feature list configurable and versioned with experiments.
- Use only information available up to the train cutoff (strict no-leakage rule).
- Prefer robust transforms/encodings for skewed values and cyclical calendar fields.
- Revisit, add, or remove features as scoring objectives and model behavior evolve.

The exact active columns for any run are controlled by pipeline configuration and may change over time.

**Feature analysis artifacts:** The pipeline produces:

- `feature_summary.csv` — Per-feature descriptive stats (null/zero rate, quantiles, cardinality) and target-aware stats (univariate signal vs validation recurrence label and vs positive-case spend). Primary inspection artifact for deciding which features to keep.
- `feature_redundancy.csv` — Pairs of numeric features with high Spearman correlation (e.g. |ρ| ≥ 0.85) for redundancy pruning.
- `feature_suggestions.yaml` — Advisory list of suggested features: hard filters (null rate, variance, cardinality) are applied, then one representative per correlated group is kept (ranked by target signal). For manual copy into `config.yaml` under `modelling.features.selected`.

**Refreshing the feature list:** `feature_suggestions.yaml` answers “which features look best for modelling?”; `modelling.features.selected` and the resulting `data/10_features_selected` output are the explicit contract used by training and scoring. Copy from the suggestion file into config when you want to align the pipeline with the heuristic; the pipeline does not modify config automatically.

---

## 6. Modelling

### Baseline (v1)

This baseline is a hand-crafted heuristic (not a learned model).  
It gives each buyer-eclass pair a score using three intuitive signals:

Score each $ (b, e) $ by:

$$
\text{score}_{\text{base}}(b, e) = \alpha \cdot m_{\text{active}} + \beta \cdot \sqrt{s_{\text{total}}} - \gamma \cdot \delta_{\text{recency}}
$$

Comments on each term:

- $ m_{\text{active}} $: number of active months in the train window (a frequency/regularity signal).  
  Higher means the buyer repeatedly buys this eclass, so we increase score.
- $ s_{\text{total}} $: total train-period spend for this pair (pipeline: `historical_purchase_value_total`).  
  We use $ \sqrt{s_{\text{total}}} $ (pipeline: `historical_purchase_value_sqrt`) so very large spend does not dominate linearly.
- $ \delta_{\text{recency}} $: time since the last purchase (a staleness signal).  
  Larger means "last purchase was long ago", so we subtract it.

How to read the weights (configured in `modelling.approaches.baseline`):

- $ \alpha $ controls how much recurring purchase activity matters.
- $ \beta $ controls how much monetary value matters.
- $ \gamma $ controls how strongly we penalize old/stale pairs.

Decision rule:

1. Tune $ \alpha, \beta, \gamma $ on the validation euro score.
2. Predict pair $ (b,e) $ if $ \text{score}_{\text{base}}(b,e) > \theta $.
3. Apply per-buyer cap: keep only top $ K $ pairs by score.

So baseline v1 is: **frequency + value - staleness**, then threshold + top-$K$.

### Two-stage model (v2)

**Stage A — Recurrence probability:**

Train a binary classifier (e.g. LightGBM) on label $ y = \text{label}(b,e) $:

$$
\hat{p}(b, e) = P(\text{recur} \mid \mathbf{x}_{b,e})
$$

**Stage B — Conditional value estimate:**

On positive training examples, regress on future spend:

$$
\hat{v}(b, e) = E[s_{\text{future}}(b,e) \mid \mathbf{x}_{b,e},\ \text{recur}=1]
$$

**Expected utility:**

$$
\widehat{EU}(b, e) = \hat{p}(b, e) \cdot \hat{v}(b, e) \cdot r - F
$$

### Pass-through (candidate-only)

The **pass_through** approach is an explicit “model” that performs no scoring: every candidate is assigned a constant positive score so that, when the selection policy is configured for pass-through (see below), no further filtering is applied. Use it to submit exactly the candidate set $ \mathcal{C}_b $ for each warm buyer — e.g. for diagnostics or as an upper-bound on recall.

- Enable the `pass_through` approach in `modelling.enabled_approaches` and use the pass-through selection configuration; then $ \hat{S}_b = \mathcal{C}_b $ for every $ b $.

---

## 7. Selection Policy

For each buyer $ b \in \mathcal{B}_{\text{warm}} $:

1. Compute $ \widehat{EU}(b, e) $ for all $ e \in \mathcal{C}_b $.
2. Keep only pairs above an EU threshold $ \tau_{EU} $:

$$
\hat{S}_b = \left\{ e \in \mathcal{C}_b \;\middle|\; \widehat{EU}(b,e) > \tau_{EU} \right\}
$$

3. Apply minimum evidence guardrail — require at least one of:
   - $ n_{\text{orders}} \geq X $ (`modelling.selection.guardrails.min_orders`, e.g. 3), or
   - $ m_{\text{active}} \geq Y $ (`modelling.selection.guardrails.min_months`, e.g. 2), or
   - `historical_purchase_value_total` $ \geq \tau_{\text{high}} $ (`modelling.selection.guardrails.high_spend`, high-value exception)

4. Cap at top $ K $ by $ \widehat{EU} $ (`modelling.selection.top_k_per_buyer`; tune on validation; start $ K = 15 $).

**Pass-through configuration** (to force all candidates into submission, e.g. when using the *pass_through* approach or for recall upper-bound):
- Set $ \tau_{EU} \leq 0 $ (e.g. `score_threshold: 0`; pass_through sets all scores $ > 0 $)
- Set $ X = 1 $, $ Y = 1 $, and $ \tau_{\text{high}} = 0 $
- Set $ K \geq \max_b |\mathcal{C}_b| $ (e.g. `top_k_per_buyer: 9999`)
- Then $ \hat{S}_b = \mathcal{C}_b $ for every warm buyer $ b $ (portfolio equals candidate set)

---

## 8. Tuning Priority

In order of expected impact on validation euro score:

| Priority | Parameter | Notes |
|---|---|---|
| 1 | EU threshold $ \tau_{EU} $ (or $ \theta $ for baseline) | Core precision-recall tradeoff; set very low for pass-through |
| 2 | Per-buyer cap $ K $ | Hard ceiling on fee exposure |
| 3 | Lookback window $ L $ | Candidate set size |
| 4 | Guardrail/candidate thresholds $ \eta, X, Y, \tau_{\text{high}} $ | Candidate strictness and edge case pruning |
| 5 | Value model calibration | Affects ranking among positive recurrence candidates |

---

## 9. Scoring Reference

| Parameter | Value |
|---|---|
| Savings function | Black box — scales non-linearly with spend (approx. $ \sqrt{\text{spend}} $ × frequency) |
| Fixed fee $ F $ | €10 per predicted element per month |
| Scoring window | ~1 month (hidden) |

> The exact savings formula and fee may be adjusted by organizers. Treat the scorer as a black box and do not hardcode a fixed savings rate. Solutions should remain robust under changes to both savings scaling and fee levels.

The break-even condition (illustrative, assuming linear savings with effective rate $ r $) is:

$$
\hat{p}(b,e) \cdot E[s_{\text{future,monthly}}] \cdot r > F
$$

In practice, use the validation euro score (per-month normalised) as the optimisation target rather than a fixed analytical threshold.
