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

| Entity type | Training data | Evaluation data |
|---|---|---|
| Warm (`cs = 1`) | `plis_training.csv` rows up to `2025-06-30` | PLIs after `2025-07-01` (hidden) |
| Cold (`cs = 0`) | No PLIs available | All PLIs (hidden) |

Focus here: **warm buyers only** ($ b \in \mathcal{B}_{\text{warm}} $).

Available columns used from `plis_training.csv`:

- `orderdate`, `legal_entity_id`, `eclass`, `manufacturer`
- `quantityvalue`, `vk_per_item` (spend proxy: $ s = \text{quantityvalue} \times \text{vk\_per\_item} $)
- `nace_code`, `estimated_number_employees` (buyer context)

---

## 3. Offline Validation Setup

To simulate the hidden evaluation, create an internal temporal split within the training window:

- **Train period:** `2023-01-01` — `2024-12-31`
- **Validation period:** `2025-01-01` — `2025-06-30`

Label a buyer-eclass pair $ (b, e) $ as **positive** in validation if it is bought again at least once:

$$
\text{label}(b, e) = \mathbf{1}\left[\text{orders in val}(b, e) \geq 1\right], \quad b \in \mathcal{B}_{\text{warm}},\ e \in \mathcal{E}
$$

Use binary recurrence target by default: bought again vs not bought again.

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
\mathcal{C}_b = \{ e \in \text{history}(b) \mid n_{\text{orders}}(b,e, L) \ge \eta \}
$$

Interpretation:
- $ \text{history}(b) \subseteq \mathcal{E} $ is the set of eclasses buyer $ b $ has ever purchased
- $ n_{\text{orders}}(b,e, L) $ is the number of orders of eclass $ e $ by buyer $ b $ within the lookback window $ L $
- $ T $ is the training cutoff timestamp
- $ L $ is the lookback window (default: 18 months)
- $ \eta $ is the minimum order frequency threshold (default: 1)
- So $ \mathcal{C}_b $ keeps only eclasses from $ b $'s history that were bought at least $ \eta $ times recently
- Therefore, by construction, $ \mathcal{C}_b \subseteq \text{history}(b) \subseteq \mathcal{E} $

---

## 5. Feature Engineering

For each candidate pair $ (b, e) $ with $ b \in \mathcal{B}_{\text{warm}} $ and $ e \in \mathcal{C}_b $ in the train period, compute:

### Frequency features
- $ n_{\text{orders}} $: total number of PLI rows
- $ m_{\text{active}} $: number of distinct calendar months with at least one purchase
- $ \rho_{\text{freq}} = m_{\text{active}} / m_{\text{observed}} $: purchase rate (months active / months buyer was observed)

### Recency features
- $ \delta_{\text{recency}} $: months since last purchase of $ e $ by $ b $

### Regularity features
- $ \sigma_{\text{gap}} $: standard deviation of inter-purchase gaps (in months)
- $ \text{CV}_{\text{gap}} = \sigma_{\text{gap}} / \mu_{\text{gap}} $: coefficient of variation (low = regular)

### Economic weight features
- $ s_{\text{total}} = \sum \text{quantityvalue} \times \text{vk\_per\_item} $: total spend
- $ \tilde{s} = \sqrt{s_{\text{total}}} $: square-root transformed spend (aligns with scoring hint)
- $ \bar{s}_{\text{line}} $: median spend per line item
- $ w_e^b = s_{\text{total}}(b,e) / s_{\text{total}}(b) $: eclass share of buyer's total spend

### Trend features
- $ \Delta_{\text{trend}} = m_{\text{active, last 3mo}} - m_{\text{active, prior 6mo}}/2 $: recent activity change

### Buyer context (static)
- $ \log(\text{employees} + 1) $
- NACE 2-digit code (categorical)

---

## 6. Modelling

### Baseline (v1)

Score each $ (b, e) $ by:

$$
\text{score}_{\text{base}}(b, e) = \alpha \cdot m_{\text{active}} + \beta \cdot \sqrt{s_{\text{total}}} - \gamma \cdot \delta_{\text{recency}}
$$

Tune $ \alpha, \beta, \gamma $ on the validation euro score. Select pairs where $ \text{score}_{\text{base}} > \theta $ and cap at top $ K $ per buyer.

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

---

## 7. Selection Policy

For each buyer $ b \in \mathcal{B}_{\text{warm}} $:

1. Compute $ \widehat{EU}(b, e) $ for all $ e \in \mathcal{C}_b $.
2. Keep only pairs with positive expected utility:

$$
\hat{S}_b = \left\{ e \in \mathcal{C}_b \;\middle|\; \widehat{EU}(b,e) > 0 \right\}
$$

3. Apply minimum evidence guardrail — require at least one of:
   - $ n_{\text{orders}} \geq X $ (e.g. 3), or
   - $ m_{\text{active}} \geq Y $ (e.g. 2), or
   - $ s_{\text{total}} \geq \tau_{\text{high}} $ (high-value exception)

4. Cap at top $ K $ by $ \widehat{EU} $ (tune $ K $ on validation; start $ K = 15 $).

---

## 8. Tuning Priority

In order of expected impact on validation euro score:

| Priority | Parameter | Notes |
|---|---|---|
| 1 | EU threshold (or $ \theta $ for baseline) | Core precision-recall tradeoff |
| 2 | Per-buyer cap $ K $ | Hard ceiling on fee exposure |
| 3 | Lookback window $ L $ | Candidate set size |
| 4 | Guardrail thresholds $ X, Y, \tau_s $ | Edge case pruning |
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
