# Modelling Approach — Core Demand Prediction (Level 1)

## 0. Global Sets & Notation

Given from data/problem setup:

- $ \mathcal{B} $: all buyers
- $ \mathcal{B}_{\text{scored}} \subseteq \mathcal{B} $: buyers included by the evaluator
- $ \mathcal{E} $: all E-Class identifiers (eclasses)
- $ \text{history}(b) \subseteq \mathcal{E} $: eclasses buyer $ b $ purchased in observed history

Optional segmentation (if used in a specific run):
- $ \mathcal{B}_{\text{warm}} \subseteq \mathcal{B} $: warm buyers (`cs = 1`)
- $ \mathcal{B}_{\text{cold}} = \mathcal{B} \setminus \mathcal{B}_{\text{warm}} $: cold buyers (`cs = 0`)

Defined by our modelling choices:

- $ \mathcal{C}_b \subseteq \mathcal{E} $: candidate set we construct for buyer $ b $ (Section 4)
- $ \hat{S}_b \subseteq \mathcal{C}_b $: final predicted set selected by the policy (Section 7)

Core symbols:
- $ T $: training cutoff timestamp
- $ L $: lookback length (months)
- $ H $: forward/scoring horizon length (months)
- $ F $: fixed fee per predicted element (monthly)
- $ b $ indexes buyers, $ e $ indexes eclasses

Time-window notation (half-open intervals):

- $ I_{\text{lookback}}(T,L) := [T-L,\;T) $
- $ I_{\text{future}}(T,H) := [T,\;T+H) $

---

## 1. Problem Framing

For each scored buyer $ b \in \mathcal{B}_{\text{scored}} $ we must select a portfolio $ \hat{S}_b \subseteq \mathcal{E} $ of E-Class identifiers that maximises **net economic benefit**:

$$
\text{Score} = \sum_{b \in \mathcal{B}_{\text{scored}}} \left[ \sum_{e \in \hat{S}_b} \text{Savings}(b, e) - |\hat{S}_b| \cdot F \right]
$$

where $ F = €10 $ is the fixed monthly fee per predicted element and savings are realised only when prediction $ e $ matches actual future purchases.

Savings per hit are a function of future spend (exact formula is a black box — savings scale non-linearly with spend, roughly with its square root, combined with demand frequency). The scoring function may change; do not hardcode a fixed savings rate.

Selection is precision-first: every element that is not a true recurring need pays a fee with zero return.

---

## 2. Data

Raw inputs:
- `data/02_raw/plis_training.csv` (all historical PLIs)
- `data/02_raw/customer_test.csv` (buyers to predict for online submission)
- `data/02_raw/features_per_sku.csv` (product attributes: SKU → key/value; used in `engineer_features_raw` to add top-K attribute columns to `data/08_features_raw`).

Focus for this modelling document: warm-buyer recurrence logic for Level 1 (`cluster = eclass`) trained on observed history and used for online submission.

Available columns used from historical PLIs:

- `orderdate`, `legal_entity_id`, `eclass`, `manufacturer`
- `quantityvalue`, `vk_per_item` (spend proxy: $ s = \text{quantityvalue} \times \text{vk\_per\_item} $)
- `nace_code`, `estimated_number_employees` (buyer context)

---

## 3. Target Definition

Modeling uses a temporal cutoff $ T $ and two explicit time intervals:

- features are computed from pre-cutoff data, primarily from $ I_{\text{lookback}}(T,L) $
- targets are computed from $ I_{\text{future}}(T,H) $

For each candidate pair $ (b,e) $, define:

$$
y_{b,e}
=
\mathbf{1}\!\left[n_{\text{orders}}\!\left(b,e, I_{\text{future}}(T,H)\right)\ge n_{\min}\right]
$$

with default $ n_{\min}=1 $.

Future spend target:

$$
s_{\text{future}}(b,e)
=
\sum_{\substack{\text{rows for }(b,e)\\ \text{with } \text{orderdate}\in I_{\text{future}}(T,H)}}
\text{quantityvalue}\times \text{vk\_per\_item}
$$

Future order count target (used by the v3 Stage A count model):

$$
n_{\text{future}}(b,e)
:=
n_{\text{orders}}\!\left(b,e, I_{\text{future}}(T,H)\right)
$$

Default learning target is binary recurrence ($ y_{b,e} $), with spend used for value-aware ranking. The count target $ n_{\text{future}} $ is used only by the v3 factorized model's Stage A.

> **Evaluator behavior note:** Predictions are set-based per buyer and cluster. Duplicate rows for the same $ (b,e) $ are sanitized and counted once by the organizer scorer.

---

## 4. Candidate Generation

For each scored buyer, restrict the candidate set to reduce fee leakage.

We construct $ \mathcal{C}_b $ as:

$$
\mathcal{C}_b
=
\left\{
e \in \text{history}(b)\ \middle|\ 
n_{\text{orders}}\!\left(b,e, I_{\text{lookback}}(T,L)\right)\ge \eta
\ \land\
s_{\text{lookback}}\!\left(b,e, I_{\text{lookback}}(T,L)\right)\ge \tau
\right\}
$$

Interpretation:
- $ \text{history}(b) \subseteq \mathcal{E} $ is the set of eclasses buyer $ b $ has ever purchased
- $ n_{\text{orders}}(b,e, I_{\text{lookback}}(T,L)) $ is the number of orders of eclass $ e $ by buyer $ b $ in interval $ [T-L, T) $
- $ s_{\text{lookback}}(b,e, I_{\text{lookback}}(T,L)) = \sum_{\substack{\text{rows for }(b,e)\\ \text{with } \text{orderdate}\in [T-L,T)}} \text{quantityvalue} \times \text{vk\_per\_item} $: total spend on eclass $ e $ by buyer $ b $ in the lookback interval (EUR)
- $ T $ is the training cutoff timestamp
- $ L $ is the lookback length (default: 18 months)
- $ \eta $ is the minimum order frequency threshold (default: 1)
- $ \tau $ is the minimum lookback spend threshold in EUR (default: 100)
- So $ \mathcal{C}_b $ keeps only eclasses from $ b $'s history that were bought at least $ \eta $ times and with at least $ \tau $ EUR spend in $ [T-L, T) $
- Therefore, by construction, $ \mathcal{C}_b \subseteq \text{history}(b) \subseteq \mathcal{E} $

---

## 5. Feature Engineering

**Pipeline data contract:** Candidate outputs (`data/07_candidates/{mode}/level{level}/candidates_raw.parquet`) contain only key columns: Level 1 = `legal_entity_id`, `eclass`; Level 2 = `legal_entity_id`, `eclass`, `manufacturer`. Raw features (`data/08_features_raw/.../features_raw.parquet`) are assembled from `data/02_raw` (and split PLIs): keys plus pair-level aggregates (e.g. `n_orders`, `historical_purchase_value_total`, orderdate min/max, `orderdates_str`), buyer context (`estimated_number_employees`, `nace_code`, `secondary_nace_code`), and top-K SKU attribute columns. Derived feature engineering then adds computed features from this raw matrix.

For each candidate pair $ (b, e) $ with $ b \in \mathcal{B}_{\text{scored}} $ and $ e \in \mathcal{C}_b $, derive a flexible feature set from pre-cutoff history (primarily $ I_{\text{lookback}}(T,L) $):

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

- `feature_summary.csv` — Per-feature descriptive stats (null/zero rate, quantiles, cardinality) and target-aware stats (univariate signal vs recurrence label and vs positive-case spend). Primary inspection artifact for deciding which features to keep.
- `feature_redundancy.csv` — Pairs of numeric features with high Spearman correlation (e.g. |ρ| ≥ 0.85) for redundancy pruning.
- `feature_suggestions.yaml` — Advisory list of suggested features: hard filters (null rate, variance, cardinality) are applied, then one representative per correlated group is kept (ranked by target signal). For manual copy into `config.yaml` under `modelling.features.selected`.

**Refreshing the feature list:** `feature_suggestions.yaml` answers “which features look best for modelling?”; `modelling.features.selected` and the resulting `data/11_features_selected` output are the explicit contract used by training and scoring. Copy from the suggestion file into config when you want to align the pipeline with the heuristic; the pipeline does not modify config automatically.

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

1. Tune $ \alpha, \beta, \gamma $ on your chosen euro-score proxy.
2. Predict pair $ (b,e) $ if $ \text{score}_{\text{base}}(b,e) > \theta $.
3. Apply per-buyer cap: keep only top $ K $ pairs by score.

So baseline v1 is: **frequency + value - staleness**, then threshold + top-$K$.

### Two-stage model (v2)

Let one training example be a buyer-eclass pair $ (b,e) $ with features $ \mathbf{x}_{b,e} $.

Define two targets:

- Binary recurrence target:
  $$
  y_{b,e} \in \{0,1\}
  $$
- Future spend target in $ I_{\text{future}}(T,H) $:
  $$
  z_{b,e} := s_{\text{future}}(b,e) \ge 0
  $$

where:
- $ y_{b,e}=1 $ means the pair recurs in interval $ I_{\text{future}}(T,H) $.
- $ z_{b,e} $ is the euro spend for that pair in the same interval.

**Stage A — recurrence model (classification):**

Fit a classifier on all candidate pairs:

$$
\hat{p}_{b,e} := P(y_{b,e}=1 \mid \mathbf{x}_{b,e})
$$

Interpretation: estimated probability that pair $ (b,e) $ will recur.

**Stage B — value model (regression on positives):**

Fit a regressor only on pairs with $ y_{b,e}=1 $:

$$
\hat{v}_{b,e} := E[z_{b,e} \mid \mathbf{x}_{b,e},\ y_{b,e}=1]
$$

Interpretation: expected spend *if recurrence happens*.

**Combine both models at inference time:**

Using the law of total expectation:

$$
E[z_{b,e}\mid \mathbf{x}_{b,e}] \approx \hat{p}_{b,e}\cdot \hat{v}_{b,e}
$$

Expected utility (illustrative linear savings with rate $ r $ and fee $ F $):

$$
\widehat{EU}(b,e)=\hat{p}_{b,e}\cdot \hat{v}_{b,e}\cdot r - F
$$

So:
- Stage A answers: "How likely is recurrence?"
- Stage B answers: "If it recurs, how much value?"
- Their product gives expected spend before fee.

### Three-stage factorized model (v3)

Another reasonable decomposition is to model the score drivers more explicitly instead of jumping directly from recurrence to future spend.

Let:

$$
\hat{n}_{b,e}
:=
E\!\left[n_{\text{orders}}(b,e, I_{\text{future}}(T,H)) \mid \mathbf{x}_{b,e}\right]
$$

be the predicted future order count for buyer-eclass pair $ (b,e) $, and let:

$$
\bar{v}_{b,e}
:=
\frac{
\sum_{\substack{\text{rows for }(b,e)\\ \text{with } \text{orderdate}\in I_{\text{lookback}}(T,L)}}
\text{quantityvalue}\times \text{vk\_per\_item}
}{
\max\!\left(1,\;\sum_{\substack{\text{rows for }(b,e)\\ \text{with } \text{orderdate}\in I_{\text{lookback}}(T,L)}}
\text{quantityvalue}\right)
}
$$

be the **historical quantity-weighted average price** for that pair computed entirely from lookback data. This is a well-defined, leakage-free quantity available at inference time. It equals `avg_spend_per_order / avg_order_quantity` and is closely related to the existing `avg_spend_per_order` feature already computed by the pipeline.

**VK price stability context:** Analysis in `src/check_vk_price_stability.py` shows that early-vs-late median drift ≥ 20% affects only ~1–3% of spend at the buyer-eclass level, confirming that $ \bar{v}_{b,e} $ is a reasonable forward proxy for the vast majority of pairs. We accept residual price drift as a known limitation of the factorization.

Conceptually:

1. **Stage A — demand frequency:** estimate $ \hat{n}_{b,e} $, the expected number of future orders. Training target: $ n_{\text{orders}}(b,e, I_{\text{future}}(T,H)) $ as a count regression (requires adding this to Section 3).
2. **Stage B — unit value proxy:** use $ \bar{v}_{b,e} $ (historical quantity-weighted average price from lookback) directly, without fitting a separate model.
3. **Stage C — score construction:** combine the two into a value proxy aligned with the evaluator, then subtract the fixed prediction fee.

One simple proxy is:

$$
\hat{z}_{b,e}^{\text{factorized}}
:=
\hat{n}_{b,e}\cdot \bar{v}_{b,e}
$$

and then:

$$
\widehat{EU}(b,e)
:=
g(\hat{z}_{b,e}^{\text{factorized}}) - F
$$

where $ g(\cdot) $ is a scorer-aligned transformation or calibrated proxy (e.g. $ \sqrt{\cdot} $ to match the approximate savings scaling noted in Section 9).

Why this can make sense:

- It separates **how often** the buyer is expected to order from **how valuable** each order is.
- Stage B requires no learned model: $ \bar{v}_{b,e} $ is a direct lookup from lookback aggregates, already computed in the feature pipeline (`avg_spend_per_order`). The inference path is unambiguous and leakage-free.
- It may be easier to calibrate than a full spend regressor because per-item prices are empirically stable for the majority of buyer-eclass pairs.

Important caveats:

- The factorization $ \hat{n}_{b,e} \cdot \bar{v}_{b,e} $ is an approximation to $ s_{\text{future}}(b,e) $ because it ignores correlation between order frequency and order size, and assumes quantity per order is constant.
- Stage A requires a count regression target ($ n_{\text{orders}} $ in $ I_{\text{future}} $) which is not yet defined in Section 3 — add it there before implementing.
- $ \bar{v}_{b,e} $ will be inaccurate for the small share of pairs with material price drift (~1–3% of spend); we accept this as a known limitation.
- Because the evaluator is fee-sensitive and set-based, the final decision rule should still be thresholded and top-$ K $ capped as in Section 7.

### Pass-through (candidate-only)

The **pass_through** approach is an explicit “model” that performs no scoring: every candidate is assigned a constant positive score so that, when the selection policy is configured for pass-through (see below), no further filtering is applied. Use it to submit exactly the candidate set $ \mathcal{C}_b $ for each scored buyer — e.g. for diagnostics or as an upper-bound on recall.

- Enable the `pass_through` approach in `modelling.enabled_approaches` and use the pass-through selection configuration; then $ \hat{S}_b = \mathcal{C}_b $ for every $ b $.

---

## 7. Selection Policy

For each buyer $ b \in \mathcal{B}_{\text{scored}} $:

1. Compute $ \widehat{EU}(b, e) $ for all $ e \in \mathcal{C}_b $.
2. Keep only pairs above an EU threshold $ \tau_{EU} $:

$$
\hat{S}_b = \left\{ e \in \mathcal{C}_b \;\middle|\; \widehat{EU}(b,e) > \tau_{EU} \right\}
$$

3. Apply minimum evidence guardrail — require at least one of:
   - $ n_{\text{orders}} \geq X $ (`modelling.selection.guardrails.min_orders`, e.g. 3), or
   - $ m_{\text{active}} \geq Y $ (`modelling.selection.guardrails.min_months`, e.g. 2), or
   - `historical_purchase_value_total` $ \geq \tau_{\text{high}} $ (`modelling.selection.guardrails.high_spend`, high-value exception)

4. Cap at top $ K $ by $ \widehat{EU} $ (`modelling.selection.top_k_per_buyer`; tune with score sensitivity runs; start $ K = 15 $).

**Pass-through configuration** (to force all candidates into submission, e.g. when using the *pass_through* approach or for recall upper-bound):
- Set $ \tau_{EU} \leq 0 $ (e.g. `score_threshold: 0`; pass_through sets all scores $ > 0 $)
- Set $ X = 1 $, $ Y = 1 $, and $ \tau_{\text{high}} = 0 $
- Set $ K \geq \max_b |\mathcal{C}_b| $ (e.g. `top_k_per_buyer: 9999`)
- Then $ \hat{S}_b = \mathcal{C}_b $ for every scored buyer $ b $ (portfolio equals candidate set)

---

## 8. Tuning Priority

In order of expected impact on euro score:

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

In practice, optimize against your chosen euro-score proxy and online results rather than a fixed analytical threshold.
