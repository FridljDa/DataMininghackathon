Warm-Only Strategy
1) Frame it as buyer-level portfolio selection
For each warm legal_entity_id, rank eclass by expected net value and keep only the best ones.

Don’t predict all historically seen eclasses.
Do optimize benefit - fee with conservative thresholds.
2) Use a strict time split for offline training
Inside plis_training (already pre-cut for warm users), create your own backtest:

Pick an internal cutoff (example: train up to T, validate on T+1... months).
Build labels: buyer-eclass is positive if it appears in future window with minimum recurrence (e.g. appears in >=2 future months or >=N future orders).
This teaches recurrence, not one-off events.
3) Candidate generation (per buyer)
Keep only plausible eclasses to reduce noise:

eclasses seen in last 12-18 months for that buyer
plus top eclasses by spend share for that buyer
optionally drop ultra-rare one-time eclasses unless very high value
This step usually gives big score gains by reducing fee leakage.

4) Core features for each buyer-eclass
From plis_training columns (orderdate, quantityvalue, vk_per_item, etc.):

Frequency: #orders, #active months, months with purchases / observed months
Recency: months since last purchase
Regularity: std of inter-purchase gaps, coefficient of variation by month
Economic weight: total spend proxy sum(quantityvalue * vk_per_item), median line value, sqrt/log transformed spend
Portfolio share: eclass spend share within buyer
Trend: recent 3-month vs prior 6-month activity/spend ratio
Buyer context: size + NACE as weak priors
5) Two-stage modeling (recommended)
Stage A (probability): predict recurrence probability P(recur)
Stage B (value): estimate conditional value if recurring E[value | recur]
Combine to expected utility:
EU = P(recur) * E[value | recur] - fee_proxy
Then select eclasses with EU > threshold.

If you want simpler v1: one model predicting a value-weighted target can work too, but two-stage is easier to debug.

6) Selection policy (crucial)
For each buyer:

Sort by EU descending
Keep only positives
Add max cap K (e.g. 5-20) tuned on validation
Add minimum evidence guardrails:
at least X historical orders OR
at least Y active months OR
very high spend exception
This prevents long-tail overprediction.

7) What to tune first
In order of impact:

Future label definition (what counts as “recurring”)
Selection threshold on EU
Per-buyer cap K
Candidate lookback window (12 vs 18 vs 24 months)
Guardrail thresholds
8) Strong baseline before ML
Build this first and benchmark it:

Score each buyer-eclass by:
freq_score = active_months
value_score = sqrt(total_spend)
recency_penalty = months_since_last_purchase
baseline_score = a*freq + b*value - c*recency
pick top-K with minimum score
Then only keep ML if it clearly beats this in temporal backtest.

If you want, I can next give you a concrete warm v1 spec with exact feature list, label SQL/pandas logic, and threshold grid so you can implement directly.