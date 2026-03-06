What AutoML actually buys you here
Your features in build_features.py are: m_active, rho_freq, delta_recency, sigma_gap, CV_gap, s_total_sqrt, w_e_b, delta_trend, log_employees, nace_2. That's ~10 mostly numeric, well-motivated features.

The signal structure is extremely strong and obvious: a buyer ordering something 12 times over 2 years with low CV_gap is core demand. A buyer with 1 order 17 months ago is not. LightGBM vs the linear baseline will likely find the same rank order — the features already encode the problem well.

Where the real leverage is, concretely
Looking at your actual pipeline, the bigger wins are elsewhere:

1. The scoring mismatch is your most critical bug right now. Your write_submission_warm.py produces cluster = eclass (Level 1), but score_submission.py evaluates against eclass|manufacturer (Level 2). That's why you have 0 hits and -€1940 locally. This is not a model problem at all.

2. Cold-start buyers are currently handled very naively. In write_submission_warm.py, every cold buyer gets the single most-frequent global eclass. That's a lot of fee-paying predictions that almost certainly miss. Cold buyers need industry-level (NACE-based) collaborative filtering or co-occurrence priors — this is a bigger opportunity than improving warm-buyer ranking.

3. The hyper-parameters in config.yaml are all at defaults and untouched. alpha=1.0, beta=1.0, gamma=0.5 and score_threshold=0.0 are untuned. Even a simple grid search over those 3 weights using your existing offline euro score proxy would likely improve ranking more than swapping in LightGBM.

4. The top_k_per_buyer=15 cap is very generous. The fee is €10/element. With the current savings assumption (10% of spend), a candidate needs €100/month in future spend just to break even. Many of those 15 slots per buyer are probably fee-negative. Tightening this (or making it buyer-specific based on history depth) is free precision.

So: is AutoML worth it?
No, not yet. The ranking problem is not the bottleneck. In roughly descending order of expected impact:

Fix	Effort	Expected impact
Fix Level 1 vs Level 2 scoring mismatch	Small	Unlocks meaningful offline score
Tune alpha/beta/gamma on euro score	Small	Better warm-buyer ranking
Tighten top-K or make it dynamic per buyer	Small	Precision up, fees down
Improve cold-start (NACE co-occurrence)	Medium	Large — many buyers get nothing useful now
Replace linear score with LightGBM	Medium	Marginal — features already capture the signal
Full AutoML (FLAML/Optuna + CV pipeline)	Large	Marginal net gain over tuned LightGBM
AutoML would make sense after those first four are solid. At that point you'd have a well-tuned linear baseline to beat, a working offline metric, and enough signal to tell whether a 50-feature LightGBM actually improves over tuned α/β/γ. Right now you can't even tell — the score is 0 due to the format bug.