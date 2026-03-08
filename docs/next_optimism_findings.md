# Next optimism: findings from updated results

Summary of leaderboard and tuning artifacts used to isolate the next experiments.

## Leaderboard (from challenge2-core-demand-prediction.md)

- **Level 1**: AAD 5th (€1,327,614.87, 64.7% spend captured). Leaders submit more (76–79% capture). Current best run is phase3_repro-style (780611.94).
- **Level 2**: AAD 3rd (€423,553.47, 25.5% capture). Leaders 1–2 have higher fees and capture (32–33%). We are still under-submitting.

## Submission shape (current_submission_shape_level*.csv)

- `lgbm_two_stage`: ~66–69 avg predictions per buyer (level1 ~69, level2 ~66).
- `phase3_repro` / `pass_through`: ~214–218 per buyer.
- Bottleneck is selection breadth (score threshold + cap), not candidate set size.

## Tuning artifacts

- **param_effects_level2.csv**: Every archived run has `config_score_threshold=0.0`. No negative-threshold experiments yet.
- **run_metrics_level2.csv**: Best archived level2 is lgbm_two_stage with top_k=400, loose guardrails. Next lever is allowing more scored rows through (negative threshold).
- **level1**: Best run 780611.94 already from broader setup; only small confirmation sweep needed (top_k 150 vs 200, threshold 0 vs -0.01 vs -0.03).

## Isolated next experiments

1. **Level 2**: Negative-threshold sweep first: -0.01, -0.03, -0.05, -0.10, -0.20 with top_k=400, guardrails 0/1/0, cold_start_top_k=50.
2. **Level 1**: Small confirmation: top_k 150 vs 200, threshold 0.0 vs -0.01 vs -0.03, guardrails 2/2/200.
3. **Hybrid**: Only if level2 stays too sparse after thresholding; use existing hybrid_lgbm_phase3 rule.
