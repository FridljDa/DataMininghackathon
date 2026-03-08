"""
Snakemake workflow for Core Demand Challenge.

All input/output paths are defined here. Python scripts receive paths only via
CLI arguments. Deliverables: one portfolio and submission per enabled approach
under data/13_portfolio/{mode}/{approach}/ and data/14_submission/{mode}/{approach}/.
"""

configfile: "config.yaml"

# --- Directory layout (single source of truth in Snakefile) ---
DATA_DIR = "data"
OUTPUT_DIR = "outputs"
DAG_SVG = f"{DATA_DIR}/01_dag/dag.svg"

# Raw input paths (config may override for alternate locations)
INPUTS = config["inputs"]
CLEAN_DIR = f"{DATA_DIR}/02_clean"
PLIS_TRAINING_RAW_CSV = config["plis_training_csv"]
PLIS_TRAINING_CSV = f"{CLEAN_DIR}/plis_training.csv"
CUSTOMER_META_CSV = config["customer_meta_csv"]
# Training/validation split (params from config; paths derived here)
SPLIT = config["training_validation_split"]
SPLIT_CUSTOMER_CSV = f"{DATA_DIR}/04_customer/customer.csv"
PLIS_TRAINING_SPLIT = f"{DATA_DIR}/05_training_validation/plis_training.csv"
PLIS_TESTING_SPLIT = f"{DATA_DIR}/05_training_validation/plis_testing.csv"

# Score outputs and archive (per mode and approach)
SCORES_DIR = f"{DATA_DIR}/15_scores"
SCORES_BEST_DIR = f"{DATA_DIR}/16_scores_best"
SCORE_RUN_ARCHIVE = {
    "online_runs_dir": f"{SCORES_DIR}/online/runs",
}
SCORING_PARAMS = config["scoring_parameters"]
BEST_RUN_COPIED_SENTINEL_PATTERN = f"{SCORES_BEST_DIR}/online/level{{level}}/best_run/.copied"

# Submission tuning diagnostics (plots and summary CSVs per level)
SUBMISSION_TUNING_DIR = f"{DATA_DIR}/17_submission_tuning"

# Exploration plots
PLOTS_DIR = f"{DATA_DIR}/06_plots"
PLOT_FILES = [
    "seasonal_purchase_volume.png",
    "purchase_value_by_task_violin.png",
    "task_distribution.png",
    "cs_task_consistency_heatmap.png",
    "nace_by_task_topn.png",
]
PLOT_OUTPUTS = expand(f"{PLOTS_DIR}/{{f}}", f=PLOT_FILES)

# VK price stability check (script writes to 06_plots)
VK_STABILITY_PLOT_LOOKUP = f"{PLOTS_DIR}/vk_lookup_like_share_by_key.png"
VK_STABILITY_PLOT_DRIFT = f"{PLOTS_DIR}/vk_material_drift_by_key.png"
VK_STABILITY_SUMMARY_CSV = f"{PLOTS_DIR}/vk_price_stability_summary.csv"

# Modelling (params only from config; paths are patterns below)
MODELLING = config["modelling"]
WIN = MODELLING["windows"]
VAL = WIN["validation"]
CAND = MODELLING["candidates"]
SEL = MODELLING["selection"]
GRD = SEL["guardrails"]


def _selection_for_level(level):
    """Resolve selection params for a level: merge defaults with by_level overrides."""
    by_level = SEL.get("by_level") or {}
    overrides = by_level.get(str(level)) or {}
    grd_default = SEL.get("guardrails") or {}
    grd = overrides.get("guardrails") or grd_default
    return {
        "score_threshold": overrides.get("score_threshold", SEL.get("score_threshold", 0.0)),
        "top_k_per_buyer": overrides.get("top_k_per_buyer", SEL.get("top_k_per_buyer", 400)),
        "min_orders_guardrail": grd.get("min_orders", 0),
        "min_months_guardrail": grd.get("min_months", 1),
        "high_spend_guardrail": grd.get("high_spend", 0),
        "min_avg_monthly_spend": grd.get("min_avg_monthly_spend", 0),
    }


def _cold_start_top_k_for_level(level):
    """Resolve cold_start_top_k for a level: use submission.by_level if present."""
    sub = config.get("submission") or {}
    by_level = sub.get("by_level") or {}
    overrides = by_level.get(str(level)) or {}
    return overrides.get("cold_start_top_k", sub.get("cold_start_top_k", 50))
FEAT = MODELLING["features"]
APP = MODELLING["approaches"]

# Feature analysis (per mode and level)
FEATURE_ANALYSIS_DIR = f"{DATA_DIR}/10_feature_analysis"
FEATURE_ANALYSIS_SUMMARY_PATTERN = f"{FEATURE_ANALYSIS_DIR}/{{mode}}/level{{level}}/feature_summary.csv"
FEATURE_ANALYSIS_PLOTS_PATTERN = [
    f"{FEATURE_ANALYSIS_DIR}/{{mode}}/level{{level}}/feature_distributions.png",
    f"{FEATURE_ANALYSIS_DIR}/{{mode}}/level{{level}}/feature_correlations.png",
]
FEATURE_ANALYSIS_SUGGESTIONS_PATTERN = f"{FEATURE_ANALYSIS_DIR}/{{mode}}/level{{level}}/feature_suggestions.yaml"

# Mode (online|offline) configuration: paths and behaviour for modelling/scoring pipeline
MODES = ["online", "offline"]
MODE_RE = "|".join(MODES)
MODE_CFG = {
    "online": {
        "customer_csv": CUSTOMER_META_CSV,
        "buyer_source": "customer-test",
        "customer_input": INPUTS["customer_test"],
        "runs_dir": SCORE_RUN_ARCHIVE["online_runs_dir"],
    },
    "offline": {
        "customer_csv": SPLIT_CUSTOMER_CSV,
        "buyer_source": "customer-split",
        "customer_input": SPLIT_CUSTOMER_CSV,
    },
}
# Path patterns for mode- and level-wildcarded rules (stages 07–11 and 12–13 per level)
CANDIDATES_DIR = f"{DATA_DIR}/07_candidates"
TRENDING_CLASSES_CSV = f"{CANDIDATES_DIR}/trending_classes.csv"
CANDIDATES_RAW_PATTERN = f"{CANDIDATES_DIR}/{{mode}}/level{{level}}/candidates_raw.parquet"
FEATURES_RAW_PATTERN = f"{DATA_DIR}/08_features_raw/{{mode}}/level{{level}}/features_raw.parquet"
FEATURES_SANITIZED_PATTERN = f"{DATA_DIR}/08_features_sanitized/{{mode}}/level{{level}}/features_sanitized.parquet"
FEATURES_ALL_PATTERN = f"{DATA_DIR}/09_features_derived/{{mode}}/level{{level}}/features_all.parquet"
FEATURES_SELECTED_PATTERN = f"{DATA_DIR}/11_features_selected/{{mode}}/level{{level}}/features_selected.parquet"
# Per-approach and per-level: scores, portfolio, submission (data/12–14 by {mode}/{approach}/level{level}/)
ENABLED_APPROACHES = MODELLING["enabled_approaches"]
ENABLED_LEVELS = MODELLING["enabled_levels"]
# Hybrid approach has no scores; portfolio is merged from lgbm_two_stage + phase3_repro
APPROACHES_WITH_SCORES = [a for a in ENABLED_APPROACHES if a != "hybrid_lgbm_phase3"]
APPROACH_RE = "|".join(ENABLED_APPROACHES)
APPROACH_RE_WITH_SCORES = "|".join(APPROACHES_WITH_SCORES)
LEVEL_RE = "|".join(str(l) for l in ENABLED_LEVELS)
SUBMISSION_TUNING_OUTPUTS = (
    expand(f"{SUBMISSION_TUNING_DIR}/run_metrics_level{{level}}.csv", level=ENABLED_LEVELS)
    + expand(f"{SUBMISSION_TUNING_DIR}/run_records_level{{level}}.jsonl", level=ENABLED_LEVELS)
    + expand(f"{SUBMISSION_TUNING_DIR}/param_effects_level{{level}}.csv", level=ENABLED_LEVELS)
    + expand(f"{SUBMISSION_TUNING_DIR}/current_submission_shape_level{{level}}.csv", level=ENABLED_LEVELS)
    + expand(f"{SUBMISSION_TUNING_DIR}/score_vs_predictions_level{{level}}.png", level=ENABLED_LEVELS)
    + expand(f"{SUBMISSION_TUNING_DIR}/score_vs_capture_level{{level}}.png", level=ENABLED_LEVELS)
    + expand(f"{SUBMISSION_TUNING_DIR}/run_timeline_level{{level}}.png", level=ENABLED_LEVELS)
    + expand(f"{SUBMISSION_TUNING_DIR}/submission_size_vs_score_level{{level}}.png", level=ENABLED_LEVELS)
)
SCORES_APPROACH_PATTERN = f"{DATA_DIR}/12_predictions/{{mode}}/{{approach}}/level{{level}}/scores.parquet"
PORTFOLIO_PATTERN = f"{DATA_DIR}/13_portfolio/{{mode}}/{{approach}}/level{{level}}/portfolio.parquet"
SUBMISSION_PATTERN = f"{DATA_DIR}/14_submission/{{mode}}/{{approach}}/level{{level}}/submission.csv"
SUBMISSION_WARM_PART_PATTERN = f"{DATA_DIR}/14_submission/{{mode}}/{{approach}}/level{{level}}/submission_warm.csv"
SUBMISSION_COLD_PART_PATTERN = f"{DATA_DIR}/14_submission/{{mode}}/{{approach}}/level{{level}}/submission_cold.csv"
ARCHIVE_SENTINEL_PATTERN = f"{SCORES_DIR}/online/runs/level{{level}}/.last_archived_{{approach}}"
SUBMITTED_SENTINEL_PATTERN = f"{DATA_DIR}/14_submission/.submitted_challenge2_{{approach}}_level{{level}}"
LIVE_SUMMARY_TEMP_PATTERN = f"{DATA_DIR}/14_submission/.score_summary_live_{{approach}}_level{{level}}.csv"

# Run-scoped patterns for sweep: predictions, portfolio, submission, and archive per run_id
RUN_ID_RE = "[^/]+"
SCORES_APPROACH_PATTERN_RUN = f"{DATA_DIR}/12_predictions/{{mode}}/{{approach}}/level{{level}}/{{run_id}}/scores.parquet"
PORTFOLIO_PATTERN_RUN = f"{DATA_DIR}/13_portfolio/{{mode}}/{{approach}}/level{{level}}/{{run_id}}/portfolio.parquet"
SUBMISSION_PATTERN_RUN = f"{DATA_DIR}/14_submission/{{mode}}/{{approach}}/level{{level}}/{{run_id}}/submission.csv"
SUBMISSION_WARM_PART_PATTERN_RUN = f"{DATA_DIR}/14_submission/{{mode}}/{{approach}}/level{{level}}/{{run_id}}/submission_warm.csv"
SUBMISSION_COLD_PART_PATTERN_RUN = f"{DATA_DIR}/14_submission/{{mode}}/{{approach}}/level{{level}}/{{run_id}}/submission_cold.csv"
LIVE_SUMMARY_TEMP_PATTERN_RUN = f"{DATA_DIR}/14_submission/.score_summary_live_{{approach}}_level{{level}}_{{run_id}}.csv"
SUBMITTED_SENTINEL_PATTERN_RUN = f"{DATA_DIR}/14_submission/.submitted_challenge2_{{approach}}_level{{level}}_{{run_id}}"
ARCHIVE_SENTINEL_RUN_PATTERN = f"{SCORES_DIR}/online/runs/level{{level}}/{{approach}}/{{run_id}}/.archived"

rule all:
    input:
        DAG_SVG,
        PLIS_TRAINING_CSV,
        CUSTOMER_META_CSV,
        PLOT_OUTPUTS,
        VK_STABILITY_PLOT_LOOKUP,
        VK_STABILITY_PLOT_DRIFT,
        VK_STABILITY_SUMMARY_CSV,
        expand(FEATURE_ANALYSIS_SUMMARY_PATTERN, mode=MODES, level=ENABLED_LEVELS),
        expand(FEATURE_ANALYSIS_PLOTS_PATTERN, mode=MODES, level=ENABLED_LEVELS),
        expand(FEATURE_ANALYSIS_SUGGESTIONS_PATTERN, mode=MODES, level=ENABLED_LEVELS),
        expand(SCORES_APPROACH_PATTERN, mode=MODES, approach=ENABLED_APPROACHES, level=ENABLED_LEVELS),
        expand(PORTFOLIO_PATTERN, mode=MODES, approach=ENABLED_APPROACHES, level=ENABLED_LEVELS),
        expand(SUBMISSION_PATTERN, mode=MODES, approach=ENABLED_APPROACHES, level=ENABLED_LEVELS),
        expand(ARCHIVE_SENTINEL_PATTERN, approach=ENABLED_APPROACHES, level=ENABLED_LEVELS),
        expand(SUBMITTED_SENTINEL_PATTERN, approach=ENABLED_APPROACHES, level=ENABLED_LEVELS),
        expand(BEST_RUN_COPIED_SENTINEL_PATTERN, level=ENABLED_LEVELS),
        SUBMISSION_TUNING_OUTPUTS,

rule generate_dag_graph:
    """Write high-level workflow graph as SVG (rule-level, not job-level)."""
    output:
        dag = DAG_SVG,
    shell:
        "mkdir -p $(dirname {output.dag}) && uv run snakemake --rulegraph | dot -Tsvg -o {output.dag}"

rule clean_plis_training:
    """Filter raw plis_training rows/columns from config and drop duplicate cleaned rows."""
    input:
        plis = PLIS_TRAINING_RAW_CSV,
        config_file = "config.yaml",
    output:
        plis = PLIS_TRAINING_CSV,
    params:
        config_key = "cleaning.plis_training",
    shell:
        "uv run src/clean_plis.py --input {input.plis} --output {output.plis} "
        "--config {input.config_file} --config-key {params.config_key}"

rule build_customer_meta:
    """Build customer metadata from plis_training (all unique customers, task from customer_test or none)."""
    input:
        plis = PLIS_TRAINING_CSV,
        customer_test = INPUTS["customer_test"],
    output:
        customer = CUSTOMER_META_CSV,
    shell:
        "uv run src/build_customer_meta.py --plis {input.plis} --customer-test {input.customer_test} --output {output.customer}"

rule generate_exploration_plots:
    """Generate EDA plots from raw CSVs and customer metadata into data/06_plots."""
    input:
        plis = PLIS_TRAINING_CSV,
        customer_test = INPUTS["customer_test"],
        les_cs = INPUTS["les_cs"],
        customer_meta = SPLIT_CUSTOMER_CSV,
    output:
        plots = PLOT_OUTPUTS,
    params:
        out_dir = PLOTS_DIR,
    shell:
        "uv run src/generate_exploration_plots.py --plis {input.plis} "
        "--customer-test {input.customer_test} --les-cs {input.les_cs} "
        "--customer-meta {input.customer_meta} --output-dir {params.out_dir}"

rule check_vk_price_stability:
    """Sanity-check vk_per_item lookup-like behaviour and time drift; writes summary CSV and two plots to data/06_plots."""
    input:
        plis = PLIS_TRAINING_CSV,
    output:
        plot_lookup = VK_STABILITY_PLOT_LOOKUP,
        plot_drift = VK_STABILITY_PLOT_DRIFT,
        summary_csv = VK_STABILITY_SUMMARY_CSV,
    params:
        out_dir = PLOTS_DIR,
    shell:
        "uv run src/check_vk_price_stability.py --plis {input.plis} --output-dir {params.out_dir}"

rule prepare_split_customer_meta:
    """Relabel task=none customers to task=testing so their purchase-value distribution matches warm (warm-matched selection)."""
    input:
        customer = CUSTOMER_META_CSV,
        plis = PLIS_TRAINING_CSV,
    output:
        customer = SPLIT_CUSTOMER_CSV,
    params:
        cutoff = SPLIT["cutoff_date"],
        n_testing = SPLIT["test_customers_count"],
        seed = SPLIT["random_seed"],
    shell:
        "uv run src/prepare_split_customer_meta.py --customer-meta {input.customer} --plis {input.plis} "
        "--output {output.customer} --cutoff-date {params.cutoff} --n-testing {params.n_testing} --random-seed {params.seed}"

rule split_plis_training_validation:
    """Split plis_training into training and testing: customers with task=testing; their rows with orderdate >= cutoff go to test, rest to training."""
    input:
        plis = PLIS_TRAINING_CSV,
        customer = SPLIT_CUSTOMER_CSV,
    output:
        train = PLIS_TRAINING_SPLIT,
        test = PLIS_TESTING_SPLIT,
    params:
        cutoff = SPLIT["cutoff_date"],
    shell:
        "uv run src/split_plis_training_validation.py --input {input.plis} --customer-meta {input.customer} "
        "--train {output.train} --test {output.test} --cutoff-date {params.cutoff}"

rule generate_trending_classes:
    """Build trending eclass list from training split data."""
    input:
        plis = PLIS_TRAINING_SPLIT,
    output:
        trending_classes = TRENDING_CLASSES_CSV,
    params:
        train_end = WIN["train_end"],
    shell:
        "uv run src/generate_trending_classes.py --plis {input.plis} --output {output.trending_classes} --train-end {params.train_end}"

rule generate_candidates:
    """Candidate generation per level: Level 1 = (legal_entity_id, eclass); Level 2 = (legal_entity_id, eclass, manufacturer)."""
    input:
        plis = PLIS_TRAINING_SPLIT,
        customer = lambda w: MODE_CFG[w.mode]["customer_csv"],
        trending_classes = TRENDING_CLASSES_CSV,
    output:
        candidates_raw = CANDIDATES_RAW_PATTERN,
    params:
        train_end = WIN["train_end"],
        lookback_months = CAND["lookback_months"],
        min_order_frequency = CAND["min_order_frequency"],
        min_lookback_spend = CAND["min_lookback_spend"],
    wildcard_constraints:
        mode = MODE_RE,
        level = LEVEL_RE,
    shell:
        "uv run src/generate_candidates.py --plis {input.plis} --customer {input.customer} "
        "--trending-classes {input.trending_classes} --output {output.candidates_raw} --train-end {params.train_end} --level {wildcards.level} "
        "--lookback-months {params.lookback_months} --min-order-frequency {params.min_order_frequency} --min-lookback-spend {params.min_lookback_spend}"

rule engineer_features_raw:
    """Assembly: keys from candidates + aggregates from PLIs + customer context + top-K SKU attributes; level2 includes manufacturer."""
    input:
        candidates_raw = CANDIDATES_RAW_PATTERN,
        plis = PLIS_TRAINING_SPLIT,
        customer = lambda w: MODE_CFG[w.mode]["customer_csv"],
        features_per_sku = INPUTS["features_per_sku"],
    output:
        features_raw = FEATURES_RAW_PATTERN,
    params:
        train_end = WIN["train_end"],
        top_k_keys = MODELLING["features_per_sku"]["top_k_keys"],
        top_k_values_per_key = MODELLING["features_per_sku"]["top_k_values_per_key"],
        chunksize = MODELLING["features_per_sku"]["chunksize"],
    wildcard_constraints:
        mode = MODE_RE,
        level = LEVEL_RE,
    shell:
        "uv run src/engineer_features_raw.py --candidates-raw {input.candidates_raw} --plis {input.plis} "
        "--customer {input.customer} --features-per-sku {input.features_per_sku} --output {output.features_raw} --level {wildcards.level} "
        "--train-end {params.train_end} --top-k-keys {params.top_k_keys} --top-k-values-per-key {params.top_k_values_per_key} --chunksize {params.chunksize}"

rule sanitize_features_raw:
    """Apply per-feature missing-value policies (drop_row, ignore, fill) from config to features_raw."""
    input:
        features_raw = FEATURES_RAW_PATTERN,
    output:
        features_sanitized = FEATURES_SANITIZED_PATTERN,
    params:
        config_file = "config.yaml",
        config_key = "modelling.missing_value_sanitation",
    wildcard_constraints:
        mode = MODE_RE,
        level = LEVEL_RE,
    shell:
        "uv run src/sanitize_features_raw.py --features-raw {input.features_raw} --output {output.features_sanitized} "
        "--config {params.config_file} --config-key {params.config_key} --level {wildcards.level}"

rule engineer_features_derived:
    """Derived feature engineering: raw features + plis/customer/nace -> full feature matrix; level2 keeps manufacturer in key."""
    input:
        features_raw = FEATURES_RAW_PATTERN,
        plis = PLIS_TRAINING_SPLIT,
        customer = lambda w: MODE_CFG[w.mode]["customer_csv"],
        nace_codes = INPUTS["nace_codes"],
    output:
        features_all = FEATURES_ALL_PATTERN,
    params:
        train_end = WIN["train_end"],
        lookback_months = CAND["lookback_months"],
    wildcard_constraints:
        mode = MODE_RE,
        level = LEVEL_RE,
    shell:
        "uv run src/engineer_features_derived.py --features-raw {input.features_raw} --plis {input.plis} "
        "--customer {input.customer} --nace-codes {input.nace_codes} --output {output.features_all} --train-end {params.train_end} --lookback-months {params.lookback_months} --level {wildcards.level}"

FEATURE_ANALYSIS_REDUNDANCY_PATTERN = f"{FEATURE_ANALYSIS_DIR}/{{mode}}/level{{level}}/feature_redundancy.csv"

rule feature_analysis:
    """Summary statistics, target-aware signal, redundancy, and plots for all engineered features (per mode and level)."""
    input:
        features_all = FEATURES_ALL_PATTERN,
        plis = PLIS_TRAINING_SPLIT,
    output:
        summary_csv = FEATURE_ANALYSIS_SUMMARY_PATTERN,
        redundancy_csv = FEATURE_ANALYSIS_REDUNDANCY_PATTERN,
        distributions_plot = f"{FEATURE_ANALYSIS_DIR}/{{mode}}/level{{level}}/feature_distributions.png",
        correlations_plot = f"{FEATURE_ANALYSIS_DIR}/{{mode}}/level{{level}}/feature_correlations.png",
    params:
        val_start = VAL["start"],
        val_end = VAL["end"],
        n_min_label = VAL["n_min_label"],
    wildcard_constraints:
        mode = MODE_RE,
        level = LEVEL_RE,
    shell:
        "uv run src/feature_analysis.py --features {input.features_all} --summary-csv {output.summary_csv} "
        "--plis {input.plis} --val-start {params.val_start} --val-end {params.val_end} --n-min-label {params.n_min_label} "
        "--redundancy-csv {output.redundancy_csv} "
        "--distributions-plot {output.distributions_plot} --correlations-plot {output.correlations_plot}"

rule suggest_features:
    """Suggest feature list from feature_summary.csv and feature_redundancy.csv for manual use in config modelling.features.selected."""
    input:
        summary_csv = FEATURE_ANALYSIS_SUMMARY_PATTERN,
        redundancy_csv = FEATURE_ANALYSIS_REDUNDANCY_PATTERN,
    output:
        suggestions_yaml = FEATURE_ANALYSIS_SUGGESTIONS_PATTERN,
    wildcard_constraints:
        mode = MODE_RE,
        level = LEVEL_RE,
    shell:
        "uv run src/suggest_features.py --summary-csv {input.summary_csv} --redundancy-csv {input.redundancy_csv} --output {output.suggestions_yaml}"

rule feature_selection:
    """Keep keys + config-driven selected features for downstream modelling (post feature_analysis); level2 keys include manufacturer."""
    input:
        features_all = FEATURES_ALL_PATTERN,
        feature_analysis_summary = FEATURE_ANALYSIS_SUMMARY_PATTERN,
    output:
        features_selected = FEATURES_SELECTED_PATTERN,
    params:
        selected_features = ",".join(FEAT["selected"]),
    wildcard_constraints:
        mode = MODE_RE,
        level = LEVEL_RE,
    shell:
        "uv run src/feature_selection.py --features {input.features_all} --selected-features '{params.selected_features}' "
        "--output {output.features_selected} --level {wildcards.level}"

rule train_approach:
    """Run a modelling approach (baseline or lgbm_two_stage); outputs data/12_predictions/{mode}/{approach}/level{level}/scores.parquet. Excludes hybrid_lgbm_phase3 (no training)."""
    input:
        candidates = FEATURES_SELECTED_PATTERN,
        plis = PLIS_TRAINING_SPLIT,
    output:
        scores = SCORES_APPROACH_PATTERN,
    params:
        val_start = VAL["start"],
        val_end = VAL["end"],
        n_min_label = VAL["n_min_label"],
        alpha = APP["baseline"]["alpha"],
        beta = APP["baseline"]["beta"],
        gamma = APP["baseline"]["gamma"],
        savings_rate = SCORING_PARAMS["savings_rate"],
        fixed_fee_eur = SCORING_PARAMS["fixed_fee_eur"],
        val_months = VAL["months"],
        eta = APP["phase3_repro"]["eta"],
        tau = APP["phase3_repro"]["tau"],
        sparse_eta_multiplier = APP["phase3_repro"]["sparse_eta_multiplier"],
        sparse_tau_multiplier = APP["phase3_repro"]["sparse_tau_multiplier"],
        use_monthly_lookback_rates = 1 if APP.get("phase3_repro", {}).get("use_monthly_lookback_rates", False) else 0,
        lgb_params_classifier = lambda w: APP.get(w.approach, {}).get("lgb_params_classifier", APP.get("lgbm_two_stage", {}).get("lgb_params_classifier", "")),
        lgb_params_regressor = lambda w: APP.get(w.approach, {}).get("lgb_params_regressor", APP.get("lgbm_two_stage", {}).get("lgb_params_regressor", "")),
        min_positive_samples_for_regressor = lambda w: APP.get("lgbm_two_stage", {}).get("min_positive_samples_for_regressor", 10),
        recency_decay_days = lambda w: APP.get("phase3_repro", {}).get("recency_decay_days", 365.0),
        score_base_constant = lambda w: APP.get("pass_through", {}).get("score_base_constant", 1.0),
    wildcard_constraints:
        mode = MODE_RE,
        approach = APPROACH_RE_WITH_SCORES,
        level = LEVEL_RE,
    shell:
        "uv run python src/modelling/run.py --approach {wildcards.approach} --level {wildcards.level} "
        "--candidates {input.candidates} --plis {input.plis} --output {output.scores} "
        "--val-start {params.val_start} --val-end {params.val_end} --n-min-label {params.n_min_label} "
        "--alpha {params.alpha} --beta {params.beta} --gamma {params.gamma} "
        "--savings-rate {params.savings_rate} --fixed-fee-eur {params.fixed_fee_eur} --val-months {params.val_months} "
        "--eta {params.eta} --tau {params.tau} "
        "--sparse-eta-multiplier {params.sparse_eta_multiplier} --sparse-tau-multiplier {params.sparse_tau_multiplier} "
        "--use-monthly-lookback-rates {params.use_monthly_lookback_rates} "
        "--lgb-params-classifier '{params.lgb_params_classifier}' --lgb-params-regressor '{params.lgb_params_regressor}' "
        "--min-positive-samples-for-regressor {params.min_positive_samples_for_regressor} "
        "--recency-decay-days {params.recency_decay_days} --score-base-constant {params.score_base_constant}"

rule train_approach_run:
    """Run-scoped: same as train_approach but output under level{level}/{run_id}/ for sweep trials."""
    input:
        candidates = FEATURES_SELECTED_PATTERN,
        plis = PLIS_TRAINING_SPLIT,
    output:
        scores = SCORES_APPROACH_PATTERN_RUN,
    params:
        val_start = VAL["start"],
        val_end = VAL["end"],
        n_min_label = VAL["n_min_label"],
        alpha = APP["baseline"]["alpha"],
        beta = APP["baseline"]["beta"],
        gamma = APP["baseline"]["gamma"],
        savings_rate = SCORING_PARAMS["savings_rate"],
        fixed_fee_eur = SCORING_PARAMS["fixed_fee_eur"],
        val_months = VAL["months"],
        eta = APP["phase3_repro"]["eta"],
        tau = APP["phase3_repro"]["tau"],
        sparse_eta_multiplier = APP["phase3_repro"]["sparse_eta_multiplier"],
        sparse_tau_multiplier = APP["phase3_repro"]["sparse_tau_multiplier"],
        use_monthly_lookback_rates = 1 if APP.get("phase3_repro", {}).get("use_monthly_lookback_rates", False) else 0,
        lgb_params_classifier = lambda w: APP.get(w.approach, {}).get("lgb_params_classifier", APP.get("lgbm_two_stage", {}).get("lgb_params_classifier", "")),
        lgb_params_regressor = lambda w: APP.get(w.approach, {}).get("lgb_params_regressor", APP.get("lgbm_two_stage", {}).get("lgb_params_regressor", "")),
        min_positive_samples_for_regressor = lambda w: APP.get("lgbm_two_stage", {}).get("min_positive_samples_for_regressor", 10),
        recency_decay_days = lambda w: APP.get("phase3_repro", {}).get("recency_decay_days", 365.0),
        score_base_constant = lambda w: APP.get("pass_through", {}).get("score_base_constant", 1.0),
    wildcard_constraints:
        mode = MODE_RE,
        approach = APPROACH_RE_WITH_SCORES,
        level = LEVEL_RE,
        run_id = RUN_ID_RE,
    shell:
        "uv run python src/modelling/run.py --approach {wildcards.approach} --level {wildcards.level} "
        "--candidates {input.candidates} --plis {input.plis} --output {output.scores} "
        "--val-start {params.val_start} --val-end {params.val_end} --n-min-label {params.n_min_label} "
        "--alpha {params.alpha} --beta {params.beta} --gamma {params.gamma} "
        "--savings-rate {params.savings_rate} --fixed-fee-eur {params.fixed_fee_eur} --val-months {params.val_months} "
        "--eta {params.eta} --tau {params.tau} "
        "--sparse-eta-multiplier {params.sparse_eta_multiplier} --sparse-tau-multiplier {params.sparse_tau_multiplier} "
        "--use-monthly-lookback-rates {params.use_monthly_lookback_rates} "
        "--lgb-params-classifier '{params.lgb_params_classifier}' --lgb-params-regressor '{params.lgb_params_regressor}' "
        "--min-positive-samples-for-regressor {params.min_positive_samples_for_regressor} "
        "--recency-decay-days {params.recency_decay_days} --score-base-constant {params.score_base_constant}"

rule select_portfolio:
    """Apply EU threshold, guardrails and per-buyer cap K to produce portfolio.parquet per approach and level; level2 portfolio includes manufacturer. Uses level-specific selection when by_level is set. Excludes hybrid_lgbm_phase3 (use rule portfolio_hybrid)."""
    input:
        scores = SCORES_APPROACH_PATTERN,
    output:
        portfolio = PORTFOLIO_PATTERN,
    params:
        score_threshold = lambda w: _selection_for_level(w.level)["score_threshold"],
        min_orders_guardrail = lambda w: _selection_for_level(w.level)["min_orders_guardrail"],
        min_months_guardrail = lambda w: _selection_for_level(w.level)["min_months_guardrail"],
        high_spend_guardrail = lambda w: _selection_for_level(w.level)["high_spend_guardrail"],
        min_avg_monthly_spend = lambda w: _selection_for_level(w.level)["min_avg_monthly_spend"],
        top_k_per_buyer = lambda w: _selection_for_level(w.level)["top_k_per_buyer"],
    wildcard_constraints:
        mode = MODE_RE,
        approach = APPROACH_RE_WITH_SCORES,
        level = LEVEL_RE,
    shell:
        "uv run src/select_portfolio.py --scores {input.scores} --output {output.portfolio} --level {wildcards.level} "
        "--score-threshold {params.score_threshold} --min-orders-guardrail {params.min_orders_guardrail} "
        "--min-months-guardrail {params.min_months_guardrail} --high-spend-guardrail {params.high_spend_guardrail} "
        "--min-avg-monthly-spend {params.min_avg_monthly_spend} --top-k-per-buyer {params.top_k_per_buyer}"

rule select_portfolio_run:
    """Run-scoped: same as select_portfolio but under level{level}/{run_id}/ for sweep trials."""
    input:
        scores = SCORES_APPROACH_PATTERN_RUN,
    output:
        portfolio = PORTFOLIO_PATTERN_RUN,
    params:
        score_threshold = lambda w: _selection_for_level(w.level)["score_threshold"],
        min_orders_guardrail = lambda w: _selection_for_level(w.level)["min_orders_guardrail"],
        min_months_guardrail = lambda w: _selection_for_level(w.level)["min_months_guardrail"],
        high_spend_guardrail = lambda w: _selection_for_level(w.level)["high_spend_guardrail"],
        min_avg_monthly_spend = lambda w: _selection_for_level(w.level)["min_avg_monthly_spend"],
        top_k_per_buyer = lambda w: _selection_for_level(w.level)["top_k_per_buyer"],
    wildcard_constraints:
        mode = MODE_RE,
        approach = APPROACH_RE_WITH_SCORES,
        level = LEVEL_RE,
        run_id = RUN_ID_RE,
    shell:
        "uv run src/select_portfolio.py --scores {input.scores} --output {output.portfolio} --level {wildcards.level} "
        "--score-threshold {params.score_threshold} --min-orders-guardrail {params.min_orders_guardrail} "
        "--min-months-guardrail {params.min_months_guardrail} --high-spend-guardrail {params.high_spend_guardrail} "
        "--min-avg-monthly-spend {params.min_avg_monthly_spend} --top-k-per-buyer {params.top_k_per_buyer}"

rule portfolio_hybrid:
    """Merge lgbm_two_stage (primary) and phase3_repro (secondary) portfolios up to target_per_buyer per buyer. Enable by adding hybrid_lgbm_phase3 to modelling.enabled_approaches."""
    input:
        primary = f"{DATA_DIR}/13_portfolio/{{mode}}/lgbm_two_stage/level{{level}}/portfolio.parquet",
        secondary = f"{DATA_DIR}/13_portfolio/{{mode}}/phase3_repro/level{{level}}/portfolio.parquet",
        scores_secondary = f"{DATA_DIR}/12_predictions/{{mode}}/phase3_repro/level{{level}}/scores.parquet",
    output:
        portfolio = f"{DATA_DIR}/13_portfolio/{{mode}}/hybrid_lgbm_phase3/level{{level}}/portfolio.parquet",
    params:
        target_per_buyer = 400,
    wildcard_constraints:
        mode = MODE_RE,
        level = LEVEL_RE,
    shell:
        "uv run src/merge_portfolio_hybrid.py --primary {input.primary} --secondary {input.secondary} "
        "--scores-secondary {input.scores_secondary} --output {output.portfolio} --level {wildcards.level} "
        "--target-per-buyer {params.target_per_buyer}"

rule write_submission_warm_only:
    """Write submission rows for warm buyers only (portfolio items per buyer)."""
    input:
        portfolio = PORTFOLIO_PATTERN,
        customer = lambda w: MODE_CFG[w.mode]["customer_input"],
        plis = PLIS_TRAINING_SPLIT,
    output:
        submission_warm = SUBMISSION_WARM_PART_PATTERN,
    params:
        buyer_source = lambda w: MODE_CFG[w.mode]["buyer_source"],
    wildcard_constraints:
        mode = MODE_RE,
        approach = APPROACH_RE,
        level = LEVEL_RE,
    run:
        arg = "--customer-test" if wildcards.mode == "online" else "--customer-split"
        shell(
            "uv run src/write_submission_warm.py --portfolio {input.portfolio} "
            "--buyer-source {params.buyer_source} " + arg + " {input.customer} --plis-training {input.plis} "
            "--level {wildcards.level} --mode warm_only --output {output.submission_warm}"
        )

rule write_submission_warm_only_run:
    """Run-scoped: warm submission part under level{level}/{run_id}/ for sweep trials."""
    input:
        portfolio = PORTFOLIO_PATTERN_RUN,
        customer = lambda w: MODE_CFG[w.mode]["customer_input"],
        plis = PLIS_TRAINING_SPLIT,
    output:
        submission_warm = SUBMISSION_WARM_PART_PATTERN_RUN,
    params:
        buyer_source = lambda w: MODE_CFG[w.mode]["buyer_source"],
    wildcard_constraints:
        mode = MODE_RE,
        approach = APPROACH_RE,
        level = LEVEL_RE,
        run_id = RUN_ID_RE,
    run:
        arg = "--customer-test" if wildcards.mode == "online" else "--customer-split"
        shell(
            "uv run src/write_submission_warm.py --portfolio {input.portfolio} "
            "--buyer-source {params.buyer_source} " + arg + " {input.customer} --plis-training {input.plis} "
            "--level {wildcards.level} --mode warm_only --output {output.submission_warm}"
        )

rule write_submission_cold_only:
    """Write submission rows for cold-start buyers only (NACE/score-based fallback). Uses level-specific cold_start_top_k when submission.by_level is set. For hybrid_lgbm_phase3 uses phase3_repro scores."""
    input:
        portfolio = PORTFOLIO_PATTERN,
        customer = lambda w: MODE_CFG[w.mode]["customer_input"],
        plis = PLIS_TRAINING_SPLIT,
        nace_codes = INPUTS["nace_codes"],
        scores = lambda w: f"{DATA_DIR}/12_predictions/{w.mode}/phase3_repro/level{w.level}/scores.parquet" if w.approach == "hybrid_lgbm_phase3" else f"{DATA_DIR}/12_predictions/{w.mode}/{w.approach}/level{w.level}/scores.parquet",
    output:
        submission_cold = SUBMISSION_COLD_PART_PATTERN,
    params:
        buyer_source = lambda w: MODE_CFG[w.mode]["buyer_source"],
        cold_start_top_k = lambda w: _cold_start_top_k_for_level(w.level),
    wildcard_constraints:
        mode = MODE_RE,
        approach = APPROACH_RE,
        level = LEVEL_RE,
    run:
        arg = "--customer-test" if wildcards.mode == "online" else "--customer-split"
        shell(
            "uv run src/write_submission_warm.py --portfolio {input.portfolio} "
            "--buyer-source {params.buyer_source} " + arg + " {input.customer} --plis-training {input.plis} "
            "--nace-codes {input.nace_codes} --scores {input.scores} --level {wildcards.level} "
            "--cold-start-top-k {params.cold_start_top_k} --mode cold_only --output {output.submission_cold}"
        )

rule write_submission_cold_only_run:
    """Run-scoped: cold submission part under level{level}/{run_id}/ for sweep trials."""
    input:
        portfolio = PORTFOLIO_PATTERN_RUN,
        customer = lambda w: MODE_CFG[w.mode]["customer_input"],
        plis = PLIS_TRAINING_SPLIT,
        nace_codes = INPUTS["nace_codes"],
        scores = lambda w: f"{DATA_DIR}/12_predictions/{w.mode}/phase3_repro/level{w.level}/{w.run_id}/scores.parquet" if w.approach == "hybrid_lgbm_phase3" else f"{DATA_DIR}/12_predictions/{w.mode}/{w.approach}/level{w.level}/{w.run_id}/scores.parquet",
    output:
        submission_cold = SUBMISSION_COLD_PART_PATTERN_RUN,
    params:
        buyer_source = lambda w: MODE_CFG[w.mode]["buyer_source"],
        cold_start_top_k = lambda w: _cold_start_top_k_for_level(w.level),
    wildcard_constraints:
        mode = MODE_RE,
        approach = APPROACH_RE,
        level = LEVEL_RE,
        run_id = RUN_ID_RE,
    run:
        arg = "--customer-test" if wildcards.mode == "online" else "--customer-split"
        shell(
            "uv run src/write_submission_warm.py --portfolio {input.portfolio} "
            "--buyer-source {params.buyer_source} " + arg + " {input.customer} --plis-training {input.plis} "
            "--nace-codes {input.nace_codes} --scores {input.scores} --level {wildcards.level} "
            "--cold-start-top-k {params.cold_start_top_k} --mode cold_only --output {output.submission_cold}"
        )

rule merge_submission_parts:
    """Concatenate warm and cold submission parts and deduplicate into final submission.csv."""
    input:
        warm = SUBMISSION_WARM_PART_PATTERN,
        cold = SUBMISSION_COLD_PART_PATTERN,
    output:
        submission = SUBMISSION_PATTERN,
    wildcard_constraints:
        mode = MODE_RE,
        approach = APPROACH_RE,
        level = LEVEL_RE,
    shell:
        "uv run python -c \""
        "import pandas as pd; "
        "w = pd.read_csv('{input.warm}'); "
        "c = pd.read_csv('{input.cold}'); "
        "out = pd.concat([w, c]).drop_duplicates(subset=['legal_entity_id', 'cluster']); "
        "out.to_csv('{output.submission}', index=False)"
        "\""

rule merge_submission_parts_run:
    """Run-scoped: merge warm and cold parts under level{level}/{run_id}/ for sweep trials."""
    input:
        warm = SUBMISSION_WARM_PART_PATTERN_RUN,
        cold = SUBMISSION_COLD_PART_PATTERN_RUN,
    output:
        submission = SUBMISSION_PATTERN_RUN,
    wildcard_constraints:
        mode = MODE_RE,
        approach = APPROACH_RE,
        level = LEVEL_RE,
        run_id = RUN_ID_RE,
    shell:
        "uv run python -c \""
        "import pandas as pd; "
        "w = pd.read_csv('{input.warm}'); "
        "c = pd.read_csv('{input.cold}'); "
        "out = pd.concat([w, c]).drop_duplicates(subset=['legal_entity_id', 'cluster']); "
        "out.to_csv('{output.submission}', index=False)"
        "\""

rule write_submission:
    """Write baseline submission CSV with required header (legal_entity_id,cluster). Use 'snakemake data/14_submission/submission_baseline.csv' to run."""
    input:
        customer_test = INPUTS["customer_test"],
        plis = PLIS_TRAINING_CSV,
    output:
        submission = f"{DATA_DIR}/14_submission/submission_baseline.csv",
    shell:
        "uv run src/write_submission.py --output {output.submission} "
        "--customer-test {input.customer_test} --plis-training {input.plis}"

rule archive_score_run:
    """Copy live score summary into a timestamp+sha run folder under data/15_scores/online/runs/level{level}/ (score_summary_live.csv + metadata.json only). Archives level-specific selection params when by_level is set."""
    input:
        live_summary = LIVE_SUMMARY_TEMP_PATTERN,
        submit_done = SUBMITTED_SENTINEL_PATTERN,
    output:
        sentinel = ARCHIVE_SENTINEL_PATTERN,
    params:
        runs_dir = lambda w: f"{SCORES_DIR}/online/runs/level{w.level}",
        train_end = WIN["train_end"],
        lookback_months = CAND["lookback_months"],
        score_threshold = lambda w: _selection_for_level(w.level)["score_threshold"],
        top_k_per_buyer = lambda w: _selection_for_level(w.level)["top_k_per_buyer"],
        min_orders = lambda w: _selection_for_level(w.level)["min_orders_guardrail"],
        min_months = lambda w: _selection_for_level(w.level)["min_months_guardrail"],
        high_spend = lambda w: _selection_for_level(w.level)["high_spend_guardrail"],
        min_avg_monthly_spend = lambda w: _selection_for_level(w.level)["min_avg_monthly_spend"],
        cold_start_top_k = lambda w: _cold_start_top_k_for_level(w.level),
        selected_features = ",".join(FEAT["selected"]),
    wildcard_constraints:
        approach = APPROACH_RE,
        level = LEVEL_RE,
    shell:
        "uv run src/archive_score_run.py --live-summary {input.live_summary} "
        "--runs-dir {params.runs_dir} --approach {wildcards.approach} --level {wildcards.level} "
        "--train-end {params.train_end} --lookback-months {params.lookback_months} "
        "--score-threshold {params.score_threshold} --top-k-per-buyer {params.top_k_per_buyer} "
        "--min-orders {params.min_orders} --min-months {params.min_months} --high-spend {params.high_spend} "
        "--min-avg-monthly-spend {params.min_avg_monthly_spend} --cold-start-top-k {params.cold_start_top_k} "
        "--selected-features '{params.selected_features}' && touch {output.sentinel}"

rule archive_score_run_run:
    """Run-scoped: archive live score into runs/level{level}/{approach}/{run_id}/ using pre-generated run_id (sweep)."""
    input:
        live_summary = LIVE_SUMMARY_TEMP_PATTERN_RUN,
        submit_done = SUBMITTED_SENTINEL_PATTERN_RUN,
    output:
        sentinel = ARCHIVE_SENTINEL_RUN_PATTERN,
    params:
        runs_dir = lambda w: f"{SCORES_DIR}/online/runs/level{w.level}/{w.approach}",
        train_end = WIN["train_end"],
        lookback_months = CAND["lookback_months"],
        score_threshold = lambda w: _selection_for_level(w.level)["score_threshold"],
        top_k_per_buyer = lambda w: _selection_for_level(w.level)["top_k_per_buyer"],
        min_orders = lambda w: _selection_for_level(w.level)["min_orders_guardrail"],
        min_months = lambda w: _selection_for_level(w.level)["min_months_guardrail"],
        high_spend = lambda w: _selection_for_level(w.level)["high_spend_guardrail"],
        min_avg_monthly_spend = lambda w: _selection_for_level(w.level)["min_avg_monthly_spend"],
        cold_start_top_k = lambda w: _cold_start_top_k_for_level(w.level),
        selected_features = ",".join(FEAT["selected"]),
    wildcard_constraints:
        approach = APPROACH_RE,
        level = LEVEL_RE,
        run_id = RUN_ID_RE,
    shell:
        "uv run src/archive_score_run.py --live-summary {input.live_summary} "
        "--runs-dir {params.runs_dir} --run-id {wildcards.run_id} --approach {wildcards.approach} --level {wildcards.level} "
        "--train-end {params.train_end} --lookback-months {params.lookback_months} "
        "--score-threshold {params.score_threshold} --top-k-per-buyer {params.top_k_per_buyer} "
        "--min-orders {params.min_orders} --min-months {params.min_months} --high-spend {params.high_spend} "
        "--min-avg-monthly-spend {params.min_avg_monthly_spend} --cold-start-top-k {params.cold_start_top_k} "
        "--selected-features '{params.selected_features}'"

rule copy_best_online_run:
    """Select best historic online run per level by total_score and copy that run directory into data/16_scores_best/online/level{level}/best_run/."""
    input:
        archived_runs = expand(ARCHIVE_SENTINEL_PATTERN, approach=ENABLED_APPROACHES, level="{level}"),
    output:
        sentinel = BEST_RUN_COPIED_SENTINEL_PATTERN,
    params:
        scores_dir = f"{SCORES_DIR}/online",
        best_run_dir = lambda w: f"{SCORES_BEST_DIR}/online/level{w.level}/best_run",
    wildcard_constraints:
        level = LEVEL_RE,
    shell:
        "uv run src/select_best_score_run.py --scores-dir {params.scores_dir} --level {wildcards.level} "
        "--best-run-dir {params.best_run_dir} && touch {output.sentinel}"

rule analyze_submission_tuning:
    """Produce submission-tuning diagnostics: run metrics, param effects, submission shape CSVs and plots per level (for hyperparameter choice)."""
    input:
        submissions = expand(
            SUBMISSION_PATTERN,
            mode=["online"],
            approach=ENABLED_APPROACHES,
            level="{level}",
        ),
        customer_test = INPUTS["customer_test"],
    output:
        run_metrics = f"{SUBMISSION_TUNING_DIR}/run_metrics_level{{level}}.csv",
        run_records = f"{SUBMISSION_TUNING_DIR}/run_records_level{{level}}.jsonl",
        param_effects = f"{SUBMISSION_TUNING_DIR}/param_effects_level{{level}}.csv",
        current_shape = f"{SUBMISSION_TUNING_DIR}/current_submission_shape_level{{level}}.csv",
        plot_score_predictions = f"{SUBMISSION_TUNING_DIR}/score_vs_predictions_level{{level}}.png",
        plot_score_capture = f"{SUBMISSION_TUNING_DIR}/score_vs_capture_level{{level}}.png",
        plot_timeline = f"{SUBMISSION_TUNING_DIR}/run_timeline_level{{level}}.png",
        plot_size_score = f"{SUBMISSION_TUNING_DIR}/submission_size_vs_score_level{{level}}.png",
    params:
        runs_dir = f"{SCORES_DIR}/online/runs/level{{level}}",
        best_run_dir = f"{SCORES_BEST_DIR}/online/level{{level}}/best_run",
        out_dir = SUBMISSION_TUNING_DIR,
    wildcard_constraints:
        level = LEVEL_RE,
    shell:
        "uv run src/analyze_submission_tuning.py --level {wildcards.level} --output-dir {params.out_dir} "
        "--runs-dir {params.runs_dir} --best-run-dir {params.best_run_dir} "
        "--customer-test {input.customer_test} --submissions {input.submissions}"

rule submit_to_portal:
    """Upload online submission to Unite evaluator (challenge 2) per approach and level. Writes live score to temp file for archive_score_run. Requires portal_credentials in config.yaml."""
    input:
        submission = f"{DATA_DIR}/14_submission/online/{{approach}}/level{{level}}/submission.csv",
    output:
        summary = LIVE_SUMMARY_TEMP_PATTERN,
        sentinel = SUBMITTED_SENTINEL_PATTERN,
    resources:
        portal_submit_slot=1,
    wildcard_constraints:
        approach = APPROACH_RE,
        level = LEVEL_RE,
    shell:
        "uv run src/submit.py --challenge 2 --file {input.submission} --level {wildcards.level} "
        "--summary-csv {output.summary} && touch {output.sentinel}"

rule submit_to_portal_run:
    """Run-scoped: upload submission under level{level}/{run_id}/ to portal; write run-scoped summary and sentinel."""
    input:
        submission = f"{DATA_DIR}/14_submission/online/{{approach}}/level{{level}}/{{run_id}}/submission.csv",
    output:
        summary = LIVE_SUMMARY_TEMP_PATTERN_RUN,
        sentinel = SUBMITTED_SENTINEL_PATTERN_RUN,
    resources:
        portal_submit_slot=1,
    wildcard_constraints:
        approach = APPROACH_RE,
        level = LEVEL_RE,
        run_id = RUN_ID_RE,
    shell:
        "uv run src/submit.py --challenge 2 --file {input.submission} --level {wildcards.level} "
        "--summary-csv {output.summary} && touch {output.sentinel}"
