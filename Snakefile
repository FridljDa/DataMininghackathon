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

# Modelling (params only from config; paths are patterns below)
MODELLING = config["modelling"]
WIN = MODELLING["windows"]
VAL = WIN["validation"]
CAND = MODELLING["candidates"]
SEL = MODELLING["selection"]
GRD = SEL["guardrails"]
FEAT = MODELLING["features"]
APP = MODELLING["approaches"]

# Feature analysis
FEATURE_ANALYSIS_DIR = f"{DATA_DIR}/10_feature_analysis"
FEATURE_ANALYSIS_SUMMARY_CSV = f"{FEATURE_ANALYSIS_DIR}/online/feature_summary.csv"
FEATURE_ANALYSIS_SUMMARY_OFFLINE_CSV = f"{FEATURE_ANALYSIS_DIR}/offline/feature_summary.csv"
FEATURE_ANALYSIS_PLOT_FILES = ["online/feature_distributions.png", "online/feature_correlations.png"]
FEATURE_ANALYSIS_PLOT_FILES_OFFLINE = ["offline/feature_distributions.png", "offline/feature_correlations.png"]
FEATURE_ANALYSIS_PLOTS = expand(f"{FEATURE_ANALYSIS_DIR}/{{f}}", f=FEATURE_ANALYSIS_PLOT_FILES)
FEATURE_ANALYSIS_PLOTS_OFFLINE = expand(f"{FEATURE_ANALYSIS_DIR}/{{f}}", f=FEATURE_ANALYSIS_PLOT_FILES_OFFLINE)
FEATURE_ANALYSIS_SUGGESTION_ONLINE = f"{FEATURE_ANALYSIS_DIR}/online/feature_suggestions.yaml"
FEATURE_ANALYSIS_SUGGESTION_OFFLINE = f"{FEATURE_ANALYSIS_DIR}/offline/feature_suggestions.yaml"
FEATURE_ANALYSIS_SUGGESTIONS_PATTERN = f"{FEATURE_ANALYSIS_DIR}/{{mode}}/feature_suggestions.yaml"

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
# Path patterns for mode-wildcarded rules
CANDIDATES_RAW_PATTERN = f"{DATA_DIR}/07_candidates/{{mode}}/candidates_raw.parquet"
FEATURES_RAW_PATTERN = f"{DATA_DIR}/08_features_raw/{{mode}}/features_raw.parquet"
FEATURES_SANITIZED_PATTERN = f"{DATA_DIR}/08_features_sanitized/{{mode}}/features_sanitized.parquet"
FEATURES_ALL_PATTERN = f"{DATA_DIR}/09_features_derived/{{mode}}/features_all.parquet"
FEATURES_SELECTED_PATTERN = f"{DATA_DIR}/11_features_selected/{{mode}}/features_selected.parquet"
# Per-approach: scores, portfolio (no level); submission and scores per level (data/12–15 by {mode}/{approach}/level{level}/)
ENABLED_APPROACHES = MODELLING["enabled_approaches"]
ENABLED_LEVELS = MODELLING["enabled_levels"]
APPROACH_RE = "|".join(ENABLED_APPROACHES)
LEVEL_RE = "|".join(str(l) for l in ENABLED_LEVELS)
SCORES_APPROACH_PATTERN = f"{DATA_DIR}/12_predictions/{{mode}}/{{approach}}/scores.parquet"
PORTFOLIO_PATTERN = f"{DATA_DIR}/13_portfolio/{{mode}}/{{approach}}/portfolio.parquet"
SUBMISSION_PATTERN = f"{DATA_DIR}/14_submission/{{mode}}/{{approach}}/level{{level}}/submission.csv"
ARCHIVE_SENTINEL_PATTERN = f"{SCORES_DIR}/online/runs/level{{level}}/.last_archived_{{approach}}"
SUBMITTED_SENTINEL_PATTERN = f"{DATA_DIR}/14_submission/.submitted_challenge2_{{approach}}_level{{level}}"
LIVE_SUMMARY_TEMP_PATTERN = f"{DATA_DIR}/14_submission/.score_summary_live_{{approach}}_level{{level}}.csv"
FEATURE_ANALYSIS_SUMMARY_PATTERN = f"{FEATURE_ANALYSIS_DIR}/{{mode}}/feature_summary.csv"

rule all:
    input:
        DAG_SVG,
        PLIS_TRAINING_CSV,
        CUSTOMER_META_CSV,
        PLOT_OUTPUTS,
        FEATURE_ANALYSIS_SUMMARY_CSV,
        FEATURE_ANALYSIS_SUMMARY_OFFLINE_CSV,
        FEATURE_ANALYSIS_PLOTS,
        FEATURE_ANALYSIS_PLOTS_OFFLINE,
        FEATURE_ANALYSIS_SUGGESTION_ONLINE,
        FEATURE_ANALYSIS_SUGGESTION_OFFLINE,
        expand(SCORES_APPROACH_PATTERN, mode=MODES, approach=ENABLED_APPROACHES),
        expand(PORTFOLIO_PATTERN, mode=MODES, approach=ENABLED_APPROACHES),
        expand(SUBMISSION_PATTERN, mode=MODES, approach=ENABLED_APPROACHES, level=ENABLED_LEVELS),
        expand(ARCHIVE_SENTINEL_PATTERN, approach=ENABLED_APPROACHES, level=ENABLED_LEVELS),
        expand(SUBMITTED_SENTINEL_PATTERN, approach=ENABLED_APPROACHES, level=ENABLED_LEVELS),
        expand(BEST_RUN_COPIED_SENTINEL_PATTERN, level=ENABLED_LEVELS),

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

rule generate_candidates:
    """Candidate generation for Level 1 (warm buyers, E-Class): lookback window, n_orders(b,e,L) >= eta, s_lookback >= tau."""
    input:
        plis = PLIS_TRAINING_SPLIT,
        customer = lambda w: MODE_CFG[w.mode]["customer_csv"],
    output:
        candidates_raw = CANDIDATES_RAW_PATTERN,
    params:
        train_end = WIN["train_end"],
        lookback_months = CAND["lookback_months"],
        min_order_frequency = CAND["min_order_frequency"],
        min_lookback_spend = CAND["min_lookback_spend"],
    wildcard_constraints:
        mode = MODE_RE,
    shell:
        "uv run src/generate_candidates.py --plis {input.plis} --customer {input.customer} "
        "--output {output.candidates_raw} --train-end {params.train_end} "
        "--lookback-months {params.lookback_months} --min-order-frequency {params.min_order_frequency} "
        "--min-lookback-spend {params.min_lookback_spend}"

rule engineer_features_raw:
    """Pass-through raw feature columns from candidates (no derivation)."""
    input:
        candidates_raw = CANDIDATES_RAW_PATTERN,
    output:
        features_raw = FEATURES_RAW_PATTERN,
    wildcard_constraints:
        mode = MODE_RE,
    shell:
        "uv run src/engineer_features_raw.py --candidates-raw {input.candidates_raw} --output {output.features_raw}"

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
    shell:
        "uv run src/sanitize_features_raw.py --features-raw {input.features_raw} --output {output.features_sanitized} "
        "--config {params.config_file} --config-key {params.config_key}"

rule engineer_features_derived:
    """Derived feature engineering: raw features + plis/customer/nace -> full feature matrix."""
    input:
        features_sanitized = FEATURES_SANITIZED_PATTERN,
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
    shell:
        "uv run src/engineer_features_derived.py --features-raw {input.features_sanitized} --plis {input.plis} "
        "--customer {input.customer} --nace-codes {input.nace_codes} --output {output.features_all} --train-end {params.train_end} --lookback-months {params.lookback_months}"

FEATURE_ANALYSIS_REDUNDANCY_PATTERN = f"{FEATURE_ANALYSIS_DIR}/{{mode}}/feature_redundancy.csv"

rule feature_analysis:
    """Summary statistics, target-aware signal, redundancy, and plots for all engineered features."""
    input:
        features_all = FEATURES_ALL_PATTERN,
        plis = PLIS_TRAINING_SPLIT,
    output:
        summary_csv = FEATURE_ANALYSIS_SUMMARY_PATTERN,
        redundancy_csv = FEATURE_ANALYSIS_REDUNDANCY_PATTERN,
        distributions_plot = f"{FEATURE_ANALYSIS_DIR}/{{mode}}/feature_distributions.png",
        correlations_plot = f"{FEATURE_ANALYSIS_DIR}/{{mode}}/feature_correlations.png",
    params:
        val_start = VAL["start"],
        val_end = VAL["end"],
        n_min_label = VAL["n_min_label"],
    wildcard_constraints:
        mode = MODE_RE,
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
    shell:
        "uv run src/suggest_features.py --summary-csv {input.summary_csv} --redundancy-csv {input.redundancy_csv} --output {output.suggestions_yaml}"

rule feature_selection:
    """Keep keys + config-driven selected features for downstream modelling (post feature_analysis)."""
    input:
        features_all = FEATURES_ALL_PATTERN,
        feature_analysis_summary = FEATURE_ANALYSIS_SUMMARY_PATTERN,
    output:
        features_selected = FEATURES_SELECTED_PATTERN,
    params:
        selected_features = ",".join(FEAT["selected"]),
    wildcard_constraints:
        mode = MODE_RE,
    shell:
        "uv run src/feature_selection.py --features {input.features_all} --selected-features '{params.selected_features}' "
        "--output {output.features_selected}"

rule train_approach:
    """Run a modelling approach (baseline or lgbm_two_stage); outputs data/11_predictions/{mode}/{approach}/scores.parquet."""
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
    wildcard_constraints:
        mode = MODE_RE,
        approach = "|".join(ENABLED_APPROACHES),
    shell:
        "uv run python src/modelling/run.py --approach {wildcards.approach} "
        "--candidates {input.candidates} --plis {input.plis} --output {output.scores} "
        "--val-start {params.val_start} --val-end {params.val_end} --n-min-label {params.n_min_label} "
        "--alpha {params.alpha} --beta {params.beta} --gamma {params.gamma} "
        "--savings-rate {params.savings_rate} --fixed-fee-eur {params.fixed_fee_eur} --val-months {params.val_months} "
        "--eta {params.eta} --tau {params.tau} "
        "--sparse-eta-multiplier {params.sparse_eta_multiplier} --sparse-tau-multiplier {params.sparse_tau_multiplier} "
        "--use-monthly-lookback-rates {params.use_monthly_lookback_rates}"

rule select_portfolio:
    """Apply EU threshold, guardrails and per-buyer cap K to produce portfolio.parquet per approach."""
    input:
        scores = SCORES_APPROACH_PATTERN,
    output:
        portfolio = PORTFOLIO_PATTERN,
    params:
        score_threshold = SEL["score_threshold"],
        min_orders_guardrail = GRD["min_orders"],
        min_months_guardrail = GRD["min_months"],
        high_spend_guardrail = GRD["high_spend"],
        min_avg_monthly_spend = GRD.get("min_avg_monthly_spend", 0),
        top_k_per_buyer = SEL["top_k_per_buyer"],
    wildcard_constraints:
        mode = MODE_RE,
        approach = APPROACH_RE,
    shell:
        "uv run src/select_portfolio.py --scores {input.scores} --output {output.portfolio} "
        "--score-threshold {params.score_threshold} --min-orders-guardrail {params.min_orders_guardrail} "
        "--min-months-guardrail {params.min_months_guardrail} --high-spend-guardrail {params.high_spend_guardrail} "
        "--min-avg-monthly-spend {params.min_avg_monthly_spend} --top-k-per-buyer {params.top_k_per_buyer}"

rule write_submission_warm:
    """Format portfolio into submission CSV (Level 1: cluster=eclass; Level 2: cluster=eclass|manufacturer); cold-start fallback."""
    input:
        portfolio = PORTFOLIO_PATTERN,
        customer = lambda w: MODE_CFG[w.mode]["customer_input"],
        plis = PLIS_TRAINING_SPLIT,
        nace_codes = INPUTS["nace_codes"],
    output:
        submission = SUBMISSION_PATTERN,
    params:
        buyer_source = lambda w: MODE_CFG[w.mode]["buyer_source"],
        cold_start_top_k = config["submission"]["cold_start_top_k"],
    wildcard_constraints:
        mode = MODE_RE,
        approach = APPROACH_RE,
        level = LEVEL_RE,
    run:
        arg = "--customer-test" if wildcards.mode == "online" else "--customer-split"
        shell(
            "uv run src/write_submission_warm.py --portfolio {input.portfolio} "
            "--buyer-source {params.buyer_source} " + arg + " {input.customer} --plis-training {input.plis} "
            "--nace-codes {input.nace_codes} --level {wildcards.level} "
            "--cold-start-top-k {params.cold_start_top_k} --output {output.submission}"
        )


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
    """Copy live score summary into a timestamp+sha run folder under data/15_scores/online/runs/level{level}/ (score_summary_live.csv + metadata.json only)."""
    input:
        live_summary = LIVE_SUMMARY_TEMP_PATTERN,
        submit_done = SUBMITTED_SENTINEL_PATTERN,
    output:
        sentinel = ARCHIVE_SENTINEL_PATTERN,
    params:
        runs_dir = lambda w: f"{SCORES_DIR}/online/runs/level{w.level}",
        train_end = WIN["train_end"],
        lookback_months = CAND["lookback_months"],
        score_threshold = SEL["score_threshold"],
        top_k_per_buyer = SEL["top_k_per_buyer"],
        min_orders = GRD["min_orders"],
        min_months = GRD["min_months"],
        high_spend = GRD["high_spend"],
        min_avg_monthly_spend = GRD.get("min_avg_monthly_spend", 0),
        cold_start_top_k = config["submission"]["cold_start_top_k"],
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

rule submit_to_portal:
    """Upload online submission to Unite evaluator (challenge 2) per approach and level. Writes live score to temp file for archive_score_run. Requires portal_credentials in config.yaml."""
    input:
        submission = f"{DATA_DIR}/14_submission/online/{{approach}}/level{{level}}/submission.csv",
    output:
        summary = LIVE_SUMMARY_TEMP_PATTERN,
        sentinel = SUBMITTED_SENTINEL_PATTERN,
    wildcard_constraints:
        approach = APPROACH_RE,
        level = LEVEL_RE,
    shell:
        "uv run src/submit.py --challenge 2 --file {input.submission} --level {wildcards.level} "
        "--summary-csv {output.summary} && touch {output.sentinel}"
