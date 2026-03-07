"""
Snakemake workflow for Core Demand Challenge.

All input/output paths are defined here. Python scripts receive paths only via
CLI arguments. Final deliverable: data/11_submission/online/submission.csv with header
legal_entity_id,cluster.
"""

configfile: "config.yaml"

# Centralized path configuration (single source of truth)
DATA_DIR = config["data_dir"]
DAG_SVG = config["dag_svg"]
INPUTS = config["inputs"]
PLIS_TRAINING_CSV = config["plis_training_csv"]
CUSTOMER_META_CSV = config["customer_meta_csv"]
OUTPUT_DIR = config["output_dir"]
SUBMISSION_CSV = config["submission_csv"]

SPLIT = config["training_validation_split"]
PLIS_TRAINING_SPLIT = SPLIT["plis_training"]
PLIS_TESTING_SPLIT = SPLIT["plis_testing"]
SPLIT_CUSTOMER_CSV = SPLIT["customer_csv"]

SCORE_OUTPUTS = config["score_outputs"]
SCORE_SUMMARY = SCORE_OUTPUTS["summary"]
SCORE_DETAILS = SCORE_OUTPUTS["details"]
SCORE_SUMMARY_OFFLINE = SCORE_OUTPUTS["summary_offline"]
SCORE_DETAILS_OFFLINE = SCORE_OUTPUTS["details_offline"]
SCORE_RUN_ARCHIVE = config["score_run_archive"]
ARCHIVE_SENTINEL_ONLINE = SCORE_RUN_ARCHIVE["online_runs_dir"] + "/.last_archived"
ARCHIVE_SENTINEL_OFFLINE = SCORE_RUN_ARCHIVE["offline_runs_dir"] + "/.last_archived"
SCORING_PARAMS = config["scoring_parameters"]

PLOTS = config["plots"]
PLOTS_DIR = PLOTS["dir"]
PLOT_FILES = PLOTS["files"]
PLOT_OUTPUTS = expand(f"{PLOTS_DIR}/{{f}}", f=PLOT_FILES)

MODELLING = config["modelling"]
CANDIDATES_RAW_PARQUET = MODELLING["candidates_raw_parquet"]
FEATURES_ALL_PARQUET = MODELLING["features_all_parquet"]
FEATURES_SELECTED_PARQUET = MODELLING["features_selected_parquet"]
SCORES_PARQUET = MODELLING["scores_parquet"]
SCORES_LGBM_PARQUET = MODELLING["scores_lgbm_parquet"]
PORTFOLIO_PARQUET = MODELLING["portfolio_parquet"]
CANDIDATES_RAW_OFFLINE_PARQUET = MODELLING["candidates_raw_offline_parquet"]
FEATURES_ALL_OFFLINE_PARQUET = MODELLING["features_all_offline_parquet"]
FEATURES_SELECTED_OFFLINE_PARQUET = MODELLING["features_selected_offline_parquet"]
SCORES_OFFLINE_PARQUET = MODELLING["scores_offline_parquet"]
PORTFOLIO_OFFLINE_PARQUET = MODELLING["portfolio_offline_parquet"]
SUBMISSION_OFFLINE_CSV = MODELLING["submission_offline_csv"]

FEATURE_ANALYSIS = config["feature_analysis"]
FEATURE_ANALYSIS_DIR = FEATURE_ANALYSIS["dir"]
FEATURE_ANALYSIS_SUMMARY_CSV = FEATURE_ANALYSIS["summary_csv"]
FEATURE_ANALYSIS_SUMMARY_OFFLINE_CSV = FEATURE_ANALYSIS["summary_offline_csv"]
FEATURE_ANALYSIS_PLOTS = expand(f"{FEATURE_ANALYSIS_DIR}/{{f}}", f=FEATURE_ANALYSIS["plot_files"])
FEATURE_ANALYSIS_PLOTS_OFFLINE = expand(f"{FEATURE_ANALYSIS_DIR}/{{f}}", f=FEATURE_ANALYSIS["plot_files_offline"])

rule all:
    input:
        DAG_SVG,
        SUBMISSION_CSV,
        CUSTOMER_META_CSV,
        PLOT_OUTPUTS,
        FEATURE_ANALYSIS_SUMMARY_CSV,
        FEATURE_ANALYSIS_SUMMARY_OFFLINE_CSV,
        FEATURE_ANALYSIS_PLOTS,
        FEATURE_ANALYSIS_PLOTS_OFFLINE,
        SCORE_SUMMARY,
        SCORE_DETAILS,
        ARCHIVE_SENTINEL_ONLINE,
        "data/12_scores/online/score_summary_live.csv",
        "data/11_submission/.submitted_challenge2",

rule generate_dag_graph:
    """Write workflow DAG as SVG (no input dependencies; run first)."""
    output:
        dag = DAG_SVG,
    shell:
        "mkdir -p $(dirname {output.dag}) && snakemake --dag | dot -Tsvg -o {output.dag}"

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
    """Candidate generation for Level 1 (warm buyers, E-Class): lookback + singleton filter."""
    input:
        plis = PLIS_TRAINING_SPLIT,
        customer = CUSTOMER_META_CSV,
    output:
        candidates_raw = CANDIDATES_RAW_PARQUET,
    params:
        train_end = MODELLING["train_end"],
        lookback_months = MODELLING["lookback_months"],
        min_spend_singleton = MODELLING["min_spend_singleton"],
    shell:
        "uv run src/generate_candidates.py --plis {input.plis} --customer {input.customer} "
        "--output {output.candidates_raw} --train-end {params.train_end} "
        "--lookback-months {params.lookback_months} --min-spend-singleton {params.min_spend_singleton}"

rule engineer_features:
    """Feature engineering from raw candidates: all modelling features."""
    input:
        candidates_raw = CANDIDATES_RAW_PARQUET,
        plis = PLIS_TRAINING_SPLIT,
        customer = CUSTOMER_META_CSV,
    output:
        features_all = FEATURES_ALL_PARQUET,
    params:
        train_end = MODELLING["train_end"],
    shell:
        "uv run src/engineer_features.py --candidates-raw {input.candidates_raw} --plis {input.plis} "
        "--customer {input.customer} --output {output.features_all} --train-end {params.train_end}"

rule feature_analysis:
    """Summary statistics and informativeness plots for all engineered features."""
    input:
        features_all = FEATURES_ALL_PARQUET,
    output:
        summary_csv = FEATURE_ANALYSIS_SUMMARY_CSV,
        distributions_plot = f"{FEATURE_ANALYSIS_DIR}/online/feature_distributions.png",
        correlations_plot = f"{FEATURE_ANALYSIS_DIR}/online/feature_correlations.png",
    shell:
        "uv run src/feature_analysis.py --features {input.features_all} --summary-csv {output.summary_csv} "
        "--distributions-plot {output.distributions_plot} --correlations-plot {output.correlations_plot}"

rule feature_selection:
    """Keep keys + config-driven selected features for downstream modelling."""
    input:
        features_all = FEATURES_ALL_PARQUET,
    output:
        features_selected = FEATURES_SELECTED_PARQUET,
    params:
        selected_features = ",".join(MODELLING["selected_features"]),
    shell:
        "uv run src/feature_selection.py --features {input.features_all} --selected-features '{params.selected_features}' "
        "--output {output.features_selected}"

rule train_baseline:
    """Baseline scorer and validation labels; outputs scores.parquet."""
    input:
        candidates = FEATURES_SELECTED_PARQUET,
        plis = PLIS_TRAINING_SPLIT,
    output:
        scores = SCORES_PARQUET,
    params:
        val_start = MODELLING["val_start"],
        val_end = MODELLING["val_end"],
        n_min_label = MODELLING["n_min_label"],
        alpha = MODELLING["alpha"],
        beta = MODELLING["beta"],
        gamma = MODELLING["gamma"],
        savings_rate = SCORING_PARAMS["savings_rate"],
        fixed_fee_eur = SCORING_PARAMS["fixed_fee_eur"],
        val_months = MODELLING["val_months"],
    shell:
        "uv run src/train_baseline.py --candidates {input.candidates} --plis {input.plis} "
        "--output {output.scores} --val-start {params.val_start} --val-end {params.val_end} "
        "--n-min-label {params.n_min_label} --alpha {params.alpha} --beta {params.beta} --gamma {params.gamma} "
        "--savings-rate {params.savings_rate} --fixed-fee-eur {params.fixed_fee_eur} --val-months {params.val_months}"

rule train_lgbm:
    """Two-stage LightGBM (recurrence classifier + value regressor); outputs scores_lgbm.parquet."""
    input:
        candidates = FEATURES_SELECTED_PARQUET,
        plis = PLIS_TRAINING_SPLIT,
    output:
        scores = SCORES_LGBM_PARQUET,
    params:
        val_start = MODELLING["val_start"],
        val_end = MODELLING["val_end"],
        n_min_label = MODELLING["n_min_label"],
        savings_rate = SCORING_PARAMS["savings_rate"],
        fixed_fee_eur = SCORING_PARAMS["fixed_fee_eur"],
        val_months = MODELLING["val_months"],
    shell:
        "uv run src/train_lgbm.py --candidates {input.candidates} --plis {input.plis} "
        "--output {output.scores} --val-start {params.val_start} --val-end {params.val_end} "
        "--n-min-label {params.n_min_label} "
        "--savings-rate {params.savings_rate} --fixed-fee-eur {params.fixed_fee_eur} --val-months {params.val_months}"

rule select_portfolio:
    """Apply threshold, guardrails and per-buyer cap K to produce portfolio.parquet."""
    input:
        scores = SCORES_PARQUET,
    output:
        portfolio = PORTFOLIO_PARQUET,
    params:
        score_threshold = MODELLING["score_threshold"],
        min_orders_guardrail = MODELLING["min_orders_guardrail"],
        min_months_guardrail = MODELLING["min_months_guardrail"],
        high_spend_guardrail = MODELLING["high_spend_guardrail"],
        top_k_per_buyer = MODELLING["top_k_per_buyer"],
    shell:
        "uv run src/select_portfolio.py --scores {input.scores} --output {output.portfolio} "
        "--score-threshold {params.score_threshold} --min-orders-guardrail {params.min_orders_guardrail} "
        "--min-months-guardrail {params.min_months_guardrail} --high-spend-guardrail {params.high_spend_guardrail} "
        "--top-k-per-buyer {params.top_k_per_buyer}"

rule write_submission_warm:
    """Format portfolio into submission CSV (Level 1: cluster=eclass); cold-start fallback to global default."""
    input:
        portfolio = PORTFOLIO_PARQUET,
        customer_test = INPUTS["customer_test"],
        plis = PLIS_TRAINING_SPLIT,
    output:
        submission = SUBMISSION_CSV,
    shell:
        "uv run src/write_submission_warm.py --portfolio {input.portfolio} "
        "--buyer-source customer-test --customer-test {input.customer_test} --plis-training {input.plis} --output {output.submission}"


rule write_submission:
    """Write baseline submission CSV with required header (legal_entity_id,cluster). Use 'snakemake data/11_submission/submission_baseline.csv' to run."""
    input:
        customer_test = INPUTS["customer_test"],
        plis = PLIS_TRAINING_CSV,
    output:
        submission = "data/11_submission/submission_baseline.csv",
    shell:
        "uv run src/write_submission.py --output {output.submission} "
        "--customer-test {input.customer_test} --plis-training {input.plis}"

rule score_submission:
    """Score submission against plis_testing holdout; write summary and details to data/12_scores."""
    input:
        submission = SUBMISSION_CSV,
        plis_testing = PLIS_TESTING_SPLIT,
    output:
        summary = SCORE_SUMMARY,
        details = SCORE_DETAILS,
    params:
        savings_rate = SCORING_PARAMS["savings_rate"],
        fixed_fee_eur = SCORING_PARAMS["fixed_fee_eur"],
        scoring_months = SCORING_PARAMS["scoring_months"],
    shell:
        "uv run src/score_submission.py --submission {input.submission} --plis-testing {input.plis_testing} "
        "--summary {output.summary} --details {output.details} "
        "--savings-rate {params.savings_rate} --fixed-fee-eur {params.fixed_fee_eur} "
        "--scoring-months {params.scoring_months}"

rule archive_score_run:
    """Copy current online score outputs into a timestamp+commit run folder and write metadata + run index."""
    input:
        summary = SCORE_SUMMARY,
        details = SCORE_DETAILS,
    output:
        sentinel = ARCHIVE_SENTINEL_ONLINE,
    params:
        runs_dir = SCORE_RUN_ARCHIVE["online_runs_dir"],
        index_csv = SCORE_RUN_ARCHIVE["run_index_online"],
    shell:
        "uv run src/archive_score_run.py --summary {input.summary} --details {input.details} "
        "--runs-dir {params.runs_dir} --index-csv {params.index_csv} && touch {output.sentinel}"

rule submit_to_portal:
    """Upload submission to Unite evaluator (challenge 2). Requires portal_credentials in config.yaml."""
    input:
        submission = SUBMISSION_CSV,
    output:
        summary = "data/12_scores/online/score_summary_live.csv",
        sentinel = "data/11_submission/.submitted_challenge2",
    shell:
        "uv run src/submit.py --challenge 2 --file {input.submission} --summary-csv {output.summary} && touch {output.sentinel}"

# --- Offline scoring: predict for testing buyers only, score with Level-1 (eclass) matching ---
rule generate_candidates_offline:
    """Candidate generation for offline pipeline (task=testing buyers included)."""
    input:
        plis = PLIS_TRAINING_SPLIT,
        customer = SPLIT_CUSTOMER_CSV,
    output:
        candidates_raw = CANDIDATES_RAW_OFFLINE_PARQUET,
    params:
        train_end = MODELLING["train_end"],
        lookback_months = MODELLING["lookback_months"],
        min_spend_singleton = MODELLING["min_spend_singleton"],
    shell:
        "uv run src/generate_candidates.py --plis {input.plis} --customer {input.customer} "
        "--output {output.candidates_raw} --train-end {params.train_end} "
        "--lookback-months {params.lookback_months} --min-spend-singleton {params.min_spend_singleton}"

rule engineer_features_offline:
    """Feature engineering for offline pipeline."""
    input:
        candidates_raw = CANDIDATES_RAW_OFFLINE_PARQUET,
        plis = PLIS_TRAINING_SPLIT,
        customer = SPLIT_CUSTOMER_CSV,
    output:
        features_all = FEATURES_ALL_OFFLINE_PARQUET,
    params:
        train_end = MODELLING["train_end"],
    shell:
        "uv run src/engineer_features.py --candidates-raw {input.candidates_raw} --plis {input.plis} "
        "--customer {input.customer} --output {output.features_all} --train-end {params.train_end}"

rule feature_analysis_offline:
    """Feature analysis for offline feature set."""
    input:
        features_all = FEATURES_ALL_OFFLINE_PARQUET,
    output:
        summary_csv = FEATURE_ANALYSIS_SUMMARY_OFFLINE_CSV,
        distributions_plot = f"{FEATURE_ANALYSIS_DIR}/offline/feature_distributions.png",
        correlations_plot = f"{FEATURE_ANALYSIS_DIR}/offline/feature_correlations.png",
    shell:
        "uv run src/feature_analysis.py --features {input.features_all} --summary-csv {output.summary_csv} "
        "--distributions-plot {output.distributions_plot} --correlations-plot {output.correlations_plot}"

rule feature_selection_offline:
    """Config-driven feature selection for offline pipeline."""
    input:
        features_all = FEATURES_ALL_OFFLINE_PARQUET,
    output:
        features_selected = FEATURES_SELECTED_OFFLINE_PARQUET,
    params:
        selected_features = ",".join(MODELLING["selected_features"]),
    shell:
        "uv run src/feature_selection.py --features {input.features_all} --selected-features '{params.selected_features}' "
        "--output {output.features_selected}"

rule train_baseline_offline:
    """Baseline scorer for offline pipeline (same as train_baseline, different input/output)."""
    input:
        candidates = FEATURES_SELECTED_OFFLINE_PARQUET,
        plis = PLIS_TRAINING_SPLIT,
    output:
        scores = SCORES_OFFLINE_PARQUET,
    params:
        val_start = MODELLING["val_start"],
        val_end = MODELLING["val_end"],
        n_min_label = MODELLING["n_min_label"],
        alpha = MODELLING["alpha"],
        beta = MODELLING["beta"],
        gamma = MODELLING["gamma"],
        savings_rate = SCORING_PARAMS["savings_rate"],
        fixed_fee_eur = SCORING_PARAMS["fixed_fee_eur"],
        val_months = MODELLING["val_months"],
    shell:
        "uv run src/train_baseline.py --candidates {input.candidates} --plis {input.plis} "
        "--output {output.scores} --val-start {params.val_start} --val-end {params.val_end} "
        "--n-min-label {params.n_min_label} --alpha {params.alpha} --beta {params.beta} --gamma {params.gamma} "
        "--savings-rate {params.savings_rate} --fixed-fee-eur {params.fixed_fee_eur} --val-months {params.val_months}"

rule select_portfolio_offline:
    """Select portfolio for offline pipeline."""
    input:
        scores = SCORES_OFFLINE_PARQUET,
    output:
        portfolio = PORTFOLIO_OFFLINE_PARQUET,
    params:
        score_threshold = MODELLING["score_threshold"],
        min_orders_guardrail = MODELLING["min_orders_guardrail"],
        min_months_guardrail = MODELLING["min_months_guardrail"],
        high_spend_guardrail = MODELLING["high_spend_guardrail"],
        top_k_per_buyer = MODELLING["top_k_per_buyer"],
    shell:
        "uv run src/select_portfolio.py --scores {input.scores} --output {output.portfolio} "
        "--score-threshold {params.score_threshold} --min-orders-guardrail {params.min_orders_guardrail} "
        "--min-months-guardrail {params.min_months_guardrail} --high-spend-guardrail {params.high_spend_guardrail} "
        "--top-k-per-buyer {params.top_k_per_buyer}"

rule write_submission_offline:
    """Build submission from portfolio for testing buyers only (offline scoring)."""
    input:
        portfolio = PORTFOLIO_OFFLINE_PARQUET,
        customer_split = SPLIT_CUSTOMER_CSV,
        plis = PLIS_TRAINING_SPLIT,
    output:
        submission = SUBMISSION_OFFLINE_CSV,
    shell:
        "uv run src/write_submission_warm.py --portfolio {input.portfolio} "
        "--buyer-source customer-split --customer-split {input.customer_split} --plis-training {input.plis} --output {output.submission}"

rule score_submission_offline:
    """Score offline submission against plis_testing with Level-1 (eclass-only) matching."""
    input:
        submission = SUBMISSION_OFFLINE_CSV,
        plis_testing = PLIS_TESTING_SPLIT,
    output:
        summary = SCORE_SUMMARY_OFFLINE,
        details = SCORE_DETAILS_OFFLINE,
    params:
        savings_rate = SCORING_PARAMS["savings_rate"],
        fixed_fee_eur = SCORING_PARAMS["fixed_fee_eur"],
        scoring_months = SCORING_PARAMS["scoring_months"],
    shell:
        "uv run src/score_submission.py --submission {input.submission} --plis-testing {input.plis_testing} "
        "--summary {output.summary} --details {output.details} --level 1 "
        "--savings-rate {params.savings_rate} --fixed-fee-eur {params.fixed_fee_eur} "
        "--scoring-months {params.scoring_months}"

rule archive_score_run_offline:
    """Copy current offline score outputs into a timestamp+commit run folder and write metadata + run index."""
    input:
        summary = SCORE_SUMMARY_OFFLINE,
        details = SCORE_DETAILS_OFFLINE,
    output:
        sentinel = ARCHIVE_SENTINEL_OFFLINE,
    params:
        runs_dir = SCORE_RUN_ARCHIVE["offline_runs_dir"],
        index_csv = SCORE_RUN_ARCHIVE["run_index_offline"],
    shell:
        "uv run src/archive_score_run.py --summary {input.summary} --details {input.details} "
        "--runs-dir {params.runs_dir} --index-csv {params.index_csv} && touch {output.sentinel}"
