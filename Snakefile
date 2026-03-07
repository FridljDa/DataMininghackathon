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
SCORES_LGBM_OFFLINE_PARQUET = MODELLING["scores_lgbm_offline_parquet"]
PORTFOLIO_OFFLINE_PARQUET = MODELLING["portfolio_offline_parquet"]
SUBMISSION_OFFLINE_CSV = MODELLING["submission_offline_csv"]

FEATURE_ANALYSIS = config["feature_analysis"]
FEATURE_ANALYSIS_DIR = FEATURE_ANALYSIS["dir"]
FEATURE_ANALYSIS_SUMMARY_CSV = FEATURE_ANALYSIS["summary_csv"]
FEATURE_ANALYSIS_SUMMARY_OFFLINE_CSV = FEATURE_ANALYSIS["summary_offline_csv"]
FEATURE_ANALYSIS_PLOTS = expand(f"{FEATURE_ANALYSIS_DIR}/{{f}}", f=FEATURE_ANALYSIS["plot_files"])
FEATURE_ANALYSIS_PLOTS_OFFLINE = expand(f"{FEATURE_ANALYSIS_DIR}/{{f}}", f=FEATURE_ANALYSIS["plot_files_offline"])

# Mode (online|offline) configuration: paths and behaviour for modelling/scoring pipeline
MODES = ["online", "offline"]
MODE_CFG = {
    "online": {
        "customer_csv": CUSTOMER_META_CSV,
        "buyer_source": "customer-test",
        "customer_input": INPUTS["customer_test"],
        "score_level": "",
        "candidates_raw": MODELLING["candidates_raw_parquet"],
        "features_all": MODELLING["features_all_parquet"],
        "features_selected": MODELLING["features_selected_parquet"],
        "scores_parquet": MODELLING["scores_parquet"],
        "scores_lgbm_parquet": MODELLING["scores_lgbm_parquet"],
        "portfolio_parquet": MODELLING["portfolio_parquet"],
        "submission_csv": SUBMISSION_CSV,
        "score_summary": SCORE_OUTPUTS["summary"],
        "score_details": SCORE_OUTPUTS["details"],
        "archive_sentinel": ARCHIVE_SENTINEL_ONLINE,
        "runs_dir": SCORE_RUN_ARCHIVE["online_runs_dir"],
        "run_index_csv": SCORE_RUN_ARCHIVE["run_index_online"],
        "feature_analysis_summary": FEATURE_ANALYSIS["summary_csv"],
        "feature_analysis_plot_subdir": "online",
    },
    "offline": {
        "customer_csv": SPLIT_CUSTOMER_CSV,
        "buyer_source": "customer-split",
        "customer_input": SPLIT_CUSTOMER_CSV,
        "score_level": "1",
        "candidates_raw": MODELLING["candidates_raw_offline_parquet"],
        "features_all": MODELLING["features_all_offline_parquet"],
        "features_selected": MODELLING["features_selected_offline_parquet"],
        "scores_parquet": MODELLING["scores_offline_parquet"],
        "scores_lgbm_parquet": MODELLING["scores_lgbm_offline_parquet"],
        "portfolio_parquet": MODELLING["portfolio_offline_parquet"],
        "submission_csv": MODELLING["submission_offline_csv"],
        "score_summary": SCORE_OUTPUTS["summary_offline"],
        "score_details": SCORE_OUTPUTS["details_offline"],
        "archive_sentinel": ARCHIVE_SENTINEL_OFFLINE,
        "runs_dir": SCORE_RUN_ARCHIVE["offline_runs_dir"],
        "run_index_csv": SCORE_RUN_ARCHIVE["run_index_offline"],
        "feature_analysis_summary": FEATURE_ANALYSIS["summary_offline_csv"],
        "feature_analysis_plot_subdir": "offline",
    },
}
# Path patterns for mode-wildcarded rules (must match config paths)
CANDIDATES_RAW_PATTERN = "data/07_features/{mode}/candidates_raw.parquet"
FEATURES_ALL_PATTERN = "data/07_features/{mode}/features_all.parquet"
FEATURES_SELECTED_PATTERN = "data/07_features/{mode}/features_selected.parquet"
SCORES_PATTERN = "data/09_predictions/{mode}/scores.parquet"
SCORES_LGBM_PATTERN = "data/09_predictions/{mode}/scores_lgbm.parquet"
PORTFOLIO_PATTERN = "data/10_portfolio/{mode}/portfolio.parquet"
SUBMISSION_PATTERN = "data/11_submission/{mode}/submission.csv"
SCORE_SUMMARY_PATTERN = "data/12_scores/{mode}/score_summary.csv"
SCORE_DETAILS_PATTERN = "data/12_scores/{mode}/score_details.parquet"
ARCHIVE_SENTINEL_PATTERN = "data/12_scores/{mode}/runs/.last_archived"
FEATURE_ANALYSIS_SUMMARY_PATTERN = "data/08_feature_analysis/{mode}/feature_summary.csv"

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
    """Candidate generation for Level 1 (warm buyers, E-Class): lookback window, n_orders(b,e,L) >= eta, s_lookback >= tau."""
    input:
        plis = PLIS_TRAINING_SPLIT,
        customer = lambda w: MODE_CFG[w.mode]["customer_csv"],
    output:
        candidates_raw = CANDIDATES_RAW_PATTERN,
    params:
        train_end = MODELLING["train_end"],
        lookback_months = MODELLING["lookback_months"],
        min_order_frequency = MODELLING["min_order_frequency"],
        min_lookback_spend = MODELLING["min_lookback_spend"],
    wildcard_constraints:
        mode = "|".join(MODES),
    shell:
        "uv run src/generate_candidates.py --plis {input.plis} --customer {input.customer} "
        "--output {output.candidates_raw} --train-end {params.train_end} "
        "--lookback-months {params.lookback_months} --min-order-frequency {params.min_order_frequency} "
        "--min-lookback-spend {params.min_lookback_spend}"

rule engineer_features:
    """Feature engineering from raw candidates: all modelling features."""
    input:
        candidates_raw = CANDIDATES_RAW_PATTERN,
        plis = PLIS_TRAINING_SPLIT,
        customer = lambda w: MODE_CFG[w.mode]["customer_csv"],
    output:
        features_all = FEATURES_ALL_PATTERN,
    params:
        train_end = MODELLING["train_end"],
    wildcard_constraints:
        mode = "|".join(MODES),
    shell:
        "uv run src/engineer_features.py --candidates-raw {input.candidates_raw} --plis {input.plis} "
        "--customer {input.customer} --output {output.features_all} --train-end {params.train_end}"

rule feature_analysis:
    """Summary statistics and informativeness plots for all engineered features."""
    input:
        features_all = FEATURES_ALL_PATTERN,
        plis = PLIS_TRAINING_SPLIT,
    output:
        summary_csv = FEATURE_ANALYSIS_SUMMARY_PATTERN,
        distributions_plot = "data/08_feature_analysis/{mode}/feature_distributions.png",
        correlations_plot = "data/08_feature_analysis/{mode}/feature_correlations.png",
        value_by_period_plot = "data/08_feature_analysis/{mode}/purchase_value_by_period.png",
        quantity_by_period_plot = "data/08_feature_analysis/{mode}/purchase_quantity_by_period.png",
    wildcard_constraints:
        mode = "|".join(MODES),
    shell:
        "uv run src/feature_analysis.py --features {input.features_all} --summary-csv {output.summary_csv} "
        "--distributions-plot {output.distributions_plot} --correlations-plot {output.correlations_plot} "
        "--plis {input.plis} --value-by-period-plot {output.value_by_period_plot} "
        "--quantity-by-period-plot {output.quantity_by_period_plot}"

rule feature_selection:
    """Keep keys + config-driven selected features for downstream modelling."""
    input:
        features_all = FEATURES_ALL_PATTERN,
    output:
        features_selected = FEATURES_SELECTED_PATTERN,
    params:
        selected_features = ",".join(MODELLING["selected_features"]),
    wildcard_constraints:
        mode = "|".join(MODES),
    shell:
        "uv run src/feature_selection.py --features {input.features_all} --selected-features '{params.selected_features}' "
        "--output {output.features_selected}"

rule train_baseline:
    """Baseline scorer and validation labels; outputs scores.parquet."""
    input:
        candidates = FEATURES_SELECTED_PATTERN,
        plis = PLIS_TRAINING_SPLIT,
    output:
        scores = SCORES_PATTERN,
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
    wildcard_constraints:
        mode = "|".join(MODES),
    shell:
        "uv run src/train_baseline.py --candidates {input.candidates} --plis {input.plis} "
        "--output {output.scores} --val-start {params.val_start} --val-end {params.val_end} "
        "--n-min-label {params.n_min_label} --alpha {params.alpha} --beta {params.beta} --gamma {params.gamma} "
        "--savings-rate {params.savings_rate} --fixed-fee-eur {params.fixed_fee_eur} --val-months {params.val_months}"

rule train_lgbm:
    """Two-stage LightGBM (recurrence classifier + value regressor); outputs scores_lgbm.parquet."""
    input:
        candidates = FEATURES_SELECTED_PATTERN,
        plis = PLIS_TRAINING_SPLIT,
    output:
        scores = SCORES_LGBM_PATTERN,
    params:
        val_start = MODELLING["val_start"],
        val_end = MODELLING["val_end"],
        n_min_label = MODELLING["n_min_label"],
        savings_rate = SCORING_PARAMS["savings_rate"],
        fixed_fee_eur = SCORING_PARAMS["fixed_fee_eur"],
        val_months = MODELLING["val_months"],
    wildcard_constraints:
        mode = "|".join(MODES),
    shell:
        "uv run src/train_lgbm.py --candidates {input.candidates} --plis {input.plis} "
        "--output {output.scores} --val-start {params.val_start} --val-end {params.val_end} "
        "--n-min-label {params.n_min_label} "
        "--savings-rate {params.savings_rate} --fixed-fee-eur {params.fixed_fee_eur} --val-months {params.val_months}"

rule select_portfolio:
    """Apply EU threshold, guardrails and per-buyer cap K to produce portfolio.parquet (default: two-stage LGBM scores)."""
    input:
        scores = SCORES_LGBM_PATTERN,
    output:
        portfolio = PORTFOLIO_PATTERN,
    params:
        score_threshold = MODELLING["score_threshold"],
        min_orders_guardrail = MODELLING["min_orders_guardrail"],
        min_months_guardrail = MODELLING["min_months_guardrail"],
        high_spend_guardrail = MODELLING["high_spend_guardrail"],
        top_k_per_buyer = MODELLING["top_k_per_buyer"],
    wildcard_constraints:
        mode = "|".join(MODES),
    shell:
        "uv run src/select_portfolio.py --scores {input.scores} --output {output.portfolio} "
        "--score-threshold {params.score_threshold} --min-orders-guardrail {params.min_orders_guardrail} "
        "--min-months-guardrail {params.min_months_guardrail} --high-spend-guardrail {params.high_spend_guardrail} "
        "--top-k-per-buyer {params.top_k_per_buyer}"

rule write_submission_warm:
    """Format portfolio into submission CSV (Level 1: cluster=eclass); cold-start fallback to global default."""
    input:
        portfolio = PORTFOLIO_PATTERN,
        customer = lambda w: MODE_CFG[w.mode]["customer_input"],
        plis = PLIS_TRAINING_SPLIT,
    output:
        submission = SUBMISSION_PATTERN,
    params:
        buyer_source = lambda w: MODE_CFG[w.mode]["buyer_source"],
    wildcard_constraints:
        mode = "|".join(MODES),
    run:
        arg = "--customer-test" if wildcards.mode == "online" else "--customer-split"
        shell(
            "uv run src/write_submission_warm.py --portfolio {input.portfolio} "
            "--buyer-source {params.buyer_source} " + arg + " {input.customer} --plis-training {input.plis} --output {output.submission}"
        )


rule select_portfolio_baseline:
    """Optional: apply threshold/guardrails to baseline scores. Use for diagnostic comparison."""
    input:
        scores = MODELLING["scores_parquet"],
    output:
        portfolio = "data/10_portfolio/online/portfolio_baseline.parquet",
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
        submission = SUBMISSION_PATTERN,
        plis_testing = PLIS_TESTING_SPLIT,
    output:
        summary = SCORE_SUMMARY_PATTERN,
        details = SCORE_DETAILS_PATTERN,
    params:
        savings_rate = SCORING_PARAMS["savings_rate"],
        fixed_fee_eur = SCORING_PARAMS["fixed_fee_eur"],
        scoring_months = SCORING_PARAMS["scoring_months"],
    wildcard_constraints:
        mode = "|".join(MODES),
    run:
        level_opt = " --level 1" if wildcards.mode == "offline" else ""
        shell(
            "uv run src/score_submission.py --submission {input.submission} --plis-testing {input.plis_testing} "
            "--summary {output.summary} --details {output.details}"
            + level_opt + " "
            "--savings-rate {params.savings_rate} --fixed-fee-eur {params.fixed_fee_eur} "
            "--scoring-months {params.scoring_months}"
        )

rule archive_score_run:
    """Copy current score outputs into a timestamp+commit run folder and write metadata + run index."""
    input:
        summary = SCORE_SUMMARY_PATTERN,
        details = SCORE_DETAILS_PATTERN,
    output:
        sentinel = ARCHIVE_SENTINEL_PATTERN,
    params:
        runs_dir = lambda w: MODE_CFG[w.mode]["runs_dir"],
        index_csv = lambda w: MODE_CFG[w.mode]["run_index_csv"],
    wildcard_constraints:
        mode = "|".join(MODES),
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

