"""
Snakemake workflow for Core Demand Challenge.

All input/output paths are defined here. Python scripts receive paths only via
CLI arguments. Final deliverable: data/10_submission/submission.csv with header
legal_entity_id,cluster.
"""

configfile: "config.yaml"

# Centralized path configuration (single source of truth)
DATA_DIR = config["data_dir"]
INPUTS = config["inputs"]
PLIS_TRAINING_CSV = config["plis_training_csv"]
CUSTOMER_META_CSV = config["customer_meta_csv"]
OUTPUT_DIR = config["output_dir"]
SUBMISSION_CSV = config["submission_csv"]

SPLIT = config["training_validation_split"]
PLIS_TRAINING_SPLIT = SPLIT["plis_training"]
PLIS_TESTING_SPLIT = SPLIT["plis_testing"]

PLOTS = config["plots"]
PLOTS_DIR = PLOTS["dir"]
PLOT_FILES = PLOTS["files"]
PLOT_OUTPUTS = expand(f"{PLOTS_DIR}/{{f}}", f=PLOT_FILES)

rule all:
    input:
        SUBMISSION_CSV,
        CUSTOMER_META_CSV,
        PLOT_OUTPUTS,

rule split_plis_training_validation:
    """Split plis_training into training and testing: 50 random task=none customers; their rows with orderdate >= cutoff go to test, rest to training."""
    input:
        plis = PLIS_TRAINING_CSV,
        customer = CUSTOMER_META_CSV,
    output:
        train = PLIS_TRAINING_SPLIT,
        test = PLIS_TESTING_SPLIT,
    params:
        cutoff = SPLIT["cutoff_date"],
        n_customers = SPLIT["test_customers_count"],
        seed = SPLIT["random_seed"],
    shell:
        "uv run src/split_plis_training_validation.py --input {input.plis} --customer-meta {input.customer} "
        "--train {output.train} --test {output.test} --cutoff-date {params.cutoff} "
        "--test-customers-count {params.n_customers} --random-seed {params.seed}"

rule build_customer_meta:
    """Build customer metadata from plis_training (all unique customers, task from customer_test or none)."""
    input:
        plis = PLIS_TRAINING_CSV,
        customer_test = INPUTS["customer_test"],
    output:
        customer = CUSTOMER_META_CSV,
    shell:
        "uv run src/build_customer_meta.py --plis {input.plis} --customer-test {input.customer_test} --output {output.customer}"

rule write_submission:
    """Write baseline submission CSV with required header (legal_entity_id,cluster)."""
    input:
        customer_test = INPUTS["customer_test"],
        plis = PLIS_TRAINING_CSV,
    output:
        submission = SUBMISSION_CSV,
    shell:
        "uv run src/write_submission.py --output {output.submission} "
        "--customer-test {input.customer_test} --plis-training {input.plis}"

rule generate_exploration_plots:
    """Generate EDA plots from raw CSVs and customer metadata into data/04_plots."""
    input:
        plis = PLIS_TRAINING_CSV,
        customer_test = INPUTS["customer_test"],
        les_cs = INPUTS["les_cs"],
        customer_meta = CUSTOMER_META_CSV,
    output:
        plots = PLOT_OUTPUTS,
    params:
        out_dir = PLOTS_DIR,
    shell:
        "uv run src/generate_exploration_plots.py --plis {input.plis} "
        "--customer-test {input.customer_test} --les-cs {input.les_cs} "
        "--customer-meta {input.customer_meta} --output-dir {params.out_dir}"
