"""
Snakemake workflow for Core Demand Challenge.

All input/output paths are defined here. Python scripts receive paths only via
CLI arguments. Final deliverable: outputs/submission.csv with header
buyer_id,predicted_id.
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

rule all:
    input:
        SUBMISSION_CSV,
        CUSTOMER_META_CSV,

rule split_plis_training_validation:
    """Split plis_training into training and testing by cutoff date; test holds a configurable fraction of rows >= cutoff."""
    input:
        plis = PLIS_TRAINING_CSV,
    output:
        train = PLIS_TRAINING_SPLIT,
        test = PLIS_TESTING_SPLIT,
    params:
        cutoff = SPLIT["cutoff_date"],
        fraction = SPLIT["test_fraction"],
        seed = SPLIT["random_seed"],
    shell:
        "uv run src/split_plis_training_validation.py --input {input.plis} --train {output.train} --test {output.test} "
        "--cutoff-date {params.cutoff} --test-fraction {params.fraction} --random-seed {params.seed}"

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
    """Write minimal submission CSV with required header (buyer_id,predicted_id)."""
    output:
        submission = SUBMISSION_CSV,
    shell:
        "uv run src/write_submission.py --output {output.submission}"
