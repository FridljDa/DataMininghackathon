"""
Snakemake workflow for Core Demand Challenge.

All input/output paths are defined here. Python scripts receive paths only via
CLI arguments. Final deliverable: outputs/submission.csv with header
buyer_id,predicted_id.
"""

# Centralized path configuration (single source of truth)
DATA_DIR = "data"
INPUTS = {
    "customer_test": DATA_DIR + "/customer_test.csv",
    "nace_codes": DATA_DIR + "/nace_codes.csv",
}
OUTPUT_DIR = "outputs"
SUBMISSION_CSV = OUTPUT_DIR + "/submission.csv"

rule all:
    input:
        SUBMISSION_CSV,

rule write_submission:
    """Write minimal submission CSV with required header (buyer_id,predicted_id)."""
    output:
        submission = SUBMISSION_CSV,
    shell:
        "uv run src/write_submission.py --output {output.submission}"
