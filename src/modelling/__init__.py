"""
Modelling package: approach-agnostic runner and per-approach scorers.

Entrypoint: run.run_main() (or `python -m modelling.run`) with --approach baseline | lgbm_two_stage | pass_through.
"""

from modelling.run import run_main

__all__ = ["run_main"]
