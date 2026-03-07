"""
Two-stage LightGBM scorer for Level 1: EU = p_recur * v_hat * r - F.

Delegates to modelling runner with --approach lgbm_two_stage.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src on path when run as script
_src = Path(__file__).resolve().parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from modelling.run import run_main

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--approach", "lgbm_two_stage"] + sys.argv[1:]
    run_main()
