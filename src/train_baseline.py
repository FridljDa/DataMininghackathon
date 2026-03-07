"""
Baseline scorer for Level 1: score_base(b,e) = α·m_active + β·√historical_purchase_value − γ·δ_recency.

Delegates to modelling runner with --approach baseline.
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
    # Inject --approach baseline before argv so run_main() sees it
    sys.argv = [sys.argv[0], "--approach", "baseline"] + sys.argv[1:]
    run_main()
