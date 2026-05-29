# Combo variant of 7b0280bc (index 19)
# combo_spec: D10/2+D14/1
# Name: 紧凑空心方块密集图

from __future__ import annotations

# Design analysis
# ---------------
# The solver-side change is still exactly `D10/2`: nodes are hollow square frames, not solid blocks.
# `D14/1` only makes the layout denser by shrinking cell gaps to the minimum. That affects generation,
# not graph semantics. So:
# - `main()` can reuse `D10/2` directly.
# - `generate_input()` can reuse the same generator with its gap parameter forced to `1`.

import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve()
PARENT = HERE.parent.parent
if str(PARENT) not in sys.path:
    sys.path.append(str(PARENT))

from _shared import load_variant


_D10 = load_variant("D10", 2)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the shortest path in a dense graph of hollow-frame nodes.

    Input structure:
    - Same as D10/2, except node placement is packed with only 1-cell inter-node gaps.

    Rule:
    - Hollow frames are still the graph nodes.
    - The shortest terminal-to-terminal route is recolored exactly as in D10/2.

    Key insight: the geometric challenge is denser spacing, not a new graph rule.
    """
    return np.asarray(_D10.main(np.asarray(grid, dtype=int)), dtype=int)


def generate_input() -> np.ndarray:
    original = list(_D10.GAP_CHOICES)
    original_retries = int(_D10.GEN_ATTEMPT_RETRIES)
    try:
        _D10.GAP_CHOICES = [1]
        # Dense hollow-frame layouts are much rarer than the source task's default spacing.
        # Keep the exact D10/2 semantics, but give the generator more attempts to hit a legal
        # packed sample instead of spuriously failing.
        _D10.GEN_ATTEMPT_RETRIES = max(original_retries, 10000)
        return np.asarray(_D10.generate_input(), dtype=int)
    finally:
        _D10.GAP_CHOICES = original
        _D10.GEN_ATTEMPT_RETRIES = original_retries
