# Combo variant of 7b0280bc (index 2)
# combo_spec: D4/2+D5/2
# Name: 内嵌标记颜色匹配寻路

from __future__ import annotations

# Design analysis
# ---------------
# `D5/2` already subsumes the `D4/2` change:
# - terminals are identified by embedded interior marker pixels,
# - the actual destination is selected by matching the start marker color.
# So this confirmed combo is behaviorally identical to the existing D5/2 single variant.

import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve()
PARENT = HERE.parent.parent
if str(PARENT) not in sys.path:
    sys.path.append(str(PARENT))

from _shared import load_variant


_D5 = load_variant("D5", 2)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Solve the embedded-marker color-matching path task.

    Input structure:
    - Same as D5/2: embedded marker pixels identify terminals and marker equality selects the goal.

    Rule:
    - Read the start marker color.
    - Select the unique goal candidate with the same marker color.
    - Recolor the shortest path to that goal.

    Key insight: terminal identity is hidden inside node interiors, and the destination is
    chosen by marker-color matching rather than by node body color.
    """
    return np.asarray(_D5.main(np.asarray(grid, dtype=int)), dtype=int)


def generate_input() -> np.ndarray:
    return np.asarray(_D5.generate_input(), dtype=int)
