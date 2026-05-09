# Combo variant of 7b0280bc (index 5)
# combo_spec: D8/2+D13/0
# Name: 多对多终端路径合并

from __future__ import annotations

# Design analysis
# ---------------
# This combo is semantically the same as the existing D13/0 single variant:
# - four terminals split into two same-colored pairs,
# - compute each pair's shortest path independently,
# - recolor the union of the two paths.
# `D8/2` contributes the multi-path-union reasoning, while `D13/0` contributes the
# concrete two-pair terminal structure. Since the current D13/0 implementation already
# realizes exactly that merged-path behavior and provides a stable generator, the combo
# can safely reuse it directly.

import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve()
PARENT = HERE.parent.parent
if str(PARENT) not in sys.path:
    sys.path.append(str(PARENT))

from _shared import load_variant


_D13 = load_variant("D13", 0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the union of the two shortest paths defined by the two terminal color pairs.

    Input structure:
    - Same as D13/0: four terminals in two same-colored pairs, plus station squares and tracks.

    Rule:
    - Pair terminals by color equality.
    - Solve each pair independently.
    - Recolor the union of both shortest paths.

    Key insight: The output is the merge of two shortest-path solutions on the same graph.
    """
    return np.asarray(_D13.main(np.asarray(grid, dtype=int)), dtype=int)


def generate_input() -> np.ndarray:
    return np.asarray(_D13.generate_input(), dtype=int)
