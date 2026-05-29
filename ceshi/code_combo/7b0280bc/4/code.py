# Combo variant of 7b0280bc (index 4)
# combo_spec: D6/0+D7/0
# Name: 擦除裁剪提取路径子图

from __future__ import annotations

# Design analysis
# ---------------
# This combo is a direct composition of two already-separated behaviors on the same graph family:
# 1. `D6/0` removes every non-path station and non-path track while keeping the original canvas.
# 2. `D7/0` then crops the result to the tight bounding box of the surviving path pixels.
# The key point is that these two changes do not fight each other: D6 decides *what remains*,
# D7 decides *how much of the canvas to keep*. Using the D6 generator is sufficient because
# the input structure is unchanged; the combo only changes the output post-processing.

import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve()
PARENT = HERE.parent.parent
if str(PARENT) not in sys.path:
    sys.path.append(str(PARENT))

from _shared import load_variant


_D6 = load_variant("D6", 0)


def _tight_crop(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(_D6.get_bg(g))
    coords = np.argwhere(g != bg)
    assert coords.size > 0, "cropping requires at least one non-background pixel"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the unique shortest-path subgraph, erase everything else, then crop tightly.

    Input structure:
    - Same square-node + track graph vocabulary as the source task.
    - Terminals and stations are detected exactly as in the source-family variants.

    Rule:
    - Keep only the shortest-path nodes/tracks using the D6/0 erase rule.
    - Then crop the output to the minimal bounding box containing those remaining pixels.

    Key insight: This is the source task's path extraction made explicit in two stages:
    erase all off-path structure first, then remove all unused canvas around the extracted path.
    """
    erased = _D6.main(np.asarray(grid, dtype=int))
    return _tight_crop(erased)


def generate_input() -> np.ndarray:
    return np.asarray(_D6.generate_input(), dtype=int)
