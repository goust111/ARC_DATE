from __future__ import annotations

import numpy as np

from arc_utils import Color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every pink pixel red while preserving every orange pixel.

    Input structure:
    - The grid is a non-empty 2D array containing exactly pink and orange pixels.
    - Pink pixels are the source mask to rewrite; orange pixels are preserved.

    Rule:
    - Copy the grid.
    - Replace every pink pixel with red.
    - Leave every orange pixel unchanged.

    Key insight:
    - The changed cells are exactly the pink mask, so the task is a direct
      in-place color normalization.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "input must be a non-empty 2D grid"

    colors = set(map(int, np.unique(g)))
    assert colors == {Color.PINK, Color.ORANGE}, "input must contain exactly pink and orange"

    out = g.copy()
    out[g == Color.PINK] = Color.RED
    return out

# --------------- generator (o5x) ---------------
# Canvas bounds are high-exclusive for np.random.randint.
ROW_RANGE, COL_RANGE = (3, 7), (4, 7)
# Palette and per-cell source-color probability.
PINK, ORANGE, RED, PINK_RATE = Color.PINK, Color.ORANGE, Color.RED, 0.5
# Retry budget for rare monochrome samples.
MAX_ATTEMPTS = 300


def _sample_input() -> np.ndarray:
    rows, cols = int(np.random.randint(*ROW_RANGE)), int(np.random.randint(*COL_RANGE))
    return np.where(np.random.random((rows, cols)) < PINK_RATE, PINK, ORANGE).astype(int)


def _oracle_accepts(inp: np.ndarray) -> bool:
    if not (np.any(inp == PINK) and np.any(inp == ORANGE)):
        return False
    expected = np.where(inp == PINK, RED, inp)
    try:
        return np.array_equal(main(inp), expected)
    except Exception:
        return False


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        inp = _sample_input()
        if _oracle_accepts(inp):
            return inp
    raise RuntimeError("failed to generate valid input")
