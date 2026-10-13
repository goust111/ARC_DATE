from __future__ import annotations

import numpy as np
from itertools import combinations
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Arrange the square input with its three quarter-turn rotations into a 2x2 pinwheel.

    Input structure:
    - A nonempty square grid; in the task examples this is a 3x3 black-background sprite in one foreground color.

    Rule:
    - Put the original grid in the upper-left block.
    - Put the 90-degree counterclockwise rotation in the upper-right block.
    - Put the 180-degree rotation in the lower-left block.
    - Put the 90-degree clockwise rotation in the lower-right block.

    Key insight:
    - The output is a block mosaic of whole-grid rotations, not per-pixel scaling.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height == width and height > 0, "expected a nonempty square grid"

    top = np.hstack((g, np.rot90(g, 1)))
    bottom = np.hstack((np.rot90(g, 2), np.rot90(g, -1)))
    return np.vstack((top, bottom))

# --------------- generator (o5x) ---------------
# Canvas and sprite bounds.
GRID_SIDE = 3
GRID_CELLS = GRID_SIDE * GRID_SIDE
# Foreground sampling.
COLOR_MIN, COLOR_MAX = 1, 9
FG_COUNT_MIN, FG_COUNT_MAX = 4, 7
# Rejection and retry limits.
FORBIDDEN_ROT90_K, MAX_GENERATION_ATTEMPTS = 1, 300
# Output oracle rotation layout.
PINWHEEL_CCW_K, PINWHEEL_HALF_K, PINWHEEL_CW_K = 1, 2, -1

def _gen_mask_catalog() -> dict[int, list[np.ndarray]]:
    masks_by_count = {k: [] for k in range(FG_COUNT_MIN, FG_COUNT_MAX + 1)}
    for count in masks_by_count:
        for cells in combinations(range(GRID_CELLS), count):
            mask = np.zeros((GRID_SIDE, GRID_SIDE), dtype=bool)
            for idx in cells:
                mask[idx // GRID_SIDE, idx % GRID_SIDE] = True
            if not np.array_equal(mask, np.rot90(mask, FORBIDDEN_ROT90_K)):
                masks_by_count[count].append(mask)
    return masks_by_count

_GEN_MASKS_BY_COUNT = _gen_mask_catalog()

def _gen_expected_output(grid: np.ndarray) -> np.ndarray:
    top = np.hstack((grid, np.rot90(grid, PINWHEEL_CCW_K)))
    bottom = np.hstack((np.rot90(grid, PINWHEEL_HALF_K), np.rot90(grid, PINWHEEL_CW_K)))
    return np.vstack((top, bottom))


def _gen_attempt() -> np.ndarray | None:
    color = random.randint(COLOR_MIN, COLOR_MAX)
    count = random.randint(FG_COUNT_MIN, FG_COUNT_MAX)
    mask = random.choice(_GEN_MASKS_BY_COUNT[count])

    grid = np.zeros((GRID_SIDE, GRID_SIDE), dtype=int)
    grid[mask] = color

    expected = _gen_expected_output(grid)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ed98d772 input")
