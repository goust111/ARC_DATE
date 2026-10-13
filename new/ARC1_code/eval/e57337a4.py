from __future__ import annotations

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the enlarged uniform grid into a 3x3 map showing which coarse blocks contain sparse marker pixels.

    Input structure
    - The input is a nonempty two-dimensional grid whose height and width are
      each divisible into three equal bands.
    - Exactly two colors are present: a unique dominant fill color and a unique
      sparse marker color.

    Rule
    - Split the input into a 3x3 array of equal rectangular blocks.
    - For each block, write the marker color in the corresponding output cell if
      that block contains any marker pixel; otherwise write the fill color.

    Key insight
    - The large grid is a scaled-up 3x3 code: marker location within a block is
      irrelevant, only the occupied coarse block matters.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    height, width = map(int, g.shape)
    assert height % 3 == 0 and width % 3 == 0, "grid must split evenly into 3x3 blocks"

    colors, counts = np.unique(g, return_counts=True)
    assert colors.size == 2, "expected exactly a fill color and a marker color"
    assert int(counts[0]) != int(counts[1]), "fill and marker colors must have distinct populations"

    fill = int(colors[int(np.argmax(counts))])
    marker = int(colors[int(np.argmin(counts))])
    block_h = height // 3
    block_w = width // 3

    out = np.full((3, 3), fill, dtype=int)
    for block_r in range(3):
        for block_c in range(3):
            block = g[
                block_r * block_h : (block_r + 1) * block_h,
                block_c * block_w : (block_c + 1) * block_w,
            ]
            if np.any(block == marker):
                out[block_r, block_c] = marker

    return out

# --------------- generator (o5x) ---------------

# Canvas/block geometry.
GRID_SHAPE, OUTPUT_SHAPE = (15, 15), (3, 3)
BLOCK_COUNT = OUTPUT_SHAPE[0] * OUTPUT_SHAPE[1]
# Color and marker sampling.
FILL_COLOR_RANGE, MARKER_COUNT_RANGE = (1, 10), (1, 4)
MARKER_COLOR = 0
# Oracle retry budget.
GENERATION_RETRIES = 300


def _gen_attempt() -> np.ndarray | None:
    fill = int(np.random.randint(*FILL_COLOR_RANGE))
    grid = np.full(GRID_SHAPE, fill, dtype=int)
    expected = np.full(OUTPUT_SHAPE, fill, dtype=int)
    block_h = GRID_SHAPE[0] // OUTPUT_SHAPE[0]
    block_w = GRID_SHAPE[1] // OUTPUT_SHAPE[1]

    marker_count = int(np.random.randint(*MARKER_COUNT_RANGE))
    occupied = np.random.choice(BLOCK_COUNT, size=marker_count, replace=False)
    for block_index in occupied:
        block_r = int(block_index // OUTPUT_SHAPE[1])
        block_c = int(block_index % OUTPUT_SHAPE[1])
        r = block_r * block_h + int(np.random.randint(block_h))
        c = block_c * block_w + int(np.random.randint(block_w))
        grid[r, c] = MARKER_COLOR
        expected[block_r, block_c] = MARKER_COLOR

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GENERATION_RETRIES} attempts")
