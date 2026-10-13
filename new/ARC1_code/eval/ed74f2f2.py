from __future__ import annotations

import numpy as np

from arc_utils import Color
import random


_CODEBOOK = {
    ((True, True, True), (False, True, False), (False, True, False)): Color.BLUE,
    ((True, True, False), (False, True, False), (False, True, True)): Color.RED,
    ((False, True, True), (False, True, False), (True, True, False)): Color.GREEN,
}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the left 3x3 grey glyph as a color and recolor the right 3x3 grey mask with that color.

    Input structure
    - The input is a 5x9 black-background grid.
    - Rows 1..3 contain two 3x3 grey/black regions separated by a black column.
    - The left region is a five-pixel color-code glyph and the right region is
      the mask to emit.

    Rule
    - Decode the left glyph with the task codebook: top-T -> blue,
      down-right path -> red, and down-left path -> green.
    - Return the right 3x3 region with grey mask pixels recolored to the decoded
      color and black pixels kept black.

    Key insight
    - The output shape is not transformed geometrically; only the left glyph
      supplies the replacement color for the cropped right mask.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (5, 9), "expected a 5x9 input grid"
    assert set(map(int, np.unique(g))) <= {Color.BLACK, Color.GREY}, "expected only black and grey"
    assert np.all(g[[0, 4], :] == Color.BLACK), "expected black top and bottom borders"
    assert np.all(g[:, [0, 4, 8]] == Color.BLACK), "expected black side and separator columns"

    code = g[1:4, 1:4] == Color.GREY
    mask = g[1:4, 5:8] == Color.GREY
    assert int(np.count_nonzero(code)) == 5, "expected a five-pixel code glyph"
    assert np.any(mask), "expected a nonempty output mask"

    code_key = tuple(tuple(bool(cell) for cell in row) for row in code)
    assert code_key in _CODEBOOK, "unrecognized code glyph"
    color = _CODEBOOK[code_key]

    out = np.full((3, 3), Color.BLACK, dtype=int)
    out[mask] = color
    return out

# --------------- generator (o5x) ---------------

# Fixed task geometry for the 5x9 input and two embedded 3x3 regions.
GRID_SHAPE, PATCH_SHAPE = (5, 9), (3, 3)
CODE_REGION, MASK_REGION = np.s_[1:4, 1:4], np.s_[1:4, 5:8]

# Sampling controls for mask density and retry budget.
MASK_COUNT_RANGE, MAX_GENERATION_ATTEMPTS = (4, 7), 300
MASK_CELL_INDICES = range(PATCH_SHAPE[0] * PATCH_SHAPE[1])

# Code glyph masks and their decoded output colors.
GLYPH_OPTIONS = (
    (np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool), Color.BLUE),
    (np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool), Color.RED),
    (np.array([[0, 1, 1], [0, 1, 0], [1, 1, 0]], dtype=bool), Color.GREEN),
)


def _gen_attempt() -> np.ndarray | None:
    grid = np.zeros(GRID_SHAPE, dtype=int)
    glyph, color = random.choice(GLYPH_OPTIONS)
    left = grid[CODE_REGION]
    left[glyph] = Color.GREY

    mask = np.zeros(PATCH_SHAPE, dtype=bool)
    mask_count = random.randint(*MASK_COUNT_RANGE)
    mask.flat[random.sample(MASK_CELL_INDICES, mask_count)] = True
    right = grid[MASK_REGION]
    right[mask] = Color.GREY

    expected = np.zeros(PATCH_SHAPE, dtype=int)
    expected[mask] = color
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid ed74f2f2 input")
