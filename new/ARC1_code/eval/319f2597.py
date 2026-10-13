from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox_coords


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the 2x2 black marker as an eraser that clears its rows and columns except for red pixels.

    Input structure:
    - The grid is otherwise filled with colored pixels.
    - There is exactly one 2x2 black marker.
    - Red pixels are protected cells that may lie in the marker's rows or columns.

    Rule:
    - Select both rows and both columns occupied by the 2x2 black marker.
    - Recolor every selected non-red pixel to black.
    - Leave all unselected pixels and selected red pixels unchanged.

    Key insight:
    - The marker does not expand as a filled rectangle; it defines a two-row and
      two-column cross, with red pixels acting as holes in the erased cross.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    marker = g == Color.BLACK
    assert int(np.count_nonzero(marker)) == 4, "expected exactly four black marker pixels"
    r0, c0, r1, c1 = bbox_coords(np.argwhere(marker))
    assert int(r1 - r0 + 1) == 2 and int(c1 - c0 + 1) == 2, "black marker must be 2x2"
    assert np.all(marker[r0 : r1 + 1, c0 : c1 + 1]), "black marker must be solid"

    erase = np.zeros(g.shape, dtype=bool)
    erase[r0 : r1 + 1, :] = True
    erase[:, c0 : c1 + 1] = True

    out = g.copy()
    out[erase & (g != Color.RED)] = Color.BLACK
    return out

# --------------- generator (o5x) ---------------

# Canvas and fill colors.
GRID_SHAPE = (20, 20)
FILL_COLOR_RANGE = (1, 10)

# Marker placement and retry controls.
MARKER_SIZE = 2
MARKER_TOP_LEFT_MIN = 1
GENERATION_RETRIES = 300


def _gen_attempt() -> np.ndarray:
    h, w = GRID_SHAPE
    lo, hi = FILL_COLOR_RANGE
    g = np.random.randint(lo, hi, size=GRID_SHAPE, dtype=int)

    r = int(np.random.randint(MARKER_TOP_LEFT_MIN, h - MARKER_SIZE))
    c = int(np.random.randint(MARKER_TOP_LEFT_MIN, w - MARKER_SIZE))
    g[r : r + MARKER_SIZE, c : c + MARKER_SIZE] = Color.BLACK
    return g


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        inp = _gen_attempt()

        out = main(inp)
        r0, c0, r1, c1 = bbox_coords(np.argwhere(inp == Color.BLACK))

        expected = inp.copy()
        erase = np.zeros(inp.shape, dtype=bool)
        erase[r0 : r1 + 1, :] = True
        erase[:, c0 : c1 + 1] = True
        expected[erase & (inp != Color.RED)] = Color.BLACK

        if np.array_equal(out, expected):
            return inp

    raise RuntimeError("failed to generate valid input")
