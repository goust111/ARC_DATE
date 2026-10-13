import numpy as np

from arc_utils import bbox, get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each color in the 2x2 marker block into the diagonally opposite clipped rectangle around that block.

    Input structure
    - The grid has a dominant background color and exactly one 2x2 foreground block.

    Rule
    - Keep the marker block unchanged.
    - Each marker pixel fills the outside rectangle diagonally opposite its position in the block.
    - Each painted rectangle is adjacent to the marker block and extends at most two rows and two columns, clipped by the grid edge.

    Key insight
    - The 2x2 block is a local quadrant legend: each cell names the color for the opposite quadrant-like region around the block.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    mask = g != bg
    assert int(np.count_nonzero(mask)) == 4

    b = bbox(mask)
    assert b is not None
    r0, c0, r1, c1 = b
    assert r1 == r0 + 1
    assert c1 == c0 + 1

    block = g[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(block != bg)

    h, w = g.shape
    up = min(2, r0)
    down = min(2, h - r1 - 1)
    left = min(2, c0)
    right = min(2, w - c1 - 1)
    assert min(up, down, left, right) > 0

    out = g.copy()
    out[r0 - up : r0, c0 - left : c0] = block[1, 1]
    out[r0 - up : r0, c1 + 1 : c1 + 1 + right] = block[1, 0]
    out[r1 + 1 : r1 + 1 + down, c0 - left : c0] = block[0, 1]
    out[r1 + 1 : r1 + 1 + down, c1 + 1 : c1 + 1 + right] = block[0, 0]
    return out

# --------------- generator (o5x) ---------------
# Canvas and marker geometry.
GRID_SIZE_MIN, GRID_SIZE_MAX = 5, 10
MARKER_SIZE, EDGE_MARGIN = 2, 1
BACKGROUND_COLOR = 0
# Color sampling and retry limits.
FOREGROUND_COLORS = tuple(range(1, 10))
MARKER_COLOR_COUNT = MARKER_SIZE * MARKER_SIZE
MAX_GENERATION_ATTEMPTS = 300

def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    w = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    r0 = random.randint(EDGE_MARGIN, h - EDGE_MARGIN - MARKER_SIZE)
    c0 = random.randint(EDGE_MARGIN, w - EDGE_MARGIN - MARKER_SIZE)
    colors = random.sample(FOREGROUND_COLORS, MARKER_COLOR_COUNT)
    grid = np.full((h, w), BACKGROUND_COLOR, dtype=int)
    block = np.array(colors, dtype=int).reshape(MARKER_SIZE, MARKER_SIZE)
    grid[r0 : r0 + MARKER_SIZE, c0 : c0 + MARKER_SIZE] = block
    try:
        main(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
