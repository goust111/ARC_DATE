import numpy as np
from arc_utils import bbox, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the colors from the foreground 2x2 block to the four grid corners, preserving their relative positions.

    Input structure:
    - The grid has one background color and exactly four foreground pixels.
    - The foreground pixels fill a 2x2 rectangle.
    - Each foreground pixel's row and column within that 2x2 block identify the matching output corner.

    Rule:
    - Create a background-filled output grid with the same shape as the input.
    - Send the top-left, top-right, bottom-left, and bottom-right pixels of the 2x2 block to the corresponding grid corners.
    - Preserve each moved pixel's color.

    Key insight:
    - The foreground block is an inset version of the corner pattern; expand the block's bounding rectangle to the full grid.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    mask = g != bg
    coords = np.argwhere(mask)
    assert coords.shape[0] == 4

    box = bbox(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    assert (r1 - r0, c1 - c0) == (1, 1)
    assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] != bg)

    h, w = g.shape
    assert h >= 2 and w >= 2

    out = np.full_like(g, bg)
    out[0, 0] = g[r0, c0]
    out[0, w - 1] = g[r0, c1]
    out[h - 1, 0] = g[r1, c0]
    out[h - 1, w - 1] = g[r1, c1]
    return out

# --------------- generator (o5x) ---------------

# Canvas and 2x2 block constraints.
GRID_SIZE_MIN, GRID_SIZE_MAX_EXCLUSIVE = 4, 9
BLOCK_SIZE, BORDER_MARGIN = 2, 1
# Color sampling and retry budget.
COLOR_MIN, COLOR_MAX_EXCLUSIVE = 0, 10
MAX_ATTEMPTS = 300

def _gen_attempt() -> np.ndarray | None:
    h = int(np.random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX_EXCLUSIVE))
    w = int(np.random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX_EXCLUSIVE))
    bg = int(np.random.randint(COLOR_MIN, COLOR_MAX_EXCLUSIVE))
    palette = [c for c in range(COLOR_MIN, COLOR_MAX_EXCLUSIVE) if c != bg]
    block = np.random.choice(palette, size=BLOCK_SIZE ** 2, replace=False).astype(int)
    block = block.reshape(BLOCK_SIZE, BLOCK_SIZE)

    r0 = int(np.random.randint(BORDER_MARGIN, h - BLOCK_SIZE))
    c0 = int(np.random.randint(BORDER_MARGIN, w - BLOCK_SIZE))
    inp = np.full((h, w), bg, dtype=int)
    inp[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE] = block

    out = np.full((h, w), bg, dtype=int)
    out[[0, 0, h - 1, h - 1], [0, w - 1, 0, w - 1]] = block.ravel()
    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 66e6c45b input")
