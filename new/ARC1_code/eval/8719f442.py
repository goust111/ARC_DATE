import numpy as np

from arc_utils import Color, colors_present
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Expand the 3x3 foreground mask into solid blocks, adding outward copies of the input pattern for edge pixels.

    Input structure
    The input is a 3x3 black-background mask with one foreground color.

    Rule
    Create a 15x15 black canvas with a 9x9 central block lattice: each foreground
    input pixel becomes a 3x3 solid block in its corresponding lattice position.
    If that pixel lies on the top, bottom, left, or right edge, stamp the original
    3x3 foreground mask immediately outward from the solid block on that side.

    Key insight
    The foreground pixels are anchors for local expansions; border anchors expose
    the original mask as an outward-facing copy while interior anchors only make
    the solid block.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (3, 3)

    bg = Color.BLACK
    assert np.any(g == bg)
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 1
    fg = fg_colors[0]

    n = 3
    mask = g != bg
    coords = np.argwhere(mask)
    assert len(coords) > 0

    out = np.full((n * (n + 2), n * (n + 2)), bg, dtype=int)

    def paint_pattern(top: int, left: int) -> None:
        assert 0 <= top <= out.shape[0] - n
        assert 0 <= left <= out.shape[1] - n
        for rr, cc in coords:
            out[top + int(rr), left + int(cc)] = fg

    for r, c in coords:
        r = int(r)
        c = int(c)
        block_r = n * (r + 1)
        block_c = n * (c + 1)
        out[block_r : block_r + n, block_c : block_c + n] = fg

        if r == 0:
            paint_pattern(block_r - n, block_c)
        elif r == n - 1:
            paint_pattern(block_r + n, block_c)

        if c == 0:
            paint_pattern(block_r, block_c - n)
        elif c == n - 1:
            paint_pattern(block_r, block_c + n)

    return out

# --------------- generator (o5x) ---------------

# Canvas/color and oracle geometry.
GEN_GRID_SIZE, GEN_BACKGROUND, GEN_FOREGROUND = 3, 0, 5
OUTPUT_SIZE = GEN_GRID_SIZE * (GEN_GRID_SIZE + 2)

# Foreground mask sampling and retries.
FG_COUNT_MIN, FG_COUNT_MAX = 3, 5
CENTER_CELL = (1, 1)
CARDINAL_CELLS = ((0, 1), (1, 0), (1, 2), (2, 1))
CANDIDATE_CELLS = tuple(cell for cell in np.ndindex(GEN_GRID_SIZE, GEN_GRID_SIZE) if cell != CENTER_CELL)
FORCE_CARDINAL_MIN_COUNT = 4
MAX_GENERATION_ATTEMPTS = 300


def _gen_expected_output(grid):
    fg = int(grid[grid != GEN_BACKGROUND][0])
    coords = np.argwhere(grid != GEN_BACKGROUND)
    out = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=int)

    def paint_pattern(top, left):
        out[top + coords[:, 0], left + coords[:, 1]] = fg

    for r, c in coords:
        r = int(r)
        c = int(c)
        block_r = GEN_GRID_SIZE * (r + 1)
        block_c = GEN_GRID_SIZE * (c + 1)
        out[block_r : block_r + GEN_GRID_SIZE, block_c : block_c + GEN_GRID_SIZE] = fg

        if r == 0:
            paint_pattern(block_r - GEN_GRID_SIZE, block_c)
        elif r == GEN_GRID_SIZE - 1:
            paint_pattern(block_r + GEN_GRID_SIZE, block_c)

        if c == 0:
            paint_pattern(block_r, block_c - GEN_GRID_SIZE)
        elif c == GEN_GRID_SIZE - 1:
            paint_pattern(block_r, block_c + GEN_GRID_SIZE)

    return out


def _gen_attempt():
    grid = np.zeros((GEN_GRID_SIZE, GEN_GRID_SIZE), dtype=int)
    fg_count = random.randint(FG_COUNT_MIN, FG_COUNT_MAX)
    candidates = list(CANDIDATE_CELLS)
    chosen = [CENTER_CELL]
    if fg_count >= FORCE_CARDINAL_MIN_COUNT:
        side = random.choice(CARDINAL_CELLS)
        chosen.append(side)
        candidates.remove(side)
        chosen.extend(random.sample(candidates, fg_count - 2))
    else:
        chosen.extend(random.sample(candidates, fg_count - 1))
    for r, c in chosen:
        grid[r, c] = GEN_FOREGROUND

    expected = _gen_expected_output(grid)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input():
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 8719f442 input")
