import numpy as np

from arc_utils import get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the colored pixels on a 5x5 X into the corresponding 3x3 square outline.

    Input structure
    - The input is a 5x5 grid with a dominant background color.
    - Every non-background pixel lies on one of the two diagonals, forming an X.
    - All diagonal positions are colored foreground pixels.

    Rule
    - Keep the center pixel at the center of the 3x3 output.
    - Project the northwest diagonal arm onto the top edge from left to center.
    - Project the northeast arm onto the right edge from top to center.
    - Project the southwest arm onto the left edge from center to bottom.
    - Project the southeast arm onto the bottom edge from center to right.

    Key insight
    - The output is not a crop; it is the same X-shaped set of nine colored pixels
      folded into the perimeter and center of a 3x3 grid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape == (5, 5), "expected a 5x5 grid"

    bg = get_bg(g)
    x_mask = np.zeros(g.shape, dtype=bool)
    for i in range(5):
        x_mask[i, i] = True
        x_mask[i, 4 - i] = True

    assert np.all(g[~x_mask] == bg), "expected only background off the X"
    assert np.all(g[x_mask] != bg), "expected all X positions to be foreground"

    out = np.full((3, 3), -1, dtype=int)
    center = 2
    for r, c in np.argwhere(x_mask):
        r = int(r)
        c = int(c)
        if r == center and c == center:
            rr, cc = 1, 1
        elif r < center and c < center:
            rr, cc = 0, r
        elif r < center and c > center:
            rr, cc = r, 2
        elif r > center and c < center:
            rr, cc = r - center, 0
        elif r > center and c > center:
            rr, cc = 2, c - center
        else:
            raise AssertionError("unexpected X position")
        out[rr, cc] = g[r, c]

    assert np.all(out >= 0), "expected every output cell to be filled"
    return out

# --------------- generator (o5x) ---------------
# Canvas, palette, and retry envelope.
GRID_SHAPE = (5, 5)
BACKGROUND_COLOR = 0
NON_BACKGROUND_COLORS = np.arange(1, 10)
PALETTE_SIZE = 5
GENERATION_RETRIES = 300

# X-pair geometry and fold oracle.
ROTATIONAL_PAIR_COORDS = np.array(
    [(0, 0), (4, 4), (0, 4), (4, 0), (1, 1), (3, 3), (1, 3), (3, 1)],
    dtype=int,
)
CENTER_COORD = (2, 2)
OUTPUT_SOURCE_COORDS = np.array(
    [[(0, 0), (1, 1), (0, 4)], [(3, 1), (2, 2), (1, 3)], [(4, 0), (3, 3), (4, 4)]],
    dtype=int,
)


def _gen_expected_output(grid: np.ndarray) -> np.ndarray:
    return grid[OUTPUT_SOURCE_COORDS[..., 0], OUTPUT_SOURCE_COORDS[..., 1]].astype(int, copy=False)


def _gen_attempt() -> np.ndarray | None:
    colors = np.random.choice(NON_BACKGROUND_COLORS, size=PALETTE_SIZE, replace=False)
    grid = np.full(GRID_SHAPE, BACKGROUND_COLOR, dtype=int)
    grid[ROTATIONAL_PAIR_COORDS[:, 0], ROTATIONAL_PAIR_COORDS[:, 1]] = np.repeat(colors[:-1], 2)
    grid[CENTER_COORD] = int(colors[-1])
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        inp = _gen_attempt()
        if inp is None:
            continue
        expected = _gen_expected_output(inp)
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate valid input")
