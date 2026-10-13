import numpy as np

from arc_utils import get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Fill any row whose left and right boundary markers have the same color with that color.

    Input structure
    - The background is the dominant color.
    - Foreground pixels appear only as paired row markers on the left and right borders.
    - Each marked row has exactly two foreground pixels, one at each horizontal boundary.
    Rule
    - Copy the grid.
    - For every marked row, compare the two boundary marker colors.
    - If the marker colors match, paint the whole row with that color; otherwise leave the row unchanged.
    Key insight
    - The two edge markers are a row-level equality test; matching endpoints authorize a full-width horizontal bar.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    H, W = g.shape
    assert W >= 2

    bg = int(get_bg(g))
    fg = np.argwhere(g != bg)
    assert len(fg) > 0
    assert bool(np.all((fg[:, 1] == 0) | (fg[:, 1] == W - 1)))

    out = g.copy()
    for r in range(H):
        marker_cols = np.flatnonzero(g[r] != bg)
        if len(marker_cols) == 0:
            continue
        assert len(marker_cols) == 2
        assert int(marker_cols[0]) == 0
        assert int(marker_cols[1]) == W - 1

        left = int(g[r, 0])
        right = int(g[r, W - 1])
        if left == right:
            out[r, :] = left

    return out

# --------------- generator (o5x) ---------------

# Canvas and marker geometry.
GRID_SHAPE = (10, 10)
MARKER_ROWS = (1, 3, 5, 7, 9)
LEFT_COL, RIGHT_COL = 0, GRID_SHAPE[1] - 1

# Row-count, color sampling, and retry controls.
MARKED_ROW_COUNTS = (4, 5)
MATCH_COUNT_RANGE = (0, 2)
COLOR_VALUES = tuple(range(1, 10))
COLOR_POOL_SIZES = (4, 5)
MAX_GENERATION_ATTEMPTS = 300


def _seed_pairs(rows, match_rows, colors):
    """Assign sampled colors to shuffled row-side units before gap filling."""
    pairs = {r: [None, None] for r in rows}
    units = []
    for r in rows:
        if r in match_rows:
            units.append((r, "both"))
        else:
            units.extend([(r, "left"), (r, "right")])

    random.shuffle(units)
    for color, (r, side) in zip(colors, units):
        if side == "both":
            pairs[r] = [color, color]
        else:
            pairs[r][0 if side == "left" else 1] = color
    return pairs


def _complete_pair(pair, colors, must_match):
    left, right = pair
    if must_match:
        if left is None:
            left = random.choice(colors)
        return int(left), int(left)

    if left is None and right is None:
        left, right = random.sample(colors, 2)
    elif left is None:
        left = random.choice([c for c in colors if c != right])
    elif right is None:
        right = random.choice([c for c in colors if c != left])
    elif left == right:
        right = random.choice([c for c in colors if c != left])
    return int(left), int(right)


def _gen_attempt() -> np.ndarray | None:
    grid = np.zeros(GRID_SHAPE, dtype=int)
    rows = list(MARKER_ROWS[:random.choice(MARKED_ROW_COUNTS)])
    match_rows = set(random.sample(rows, random.randint(*MATCH_COUNT_RANGE)))
    colors = random.sample(COLOR_VALUES, random.choice(COLOR_POOL_SIZES))
    pairs = _seed_pairs(rows, match_rows, colors)

    for r in rows:
        left, right = _complete_pair(pairs[r], colors, r in match_rows)
        grid[r, [LEFT_COL, RIGHT_COL]] = (left, right)

    expected = grid.copy()
    for r in match_rows:
        expected[r, :] = grid[r, LEFT_COL]

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        gen_grid = _gen_attempt()
        if gen_grid is not None:
            return gen_grid
    raise RuntimeError("failed to generate a valid input")
