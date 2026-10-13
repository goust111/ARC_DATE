import numpy as np

from arc_utils import get_bg


def _triangular_columns(width: int) -> list[int]:
    cols = []
    col = 0
    step = 1
    while col < width:
        cols.append(col)
        col += step
        step += 1
    return cols


def main(grid: np.ndarray) -> np.ndarray:
    """Extend the given seed color cycle across the middle row at triangular-number column positions.

    Input structure
    The grid has three rows with a dominant background color. All foreground
    pixels lie on one row, and their columns are the initial triangular sequence
    0, 1, 3, 6, ...

    Rule
    Read the foreground colors from left to right as a repeating cycle. Starting
    from a background grid of the same shape, write that color cycle at every
    triangular column that fits within the width.

    Key insight
    The increasing gaps between active columns are 1, 2, 3, 4, ...; the input
    supplies only the first colors of the cycle, and the missing triangular
    positions continue the same cycle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape[0] == 3

    bg = get_bg(g)
    foreground_rows = np.flatnonzero(np.any(g != bg, axis=1))
    assert len(foreground_rows) == 1
    row = int(foreground_rows[0])
    assert row == 1

    seed_cols = np.flatnonzero(g[row] != bg)
    assert len(seed_cols) > 0

    tri_cols = _triangular_columns(g.shape[1])
    assert len(seed_cols) <= len(tri_cols)
    assert np.array_equal(seed_cols, np.array(tri_cols[: len(seed_cols)]))

    seed_colors = [int(g[row, c]) for c in seed_cols]
    out = np.full_like(g, bg)
    for i, col in enumerate(tri_cols):
        out[row, col] = seed_colors[i % len(seed_colors)]
    return out

# --------------- generator (o5x) ---------------

# CFG: structure, color, and retry controls.
GRID_HEIGHT, ACTIVE_ROW, BACKGROUND = 3, 1, 0
WIDTH_MIN, WIDTH_MAX_EXCL, CYCLE_LEN_MIN, CYCLE_LEN_MAX_EXCL = 19, 30, 2, 4
MIN_TRI_COLS, MAX_TRI_COLS, MIN_NEW_COLS, MAX_NEW_COLS = 6, 8, 3, 6
COLOR_POOL, GEN_RETRIES = np.arange(1, 10), 300

def _build_pair(width, tri_cols, colors) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((GRID_HEIGHT, width), BACKGROUND, dtype=int)
    out = np.full_like(inp, BACKGROUND)
    idx = np.arange(len(tri_cols))
    out[ACTIVE_ROW, tri_cols] = colors[idx % len(colors)]
    inp[ACTIVE_ROW, tri_cols[: len(colors)]] = colors
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    width = int(np.random.randint(WIDTH_MIN, WIDTH_MAX_EXCL))
    tri_cols = _triangular_columns(width)
    if not (MIN_TRI_COLS <= len(tri_cols) <= MAX_TRI_COLS):
        return None

    cycle_len = int(np.random.randint(CYCLE_LEN_MIN, CYCLE_LEN_MAX_EXCL))
    if not (MIN_NEW_COLS <= len(tri_cols) - cycle_len <= MAX_NEW_COLS):
        return None

    colors = np.random.choice(COLOR_POOL, size=cycle_len, replace=False)
    inp, out = _build_pair(width, tri_cols, colors)
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
