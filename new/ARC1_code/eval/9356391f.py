import numpy as np

from arc_utils import get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """Expand the lower seed pixel into concentric square rings using the top-row code as the distance palette.

    Input structure
    The grid has a dominant background, one full-width separator row, a code row
    immediately above that separator, and exactly one non-background seed pixel
    below it whose color is the first code entry.
    Rule
    Read the code row from the left edge through its last non-background entry,
    preserving any background entries inside that span as blank ring colors.  A
    cell at Chebyshev distance d from the seed takes code[d], clipped to the
    grid.  If a non-background code pixel lies directly above the seed column,
    replace that consumed code pixel with the separator color.
    Key insight
    The top row is a one-dimensional palette indexed by square-ring radius; the
    lower singleton only supplies the center of that expansion.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = map(int, g.shape)

    bg = get_bg(g)
    separator_rows: list[int] = []
    for r in range(h):
        row_values = np.unique(g[r])
        if len(row_values) == 1 and int(row_values[0]) != bg:
            separator_rows.append(r)
    assert len(separator_rows) == 1

    separator_r = separator_rows[0]
    assert separator_r > 0
    separator_color = int(g[separator_r, 0])

    code_r = separator_r - 1
    code_non_bg = np.flatnonzero(g[code_r] != bg)
    assert len(code_non_bg) > 0
    assert int(code_non_bg[0]) == 0
    last_code_c = int(code_non_bg[-1])
    code = g[code_r, : last_code_c + 1].astype(int)
    assert int(code[0]) != bg

    lower_non_bg = np.argwhere(g[separator_r + 1 :] != bg)
    assert len(lower_non_bg) == 1
    seed_r = int(lower_non_bg[0, 0]) + separator_r + 1
    seed_c = int(lower_non_bg[0, 1])
    assert int(g[seed_r, seed_c]) == int(code[0])

    radius = len(code) - 1
    assert seed_r - radius > separator_r

    out = g.copy()
    if int(g[code_r, seed_c]) != bg:
        assert seed_c <= last_code_c
        out[code_r, seed_c] = separator_color

    rows, cols = np.indices(g.shape)
    distance = np.maximum(np.abs(rows - seed_r), np.abs(cols - seed_c))
    for dist, color in enumerate(code):
        out[distance == dist] = int(color)
    return out

# --------------- generator (o5x) ---------------
# Canvas/header layout.
BG, SEPARATOR_COLOR, GRID_SIZE = 0, 5, 16
CODE_ROW, SEPARATOR_ROW = 0, 1
# Code palette sampling.
CODE_LEN_RANGE = (4, 7)
CODE_INTERIOR_BLANK_PROB = 0.35
CODE_COLORS = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
# Seed placement and retries.
SEED_ROW_RANGE, SEED_COL_RANGE = (6, 12), (5, 11)
GEN_RETRIES = 300
GRID_ROWS, GRID_COLS = np.indices((GRID_SIZE, GRID_SIZE))


def _write_header(grid: np.ndarray, code: np.ndarray) -> None:
    """Write the fixed header rows in place."""
    grid[CODE_ROW, :] = BG
    grid[CODE_ROW, : len(code)] = code
    grid[SEPARATOR_ROW, :] = SEPARATOR_COLOR


def _gen_attempt() -> np.ndarray | None:
    code_len = int(np.random.randint(*CODE_LEN_RANGE))
    radius = code_len - 1

    code = np.empty(code_len, dtype=int)
    code[0] = int(np.random.choice(CODE_COLORS))
    for idx in range(1, code_len - 1):
        if np.random.random() < CODE_INTERIOR_BLANK_PROB:
            code[idx] = BG
        else:
            code[idx] = int(np.random.choice(CODE_COLORS))
    code[-1] = int(np.random.choice(CODE_COLORS))

    seed_r_min = max(SEED_ROW_RANGE[0], radius + SEPARATOR_ROW + 1)
    seed_r = int(np.random.randint(seed_r_min, SEED_ROW_RANGE[1]))
    seed_c = int(np.random.randint(*SEED_COL_RANGE))

    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    _write_header(out, code)

    distance = np.maximum(np.abs(GRID_ROWS - seed_r), np.abs(GRID_COLS - seed_c))
    for dist, color in enumerate(code):
        out[distance == dist] = int(color)
    if seed_c < code_len and int(code[seed_c]) != BG:
        out[CODE_ROW, seed_c] = SEPARATOR_COLOR

    inp = out.copy()
    inp[distance <= radius] = BG
    _write_header(inp, code)
    inp[seed_r, seed_c] = int(code[0])

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 9356391f input")
