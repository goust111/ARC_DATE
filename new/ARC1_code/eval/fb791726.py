import numpy as np

from arc_utils import Color, get_bg


def _gap_rows(g: np.ndarray, bg: int) -> list[int]:
    height = int(g.shape[0])
    has_fg = np.any(g != int(bg), axis=1)

    rows = []
    covered_rows: set[int] = set()
    for r in range(height - 2):
        if not bool(has_fg[r]):
            continue
        if np.array_equal(g[r], g[r + 2]):
            assert not bool(has_fg[r + 1]), "paired rows must be separated by a blank row"
            rows.append(r + 1)
            covered_rows.update((r, r + 2))

    fg_rows = {int(r) for r in np.flatnonzero(has_fg)}
    assert rows, "expected at least one foreground row pair"
    assert covered_rows == fg_rows, "all foreground rows must belong to matched row pairs"
    return rows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Double the canvas, copy the input on the main diagonal, and draw green separator rows through each vertical gap.

    Input structure
    The background is black. All foreground pixels form one or more identical
    non-background row pairs separated by a blank row.

    Rule
    Build a 2H x 2W black canvas, paste the input into the upper-left and
    lower-right H x W blocks, and color the blank row between each paired row
    green across the full output width in both diagonal copies.

    Key insight
    The foreground marks horizontal separator rows by bracketing each gap with
    matching rows; the whole pattern is then repeated on the diagonal.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "input must be a non-empty 2D grid"

    bg = get_bg(g)
    assert bg == Color.BLACK, "expected black background"
    gap_rows = _gap_rows(g, bg)

    height, width = map(int, g.shape)
    out = np.full((2 * height, 2 * width), bg, dtype=int)
    out[:height, :width] = g
    out[height:, width:] = g

    for row in gap_rows:
        out[row, :] = Color.GREEN
        out[row + height, :] = Color.GREEN

    return out

# --------------- generator (o5x) ---------------

# Configuration: canvas, pair-block structure, colors, and retry budget.
SIZE_MIN, SIZE_MAX_EXCLUSIVE = 3, 9
MAX_PAIR_BLOCKS = 3
PAIR_HEIGHT, GAP_OFFSET, MATCH_OFFSET = 3, 1, 2
BACKGROUND_COLOR, SEPARATOR_COLOR = Color.BLACK, Color.GREEN
FOREGROUND_COLORS = [1, 2, 4, 5, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 300


def _gen_attempt() -> np.ndarray | None:
    size = int(np.random.randint(SIZE_MIN, SIZE_MAX_EXCLUSIVE))
    max_pairs = min(MAX_PAIR_BLOCKS, size // PAIR_HEIGHT)
    num_pairs = int(np.random.randint(1, max_pairs + 1))
    fg = int(np.random.choice(FOREGROUND_COLORS))
    pair_rows = PAIR_HEIGHT * np.arange(num_pairs)
    cols = [int(np.random.randint(0, size)) for _ in range(num_pairs)]

    grid = np.full((size, size), BACKGROUND_COLOR, dtype=int)
    grid[pair_rows, cols] = fg
    grid[pair_rows + MATCH_OFFSET, cols] = fg

    expected = np.full((2 * size, 2 * size), BACKGROUND_COLOR, dtype=int)
    expected[:size, :size] = grid
    expected[size:, size:] = grid
    gap_rows = pair_rows + GAP_OFFSET
    expected[gap_rows, :] = SEPARATOR_COLOR
    expected[gap_rows + size, :] = SEPARATOR_COLOR

    try:
        out = main(grid)
    except Exception:
        return None
    if not np.array_equal(out, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is None:
            continue
        return grid
    raise RuntimeError("failed to generate a valid fb791726 input")
