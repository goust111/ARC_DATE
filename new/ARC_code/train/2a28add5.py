import numpy as np

from arc_utils import Color, get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Paint each pink control pixel as a teal horizontal bar whose reach equals the foreground counts on that row.

    Input structure
    - The grid has a dominant background color and one or more pink control pixels.
    - Each marker row has exactly one pink pixel; all other non-background pixels
      in that row are counters.
    Rule
    - Start from an all-background output grid.
    - For each pink pixel, count non-background pixels strictly to its left and
      strictly to its right in the same row.
    - Paint teal from c - left_count through c + right_count on that row.
    Key insight
    - The foreground pixels are not copied; only their counts on each side of
      each pink marker encode the teal bar length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    assert bg not in (Color.PINK, Color.TEAL)

    markers = np.argwhere(g == Color.PINK)
    assert len(markers) > 0

    out = np.full_like(g, bg)
    for r, c in markers:
        assert np.count_nonzero(g[r] == Color.PINK) == 1
        left_count = int(np.count_nonzero(g[r, :c] != bg))
        right_count = int(np.count_nonzero(g[r, c + 1 :] != bg))

        start = int(c) - left_count
        end = int(c) + right_count
        assert 0 <= start <= c <= end < g.shape[1]
        out[int(r), start : end + 1] = Color.TEAL

    return out

# --------------- generator (o5x) ---------------

# Canvas and color palette.
GRID_HEIGHT, GRID_WIDTH, BACKGROUND_COLOR = 10, 10, 7
COUNTER_COLORS = [0, 1, 2, 3, 4, 5, Color.TEAL, 9]

# Marker-row counts and encoded teal bar lengths.
MARKER_COUNT_RANGE, SIDE_COUNT_RANGE, BAR_LENGTH_RANGE = (4, 5), range(6), (3, 7)
VALID_SIDE_COUNTS = [
    (left, right)
    for left in SIDE_COUNT_RANGE
    for right in SIDE_COUNT_RANGE
    if BAR_LENGTH_RANGE[0] <= left + right + 1 <= BAR_LENGTH_RANGE[1]
]

# Distractor density and retry budget.
DISTRACTOR_COUNT_RANGE, GENERATION_RETRY_LIMIT = (0, 7), 300

def _balanced_counter_color():
    """Yield counter colors from shuffled complete bags."""
    bag = []
    while True:
        if not bag:
            bag = COUNTER_COLORS[:]
            random.shuffle(bag)
        yield bag.pop()

def _choose_row_spec(row):
    left_count, right_count = random.choice(VALID_SIDE_COUNTS)
    col = random.randint(left_count, GRID_WIDTH - right_count - 1)
    return row, col, left_count, right_count

def _choose_row_specs():
    """Return marker rows and specs as (row, pink_col, left_count, right_count)."""
    marker_count = random.randint(*MARKER_COUNT_RANGE)
    marker_rows = sorted(random.sample(range(GRID_HEIGHT), marker_count))
    return marker_rows, [_choose_row_spec(row) for row in marker_rows]

def _scatter_marker_rows(inp, row_specs, colors):
    for row, col, left_count, right_count in row_specs:
        inp[row, col] = Color.PINK
        left_cols = random.sample(range(col), left_count)
        right_cols = random.sample(range(col + 1, GRID_WIDTH), right_count)
        counter_cols = left_cols + right_cols
        random.shuffle(counter_cols)
        for cc in counter_cols:
            inp[row, cc] = next(colors)

def _scatter_distractor_rows(inp, marker_rows, colors):
    marker_row_set = set(marker_rows)
    for row in range(GRID_HEIGHT):
        if row in marker_row_set:
            continue
        count = random.randint(*DISTRACTOR_COUNT_RANGE)
        for col in random.sample(range(GRID_WIDTH), count):
            inp[row, col] = next(colors)

def _gen_attempt() -> np.ndarray | None:
    marker_rows, row_specs = _choose_row_specs()
    out = np.full((GRID_HEIGHT, GRID_WIDTH), BACKGROUND_COLOR, dtype=int)
    for row, col, left_count, right_count in row_specs:
        out[row, col - left_count : col + right_count + 1] = Color.TEAL

    inp = np.full((GRID_HEIGHT, GRID_WIDTH), BACKGROUND_COLOR, dtype=int)
    colors = _balanced_counter_color()

    _scatter_marker_rows(inp, row_specs, colors)
    _scatter_distractor_rows(inp, marker_rows, colors)

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None

def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for 2a28add5")
