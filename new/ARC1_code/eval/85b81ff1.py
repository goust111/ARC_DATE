import numpy as np

from arc_utils import Color, colors_present
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort the embedded binary columns by how many foreground pixels they contain.

    Input structure
    A fixed black-and-foreground lattice contains an editable 6x5 binary matrix:
    editable rows are every other row starting at row 1, and editable columns are
    every third column starting at column 1.

    Rule
    Count the foreground pixels in each editable column, then permute those
    editable columns as whole columns into ascending count order.

    Key insight
    The unchanged lattice cells are just separators; the transformation is a
    stable column permutation of the hidden binary matrix, with unique column
    counts so no tie-break is needed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    fg_colors = colors_present(g, exclude={Color.BLACK})
    assert len(fg_colors) == 1
    fg = fg_colors[0]
    assert set(map(int, np.unique(g))).issubset({Color.BLACK, fg})

    height, width = g.shape
    active_rows = np.arange(1, height, 2)
    active_cols = np.arange(1, width, 3)
    anchor_cols = np.arange(0, width, 3)
    separator_cols = np.arange(2, width, 3)
    assert active_rows.size > 0
    assert active_cols.size > 0

    assert np.all(g[np.ix_(np.arange(0, height, 2), active_cols)] == fg)
    assert np.all(g[np.ix_(np.arange(height), anchor_cols)] == fg)
    assert np.all(g[np.ix_(np.arange(height), separator_cols)] == Color.BLACK)

    editable = g[np.ix_(active_rows, active_cols)]
    counts = np.count_nonzero(editable != Color.BLACK, axis=0)
    assert len(set(map(int, counts))) == len(counts)

    order = np.argsort(counts)
    out = g.copy()
    out[np.ix_(active_rows, active_cols)] = editable[:, order]
    return out

# --------------- generator (o5x) ---------------
# Canvas/lattice geometry.
GRID_SHAPE = (13, 14)
EDITABLE_SHAPE = (6, 5)
ACTIVE_ROWS = np.arange(1, GRID_SHAPE[0], 2)
BAR_ROWS = np.arange(0, GRID_SHAPE[0], 2)
ACTIVE_COLS = np.arange(1, GRID_SHAPE[1], 3)
ANCHOR_COLS = np.arange(0, GRID_SHAPE[1], 3)
# Count/color sampling and retry limits.
FOREGROUND_COLORS = list(range(1, 10))
COUNT_VALUES = range(7)
COUNT_RUN_START_MIN, COUNT_RUN_START_MAX = 0, 2
CONSECUTIVE_COUNTS_PROB = 0.8
GEN_RETRIES = 300

def _gen_embed_matrix(matrix: np.ndarray, fg: int) -> np.ndarray:
    """Render a 6x5 editable matrix into the fixed foreground/black lattice."""
    grid = np.zeros(GRID_SHAPE, dtype=int)
    grid[:, ANCHOR_COLS] = fg
    grid[np.ix_(BAR_ROWS, ACTIVE_COLS)] = fg
    grid[np.ix_(ACTIVE_ROWS, ACTIVE_COLS)] = np.where(matrix, fg, Color.BLACK)
    return grid

def _gen_binary_column(count: int) -> np.ndarray:
    col = np.zeros(EDITABLE_SHAPE[0], dtype=int)
    if count:
        col[random.sample(range(EDITABLE_SHAPE[0]), count)] = 1
    return col

def _choose_output_counts() -> list[int]:
    """Choose distinct output column counts in ascending order."""
    if random.random() < CONSECUTIVE_COUNTS_PROB:
        start = random.randint(COUNT_RUN_START_MIN, COUNT_RUN_START_MAX)
        return list(range(start, start + EDITABLE_SHAPE[1]))
    return sorted(random.sample(COUNT_VALUES, EDITABLE_SHAPE[1]))

def _shuffled_non_identity_order(width: int) -> list[int]:
    """Return a shuffled column order, forcing one adjacent swap if unchanged."""
    order = list(range(width))
    random.shuffle(order)
    if order == list(range(width)):
        i = random.randrange(width - 1)
        order[i], order[i + 1] = order[i + 1], order[i]
    return order

def _gen_attempt() -> np.ndarray | None:
    fg = random.choice(FOREGROUND_COLORS)
    counts = _choose_output_counts()
    output_matrix = np.stack([_gen_binary_column(count) for count in counts], axis=1)
    order = _shuffled_non_identity_order(EDITABLE_SHAPE[1])
    input_matrix = output_matrix[:, order]
    input_grid = _gen_embed_matrix(input_matrix, fg)
    output_grid = _gen_embed_matrix(output_matrix, fg)

    if not np.array_equal(main(input_grid), output_grid):
        return None
    return input_grid

def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        if (grid := _gen_attempt()) is not None:
            return grid
    raise RuntimeError("failed to generate valid input for 85b81ff1")
