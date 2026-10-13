import numpy as np

from arc_utils import get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend every foreground pixel straight downward, filling its column to the bottom with that pixel's color.

    Input structure
    - The background is the dominant color.
    - Each non-background pixel is a source in a distinct column.
    - Cells below each source in the same column are background.

    Rule
    - Copy the input grid.
    - For each foreground source at (r, c), color every cell from row r through
      the bottom of column c with the source color.
    - Preserve all cells outside those downward vertical rays.

    Key insight
    - Each colored pixel defines both the color and the column of a vertical
      downward fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    source_coords = np.argwhere(g != bg)
    assert source_coords.shape[0] > 0, "expected at least one foreground source"

    source_cols = source_coords[:, 1]
    assert len(np.unique(source_cols)) == source_coords.shape[0], "expected one source per used column"

    out = g.copy()
    for r_raw, c_raw in source_coords:
        r = int(r_raw)
        c = int(c_raw)
        assert np.all(g[r + 1 :, c] == bg), "source column must be clear below the source"
        out[r:, c] = int(g[r, c])

    return out

# --------------- generator (o5x) ---------------
# Canvas, palette, and retry budget.
GRID_SHAPE = (3, 3)
BACKGROUND = 0
COLUMNS = range(3)
COLOR_VALUES = range(1, 10)
MAX_ATTEMPTS = 300
# Source count and row layout weights.
SOURCE_COUNT_RANGE = (2, 3)
TWO_SOURCE_ROW_PATTERNS = ((0, 0), (0, 1), (1, 0))
THREE_SOURCE_ROW_PATTERNS = ((1, 1, 1), (0, 1, 2), (0, 0, 2), (0, 1, 1))
def _gen_attempt() -> np.ndarray | None:
    inp = np.full(GRID_SHAPE, BACKGROUND, dtype=int)
    n_sources = random.randint(*SOURCE_COUNT_RANGE)
    cols = random.sample(COLUMNS, n_sources)
    colors = random.sample(COLOR_VALUES, n_sources)
    if n_sources == SOURCE_COUNT_RANGE[0]:
        row_pattern = random.choice(TWO_SOURCE_ROW_PATTERNS)
    else:
        row_pattern = list(random.choice(THREE_SOURCE_ROW_PATTERNS))
        random.shuffle(row_pattern)
    expected = inp.copy()
    for row, col, color in zip(row_pattern, cols, colors):
        inp[row, col] = color
        expected[row:, col] = color
    return inp if np.array_equal(main(inp), expected) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
