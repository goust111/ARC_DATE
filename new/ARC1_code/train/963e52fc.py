import numpy as np


def _minimal_horizontal_period(g: np.ndarray) -> int:
    h, w = g.shape
    assert h > 0 and w > 1, "expected a nonempty grid with multiple columns"

    source_cols = np.arange(w)
    for period in range(1, w):
        if np.array_equal(g, g[:, source_cols % period]):
            return period
    raise AssertionError("expected a proper horizontal period")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the existing horizontal color pattern to twice the input width.

    Input structure
    - The grid has a black background and one or more non-background rows.
    - Every column belongs to a repeating horizontal period that may be truncated
      at the right edge of the input.
    Rule
    - Find the shortest horizontal period that explains all input rows.
    - Keep the height fixed and continue that period until the grid is twice as
      wide as the input.
    Key insight
    - The output is not a copy of the input appended to itself; it is the next
      columns of the underlying periodic tile.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _, w = g.shape
    period = _minimal_horizontal_period(g)
    out_cols = np.arange(2 * w) % period
    return g[:, out_cols]

# --------------- generator (o5x) ---------------
from itertools import product

# Canvas and repeated-pattern size bounds.
GEN_PERIOD_BOUNDS = (2, 5)
GEN_WIDTH_BOUNDS = (6, 10)
GEN_HEIGHT_BOUNDS = (3, 8)
GEN_MIN_FULL_REPEATS = 2
# Active-row and color constraints.
GEN_ROW_COUNT_BOUNDS = (1, 3)
GEN_COLOR_VALUES = np.arange(1, 10)
GEN_COLOR_COUNT = 2
# Rejection sampling budget.
GEN_RETRY_LIMIT = 300

_GEN_TILE_CACHE = {}


def _gen_row_options(period):
    rows = []
    for mask in range(1, 1 << period):
        bits = np.array([(mask >> i) & 1 for i in range(period)], dtype=int)
        if bits.min() == 0 and bits.max() == 1:
            rows.append(bits)
    return rows


def _gen_tile_options(period, row_count, width):
    key = (period, row_count, width)
    if key in _GEN_TILE_CACHE:
        return _GEN_TILE_CACHE[key]

    rows = _gen_row_options(period)
    options = []
    for row_stack in product(rows, repeat=row_count):
        tile = np.vstack(row_stack)
        probe = tile[:, np.arange(width) % period]
        if _minimal_horizontal_period(probe) == period:
            options.append(tile.copy())

    _GEN_TILE_CACHE[key] = options
    return options


def _sample_layout():
    period = int(np.random.randint(*GEN_PERIOD_BOUNDS))
    min_width = max(GEN_WIDTH_BOUNDS[0], GEN_MIN_FULL_REPEATS * period)
    width = int(np.random.randint(min_width, GEN_WIDTH_BOUNDS[1]))
    height = int(np.random.randint(*GEN_HEIGHT_BOUNDS))
    row_count = int(np.random.randint(*GEN_ROW_COUNT_BOUNDS))
    row0 = int(np.random.randint(0, height - row_count + 1))
    return period, width, height, row_count, row0


def _render_grid(height, width, row0, period, binary_tile):
    colors = np.random.choice(GEN_COLOR_VALUES, size=GEN_COLOR_COUNT, replace=False)
    tile = np.where(binary_tile == 0, colors[0], colors[1])
    grid = np.zeros((height, width), dtype=int)
    cols = np.arange(width) % period
    grid[row0 : row0 + binary_tile.shape[0], :] = tile[:, cols]
    return grid


def _passes_oracle(grid, period):
    expected_cols = np.arange(2 * grid.shape[1]) % period
    return (
        _minimal_horizontal_period(grid) == period
        and np.array_equal(main(grid), grid[:, expected_cols])
    )


def _gen_attempt():
    period, width, height, row_count, row0 = _sample_layout()

    options = _gen_tile_options(period, row_count, width)
    if not options:
        return None
    binary_tile = options[int(np.random.randint(0, len(options)))]

    grid = _render_grid(height, width, row0, period, binary_tile)
    if not _passes_oracle(grid, period):
        return None

    return grid


def generate_input():
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic grid")
