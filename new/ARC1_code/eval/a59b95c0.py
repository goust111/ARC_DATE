import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """Repeat the input tile in both directions once for each distinct input color.

    Input structure
    A non-empty 2D color grid; the examples are 3x3 tiles containing two to four
    distinct colors.

    Rule
    Count the number of distinct colors in the input. Tile the entire input grid
    that many times vertically and that many times horizontally.

    Key insight
    The output size is controlled by the palette size of the input, and the unit
    being repeated is the whole grid rather than individual pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.size > 0

    repeat_count = len(np.unique(g))
    assert repeat_count > 0
    return np.tile(g, (repeat_count, repeat_count))

# --------------- generator (o5x) ---------------
# CFG: palette bounds, grid geometry, and retry limit.
PALETTE_SIZE_RANGE = (2, 5)
COLOR_POOL = np.arange(1, 10)
GRID_SHAPE, CELL_COUNT = (3, 3), 9
MAX_ATTEMPTS = 300

def _gen_attempt() -> np.ndarray | None:
    k = int(np.random.randint(*PALETTE_SIZE_RANGE))
    colors = np.random.choice(COLOR_POOL, size=k, replace=False).astype(int)

    cells = np.empty(CELL_COUNT, dtype=int)
    cells[:k] = colors
    cells[k:] = np.random.choice(colors, size=CELL_COUNT - k, replace=True)
    np.random.shuffle(cells)

    grid = cells.reshape(GRID_SHAPE)
    expected = np.tile(grid, (k, k))
    if np.array_equal(main(grid), expected):
        return grid
    return None

def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid a59b95c0 input")
