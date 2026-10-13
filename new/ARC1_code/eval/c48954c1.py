import numpy as np
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a 3x3 mosaic of whole-grid mirror reflections around the original tile.

    Input structure
    - The input is a 3x3 color grid; every cell belongs to the source tile.

    Rule
    - Produce a 9x9 grid split into nine 3x3 tile slots.
    - The center slot is the original tile.
    - Slots above or below the center reverse source rows.
    - Slots left or right of the center reverse source columns.
    - Corner slots reverse both rows and columns.

    Key insight
    The output is reflection wallpaper of the entire small grid, not per-pixel scaling or color-specific processing.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape == (3, 3), "expected the 3x3 source tile"

    rows = np.arange(g.shape[0])
    cols = np.arange(g.shape[1])
    row_order = np.concatenate((rows[::-1], rows, rows[::-1]))
    col_order = np.concatenate((cols[::-1], cols, cols[::-1]))
    return g[row_order[:, None], col_order[None, :]]

# --------------- generator (o5x) ---------------
# Color/tile sampling config.
TILE_SHAPE, TILE_CELLS = (3, 3), 9
COLOR_VALUES = range(1, 10)
PALETTE_SIZE_RANGE = (3, 4)
MAX_ATTEMPTS = 300
# Oracle reflection config.
REFLECTION_ORDER = np.array([2, 1, 0, 0, 1, 2, 2, 1, 0])

def _gen_expected_output(tile: np.ndarray) -> np.ndarray:
    return tile[REFLECTION_ORDER[:, None], REFLECTION_ORDER[None, :]]


def _gen_attempt() -> np.ndarray | None:
    palette_size = random.randint(*PALETTE_SIZE_RANGE)
    palette = random.sample(COLOR_VALUES, palette_size)

    cells = list(palette)
    cells.extend(random.choices(palette, k=TILE_CELLS - palette_size))
    random.shuffle(cells)

    return np.array(cells, dtype=int).reshape(TILE_SHAPE)


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        inp = _gen_attempt()
        if inp is None:
            continue
        out = _gen_expected_output(inp)
        if np.array_equal(main(inp), out):
            return inp
    raise RuntimeError("failed to generate a valid c48954c1 input")
