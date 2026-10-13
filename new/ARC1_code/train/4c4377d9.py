import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stack a vertical reflection of the whole input grid above the original grid.

    Input structure:
    - A nonempty rectangular grid whose complete contents form the source tile.

    Rule:
    - Build an output with the same width and twice the height.
    - Put the vertically flipped input in the top half.
    - Put the unchanged input in the bottom half.

    Key insight:
    - The transformation is not object- or color-based; the whole input grid is
      the tile, and the output completes a vertical mirror pattern around the
      join between the two copies.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] > 0 and g.shape[1] > 0, "expected a nonempty grid"

    return np.vstack((np.flipud(g), g))

# --------------- generator (o5x) ---------------
# CFG: grid size bounds, palette constraints, and generation retry limit.
GRID_SIZE_MIN, GRID_SIZE_MAX_EXCL = 2, 7
NON_BLACK_COLORS = np.arange(1, 10)
PALETTE_SIZE = 2
GENERATION_RETRY_LIMIT = 300

def _gen_attempt() -> np.ndarray | None:
    rows = int(np.random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX_EXCL))
    cols = int(np.random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX_EXCL))
    colors = np.random.choice(NON_BLACK_COLORS, size=PALETTE_SIZE, replace=False).astype(int)
    flat = np.random.choice(colors, size=rows * cols, replace=True).astype(int)
    flat[:PALETTE_SIZE] = colors
    np.random.shuffle(flat)
    grid = flat.reshape(rows, cols)
    expected = np.vstack((np.flipud(grid), grid))
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
