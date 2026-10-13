import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile the whole input grid once to the right to make two identical horizontal copies.

    Input structure:
    - A nonempty 2D color grid; all colors, including background, are part of the tile.

    Rule:
    - Preserve the input as the left half of the output.
    - Copy the same full grid into the right half without changing colors or orientation.

    Key insight:
    - The output is a width-doubling repetition of the entire grid, not a per-object
      recolor or local edit.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height > 0 and width > 0, "expected a nonempty grid"

    return np.concatenate([g, g], axis=1)

# --------------- generator (o5x) ---------------
# CFG: np.random.randint ranges use upper-exclusive endpoints.
GRID_HEIGHT_RANGE = (3, 6)
GRID_WIDTH_RANGE = (3, 5)
NONZERO_COLOR_COUNT_RANGE = (2, 6)
NONZERO_COLOR_IDS = np.arange(1, 10)
# Fill density, background color, and rejection retry guard.
FILL_FRACTION_MIN = 0.40
FILL_FRACTION_MAX = 0.70
BACKGROUND_COLOR = 0
GENERATION_ATTEMPTS = 300

def _gen_attempt() -> np.ndarray | None:
    height = int(np.random.randint(*GRID_HEIGHT_RANGE))
    width = int(np.random.randint(*GRID_WIDTH_RANGE))
    cell_count = height * width
    nonzero_color_count = int(np.random.randint(*NONZERO_COLOR_COUNT_RANGE))
    colors = np.random.choice(NONZERO_COLOR_IDS, size=nonzero_color_count, replace=False)
    min_filled = max(nonzero_color_count, int(np.ceil(FILL_FRACTION_MIN * cell_count)))
    max_filled = int(np.floor(FILL_FRACTION_MAX * cell_count))
    filled_count = int(np.random.randint(min_filled, max_filled + 1))

    grid = np.full((height, width), BACKGROUND_COLOR, dtype=int)
    flat_positions = np.random.choice(cell_count, size=filled_count, replace=False)
    values = np.empty(filled_count, dtype=int)
    values[:nonzero_color_count] = colors
    if filled_count > nonzero_color_count:
        values[nonzero_color_count:] = np.random.choice(
            colors, size=filled_count - nonzero_color_count, replace=True
        )
    np.random.shuffle(values)
    grid.flat[flat_positions] = values

    if np.array_equal(grid, grid[:, ::-1]) or np.array_equal(grid, grid[::-1, :]):
        return None

    expected = np.concatenate([grid, grid], axis=1)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GENERATION_ATTEMPTS} attempts")
