import numpy as np

from arc_utils import Color, colors_present, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """Mark background cells in foreground-bearing columns with teal, then repeat the marked grid as a 2x2 tile.

    Input structure
    - The input is a nonempty 2D grid with one dominant background color.
    - Exactly one non-background foreground color appears, and it is not teal.

    Rule
    - Preserve every original foreground pixel.
    - In any column containing at least one foreground pixel, recolor background
      cells to teal.
    - Repeat the resulting grid twice vertically and twice horizontally.

    Key insight
    - The foreground pixels select whole columns of the tile; the output is a
      repeated copy of that completed tile, not a nearest-neighbor scale-up.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    bg = get_bg(g)
    foreground_colors = colors_present(g, exclude={bg})
    assert len(foreground_colors) == 1, "expected exactly one foreground color"
    assert (
        bg != Color.TEAL and foreground_colors[0] != Color.TEAL
    ), "teal is reserved for the new marks"

    foreground_mask = g != bg
    foreground_cols = np.any(foreground_mask, axis=0)
    assert np.any(foreground_cols), "expected at least one foreground-bearing column"

    tile = g.copy()
    tile[(g == bg) & foreground_cols[np.newaxis, :]] = Color.TEAL
    return np.tile(tile, (2, 2))

# --------------- generator (o5x) ---------------
SIZE_MIN, SIZE_MAX_EXCLUSIVE = 2, 7  # Square canvas side bounds.
MAX_FOREGROUND_PIXELS, GENERATION_RETRIES = 3, 300  # Sparse count cap and retry limit.
BACKGROUND_COLOR, MARK_COLOR, TILE_REPEATS = 0, 8, (2, 2)  # Marked tile rendering.
FOREGROUND_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)  # Excludes bg and teal.


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    size = int(rng.integers(SIZE_MIN, SIZE_MAX_EXCLUSIVE))
    color = int(rng.choice(FOREGROUND_COLORS))
    k = int(rng.integers(1, min(MAX_FOREGROUND_PIXELS, size - 1) + 1))

    grid = np.zeros((size, size), dtype=int)
    cols = rng.choice(size, size=k, replace=False)
    rows = rng.integers(0, size, size=k)
    grid[rows, cols] = color

    selected_cols = np.any(grid != BACKGROUND_COLOR, axis=0)
    tile = grid.copy()
    tile[(grid == BACKGROUND_COLOR) & selected_cols[np.newaxis, :]] = MARK_COLOR
    expected = np.tile(tile, TILE_REPEATS)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid f5b8619d input")
