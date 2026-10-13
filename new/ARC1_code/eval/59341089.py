import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """
    Create a horizontally mirrored two-block pattern by repeating each row as reversed-then-original twice.

    Input structure
    - A non-empty 2D color grid.

    Rule
    - Mirror the whole grid left-to-right.
    - Concatenate the mirrored grid, the original grid, the mirrored grid, and the original grid.

    Key insight
    - The output is a horizontal periodic pattern whose tile is the input preceded by its left-right reflection.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    assert g.size > 0, "input grid must be non-empty"

    tile = np.concatenate((np.fliplr(g), g), axis=1)
    out = np.concatenate((tile, tile), axis=1)
    assert out.shape == (g.shape[0], g.shape[1] * 4)
    return out

# --------------- generator (o5x) ---------------

GRID_SIZE_RANGE = (2, 6)  # rows/cols, NumPy high endpoint is exclusive
COLOR_COUNT_RANGE = (2, 4)  # selected non-black colors, high exclusive
NON_BLACK_COLORS = np.arange(1, 10)
MAX_GENERATION_ATTEMPTS = 300


def _sample_grid(rng: np.random.Generator) -> np.ndarray:
    """Sample a random grid containing every selected color at least once."""
    rows = int(rng.integers(*GRID_SIZE_RANGE))
    cols = int(rng.integers(*GRID_SIZE_RANGE))
    num_colors = int(rng.integers(*COLOR_COUNT_RANGE))
    colors = rng.choice(NON_BLACK_COLORS, size=num_colors, replace=False)
    flat = rng.choice(colors, size=rows * cols, replace=True)
    flat[rng.choice(rows * cols, size=num_colors, replace=False)] = colors
    return flat.reshape(rows, cols).astype(int)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    grid = _sample_grid(rng)
    if np.array_equal(grid, np.fliplr(grid)):
        return None
    out = main(grid)
    expected = np.concatenate((np.fliplr(grid), grid, np.fliplr(grid), grid), axis=1)
    if not np.array_equal(out, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid non-symmetric input")
