import numpy as np

from arc_utils import get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the sparse lattice by deleting the all-background separator rows and columns.

    Input structure
    - The grid has even height and width.
    - Every odd-indexed row and every odd-indexed column is background.
    - The meaningful colored pixels occupy the even-row/even-column lattice positions.

    Rule
    - Keep the cells at even row and even column indices, preserving their colors.
    - The output height and width are half of the input height and width.

    Key insight
    - The odd rows and columns are separators between logical cells; removing them reveals the compact symbol grid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    height, width = g.shape
    assert height % 2 == 0 and width % 2 == 0, "expected even grid dimensions"

    bg = get_bg(g)
    assert np.all(g[1::2, :] == bg), "odd rows must be separator rows"
    assert np.all(g[:, 1::2] == bg), "odd columns must be separator columns"

    return g[::2, ::2].copy()

# --------------- generator (o5x) ---------------

# Logical grid, colors, rendering, and retry bounds.
LOGICAL_SIZE_MIN, LOGICAL_SIZE_STOP = 2, 6
NON_BG_COUNT_MIN, NON_BG_COUNT_MAX = 4, 7
DISTINCT_COLOR_MIN, DISTINCT_COLOR_MAX = 3, 4
COLOR_CHOICES = np.arange(1, 10)
BACKGROUND, LATTICE_STRIDE, MAX_ATTEMPTS = 0, 2, 300


def _sample_logical_grid(rng: np.random.Generator) -> np.ndarray:
    """Return compact colored cells before separator rows/columns are inserted."""
    logical_h = int(rng.integers(LOGICAL_SIZE_MIN, LOGICAL_SIZE_STOP))
    logical_w = int(rng.integers(LOGICAL_SIZE_MIN, LOGICAL_SIZE_STOP))
    capacity = logical_h * logical_w
    non_bg_count = int(rng.integers(NON_BG_COUNT_MIN, min(NON_BG_COUNT_MAX, capacity) + 1))
    distinct_count = int(rng.integers(DISTINCT_COLOR_MIN, min(DISTINCT_COLOR_MAX, non_bg_count) + 1))

    colors = rng.choice(COLOR_CHOICES, size=distinct_count, replace=False)
    pixel_colors = np.empty(non_bg_count, dtype=int)
    pixel_colors[:distinct_count] = colors
    if non_bg_count > distinct_count:
        extra_count = non_bg_count - distinct_count
        pixel_colors[distinct_count:] = rng.choice(colors, size=extra_count, replace=True)
    rng.shuffle(pixel_colors)

    logical = np.zeros((logical_h, logical_w), dtype=int)
    logical.flat[rng.choice(capacity, size=non_bg_count, replace=False)] = pixel_colors
    return logical


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    logical = _sample_logical_grid(rng)
    inp = np.full(tuple(dim * LATTICE_STRIDE for dim in logical.shape), BACKGROUND, dtype=int)
    inp[::LATTICE_STRIDE, ::LATTICE_STRIDE] = logical
    return inp if np.array_equal(main(inp), logical) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 68b67ca3 input")
