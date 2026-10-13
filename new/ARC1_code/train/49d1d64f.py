import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """
    Add a one-cell border by extending each input edge outward while keeping the new corners black.

    Input structure:
    - A nonempty 2D grid of ARC colors.
    - The whole input is the source region; its outer rows and columns define the border to add.

    Rule:
    - Place the input in the center of an output with two extra rows and two extra columns.
    - Copy the top, bottom, left, and right input edges into the adjacent new border cells.
    - Leave the four new corner cells black.

    Key insight:
    - The output is not a scale-up; it is the original grid surrounded by a one-pixel extrusion of its boundary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height > 0 and width > 0, "expected a nonempty grid"

    out = np.zeros((height + 2, width + 2), dtype=int)
    out[1 : height + 1, 1 : width + 1] = g
    out[0, 1 : width + 1] = g[0, :]
    out[height + 1, 1 : width + 1] = g[-1, :]
    out[1 : height + 1, 0] = g[:, 0]
    out[1 : height + 1, width + 1] = g[:, -1]
    return out

# --------------- generator (o5x) ---------------

# Geometry bounds.
HEIGHT_RANGE, WIDTH_RANGE = (2, 6), (2, 6)
# Color sampling.
DISTINCT_COLOR_RANGE = (3, 6)
NONZERO_COLORS = np.arange(1, 10)
BLACK = 0
INCLUDE_BLACK_RANDINT_HIGH = 2
# Retry budget.
MAX_GENERATION_ATTEMPTS = 300


def _gen_expected_output(grid: np.ndarray) -> np.ndarray:
    out = np.zeros((grid.shape[0] + 2, grid.shape[1] + 2), dtype=int)
    out[1:-1, 1:-1] = grid
    out[0, 1:-1], out[-1, 1:-1] = grid[0, :], grid[-1, :]
    out[1:-1, 0], out[1:-1, -1] = grid[:, 0], grid[:, -1]
    return out


def _cell_groups(height, width):
    """Return row-major perimeter cells and row-major interior cells."""
    border = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if r in (0, height - 1) or c in (0, width - 1)
    ]
    interior = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
    return border, interior


def _fill_grid(height, width, border, interior, include_black, nonzero_colors):
    """Populate border first, then optionally reserve one black interior cell."""
    grid = np.empty((height, width), dtype=int)
    np.random.shuffle(border)
    for color, (r, c) in zip(nonzero_colors, border):
        grid[r, c] = int(color)
    for r, c in border[len(nonzero_colors) :]:
        grid[r, c] = int(np.random.choice(nonzero_colors))
    np.random.shuffle(interior)
    fill_colors = np.concatenate(([BLACK], nonzero_colors)) if include_black else nonzero_colors
    if include_black:
        r, c = interior[0]
        grid[r, c] = BLACK
        interior = interior[1:]
    for r, c in interior:
        grid[r, c] = int(np.random.choice(fill_colors))
    return grid


def _gen_attempt() -> np.ndarray | None:
    height = int(np.random.randint(*HEIGHT_RANGE))
    width = int(np.random.randint(*WIDTH_RANGE))
    n_cells = height * width
    n_colors = int(np.random.randint(DISTINCT_COLOR_RANGE[0], min(DISTINCT_COLOR_RANGE[1], n_cells) + 1))
    border, interior = _cell_groups(height, width)
    include_black = bool(interior) and bool(np.random.randint(INCLUDE_BLACK_RANDINT_HIGH))
    n_nonzero = n_colors - int(include_black)
    nonzero_colors = np.random.choice(NONZERO_COLORS, size=n_nonzero, replace=False)
    grid = _fill_grid(height, width, border, interior, include_black, nonzero_colors)

    expected = _gen_expected_output(grid)
    if np.array_equal(main(grid), expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_GENERATION_ATTEMPTS} attempts")
