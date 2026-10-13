from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop out the separator cross and return the quadrant containing the single marker pixel.

    Input structure
    - The grid has a dominant background color and exactly two foreground colors.
    - One foreground color forms a unique solid row and solid column of the same
      color, splitting the grid into four nonempty regions.
    - The other foreground color appears as one marker pixel in one of those
      regions.

    Rule
    - Find the unique same-color solid row/column separator cross.
    - Select the region on the same side of the separator row and column as the
      marker pixel.
    - Return that region unchanged, which removes the separator cross and keeps
      the marker at its local position.

    Key insight
    - The cross is not part of the output pattern; it is a divider telling which
      quadrant-sized crop to keep.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)

    bg = get_bg(g)
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 2, "expected separator and marker foreground colors"

    solid_rows = [
        (row, color)
        for row in range(height)
        for color in fg_colors
        if np.all(g[row, :] == color)
    ]
    solid_cols = [
        (col, color)
        for col in range(width)
        for color in fg_colors
        if np.all(g[:, col] == color)
    ]
    crosses = [
        (row, col, row_color)
        for row, row_color in solid_rows
        for col, col_color in solid_cols
        if row_color == col_color
    ]
    assert len(crosses) == 1, "expected exactly one solid same-color separator cross"

    cross_row, cross_col, separator = crosses[0]
    assert 0 < cross_row < height - 1, "separator row must split nonempty regions"
    assert 0 < cross_col < width - 1, "separator column must split nonempty regions"

    marker_colors = [color for color in fg_colors if color != separator]
    assert len(marker_colors) == 1, "expected one marker color"
    marker_color = marker_colors[0]
    marker_coords = np.argwhere(g == marker_color)
    assert marker_coords.shape == (1, 2), "expected one marker pixel"
    marker_row, marker_col = map(int, marker_coords[0])
    assert marker_row != cross_row and marker_col != cross_col, "marker must lie in a quadrant"

    row_slice = slice(0, cross_row) if marker_row < cross_row else slice(cross_row + 1, height)
    col_slice = slice(0, cross_col) if marker_col < cross_col else slice(cross_col + 1, width)
    out = g[row_slice, col_slice].copy()
    assert out.size > 0, "selected region must be nonempty"
    assert set(map(int, out.reshape(-1))) <= {bg, marker_color}, "selected region should contain only background and marker"

    return out

# --------------- generator (o5x) ---------------
# Canvas/color bounds.
N_MIN, N_MAX_EXCLUSIVE = 2, 9
ARC_COLOR_COUNT, COLOR_ROLE_COUNT = 10, 3
# Layout and retry controls.
QUADRANT_GRID_WIDTH, SEPARATOR_WIDTH = 2, 1
QUADRANT_COUNT, GENERATION_RETRIES = 4, 300

def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.randint(N_MIN, N_MAX_EXCLUSIVE))
    bg, separator, marker = map(int, np.random.choice(ARC_COLOR_COUNT, size=COLOR_ROLE_COUNT, replace=False))
    size = QUADRANT_GRID_WIDTH * n + SEPARATOR_WIDTH

    grid = np.full((size, size), bg, dtype=int)
    grid[n, :] = separator
    grid[:, n] = separator

    quadrant = int(np.random.randint(QUADRANT_COUNT))
    row0 = 0 if quadrant < QUADRANT_GRID_WIDTH else n + SEPARATOR_WIDTH
    col0 = 0 if quadrant % QUADRANT_GRID_WIDTH == 0 else n + SEPARATOR_WIDTH
    local_r = int(np.random.randint(n))
    local_c = int(np.random.randint(n))
    grid[row0 + local_r, col0 + local_c] = marker

    expected = np.full((n, n), bg, dtype=int)
    expected[local_r, local_c] = marker
    if np.array_equal(main(grid), expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 2dc579da input")
