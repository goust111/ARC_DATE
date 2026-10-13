import numpy as np

from arc_utils import Color, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint the even square-distance rings around the solid foreground cross yellow.

    Input structure:
    - The grid has one background color and one foreground color.
    - The foreground is exactly one full row and one full column crossing each other.
    - Yellow is absent from the input and is the output fill color.

    Rule:
    - Locate the unique solid foreground row and column, whose intersection is the ring center.
    - For every background pixel, compute its Chebyshev distance from that center.
    - Recolor background pixels with even Chebyshev distance to yellow.
    - Keep the foreground cross unchanged.

    Key insight:
    - The output is not drawn from separate quadrant patterns; it is one set of alternating square rings centered on the cross intersection.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert not np.any(g == Color.YELLOW), "yellow must be absent from the input"

    bg = get_bg(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert len(colors) == 1, "expected exactly one foreground color"
    fg = colors[0]

    height, width = map(int, g.shape)
    full_rows = [r for r in range(height) if np.all(g[r, :] == fg)]
    full_cols = [c for c in range(width) if np.all(g[:, c] == fg)]
    assert len(full_rows) == 1, "expected exactly one full foreground row"
    assert len(full_cols) == 1, "expected exactly one full foreground column"

    center_r = full_rows[0]
    center_c = full_cols[0]
    assert 0 < center_r < height - 1, "foreground row must have background above and below"
    assert 0 < center_c < width - 1, "foreground column must have background left and right"

    rows, cols = np.indices(g.shape)
    cross_mask = (rows == center_r) | (cols == center_c)
    assert np.array_equal(g == fg, cross_mask), "foreground must be exactly the solid cross"

    distance = np.maximum(np.abs(rows - center_r), np.abs(cols - center_c))
    out = g.copy()
    out[(g == bg) & (distance % 2 == 0)] = Color.YELLOW
    return out

# --------------- generator (o5x) ---------------
# Canvas and cross geometry.
GRID_SIZE_MIN, GRID_SIZE_MAX = 10, 20
CENTER_MARGIN = 3
# Fixed task colors.
BACKGROUND_COLOR, FOREGROUND_COLOR, OUTPUT_RING_COLOR = Color.BLACK, Color.GREEN, Color.YELLOW
# Retry budget for oracle-checked samples.
MAX_ATTEMPTS = 300
def _gen_expected_output(grid: np.ndarray, center_r: int, center_c: int) -> np.ndarray:
    rows, cols = np.indices(grid.shape)
    distance = np.maximum(np.abs(rows - center_r), np.abs(cols - center_c))
    out = grid.copy()
    out[(grid == BACKGROUND_COLOR) & (distance % 2 == 0)] = OUTPUT_RING_COLOR
    return out


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX + 1))
    center_r = int(np.random.randint(CENTER_MARGIN, n - CENTER_MARGIN))
    center_c = int(np.random.randint(CENTER_MARGIN, n - CENTER_MARGIN))

    grid = np.full((n, n), BACKGROUND_COLOR, dtype=int)
    grid[center_r, :] = FOREGROUND_COLOR
    grid[:, center_c] = FOREGROUND_COLOR

    expected = _gen_expected_output(grid, center_r, center_c)
    try:
        if np.array_equal(main(grid), expected):
            return grid
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for 759f3fd3")
