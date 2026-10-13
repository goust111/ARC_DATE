import numpy as np

from arc_utils import Color, get_bg


UPWARD_DISTANCE = {
    Color.RED: 4,
    Color.TEAL: 3,
}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each red or teal marker upward from the blue row and connect it back with blue cells.

    Input structure
    - The background is the dominant empty color.
    - All foreground pixels lie in one full-width horizontal row.
    - That row is blue except for red and/or teal marker pixels.

    Rule
    - A red marker is copied four rows above itself; a teal marker is copied three rows above itself.
    - Fill the intervening cells in the same column with blue, leaving the original row unchanged.

    Key insight
    - The marker color encodes the upward distance of the vertical stem.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    foreground_rows = np.flatnonzero(np.any(g != bg, axis=1))
    assert foreground_rows.size == 1, "expected exactly one foreground row"
    row = int(foreground_rows[0])
    assert np.all(g[row] != bg), "foreground row must span the full width"
    assert Color.BLUE in set(map(int, g[row])), "foreground row must include blue cells"

    markers: list[tuple[int, int]] = []
    for col, color in enumerate(map(int, g[row])):
        if color == Color.BLUE:
            continue
        assert color in UPWARD_DISTANCE, "expected only red or teal markers in the blue row"
        markers.append((col, color))
    assert markers, "expected at least one marker"

    for col, color in markers:
        distance = UPWARD_DISTANCE[color]
        top_row = row - distance
        assert top_row >= 0, "marker must have enough empty space above it"
        assert np.all(g[top_row:row, col] == bg), "stem path above marker must be empty"

        out[top_row, col] = color
        out[top_row + 1 : row, col] = Color.BLUE

    return out

# --------------- generator (o5x) ---------------

# Canvas, layout, and retry bounds.
ROW_RANGE, COL_RANGE = (9, 14), (6, 12)
ROWS_BELOW_RANGE, MARKER_COUNT_RANGE = (1, 5), (1, 5)
MAX_GENERATION_ATTEMPTS = 300
# Marker colors and stem geometry.
MARKER_DISTANCES = {Color.RED: 4, Color.TEAL: 3}
MARKER_COLORS = np.array(tuple(MARKER_DISTANCES), dtype=int)


def _sample_layout() -> tuple[int, int, int, np.ndarray, np.ndarray]:
    """Return shape, blue row, marker columns, and marker colors for one case."""
    rows = int(np.random.randint(*ROW_RANGE))
    cols = int(np.random.randint(*COL_RANGE))
    max_rows_below = min(ROWS_BELOW_RANGE[1], rows - 5)
    blue_row = rows - 1 - int(np.random.randint(ROWS_BELOW_RANGE[0], max_rows_below + 1))
    marker_count = int(np.random.randint(*MARKER_COUNT_RANGE))
    marker_cols = np.random.choice(np.arange(1, cols - 1), size=marker_count, replace=False)
    marker_colors = np.random.choice(MARKER_COLORS, size=marker_count, replace=True)
    return rows, cols, blue_row, marker_cols, marker_colors


def _gen_attempt() -> np.ndarray | None:
    rows, cols, blue_row, marker_cols, marker_colors = _sample_layout()
    out = np.zeros((rows, cols), dtype=int)
    out[blue_row, :] = Color.BLUE
    for col, color in zip(marker_cols, marker_colors):
        color = int(color)
        top_row = blue_row - MARKER_DISTANCES[color]
        out[top_row, col] = color
        out[top_row + 1 : blue_row, col] = Color.BLUE
        out[blue_row, col] = color
    inp = out.copy()
    inp[:blue_row, :] = Color.BLACK
    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 72a961c9 input")
