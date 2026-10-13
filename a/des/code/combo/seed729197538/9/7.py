import random
from collections import Counter

import numpy as np


BG_COLOR = 0
PREFERRED_OVERLAP_COLORS = (3, 2, 6, 8, 4, 5, 7, 1, 9, 0)

# Generator configuration.
GRID_SIDE_RANGE = (15, 25)
MIN_STRIPES_PER_AXIS = 2
MAX_STRIPES_PER_AXIS = 6
MAX_COVERAGE_RATIO = 0.45
GEN_COLOR_POOL = (1, 2, 4, 5, 6, 7, 8, 9)
GEN_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    corners = [
        int(grid[0, 0]),
        int(grid[0, -1]),
        int(grid[-1, 0]),
        int(grid[-1, -1]),
    ]
    corner_counts = Counter(corners)
    color, freq = corner_counts.most_common(1)[0]
    if freq >= 2:
        return int(color)
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _pick_overlap_color(used_colors: set[int]) -> int:
    # Prefer green when it is available, matching the puzzle description.
    for color in PREFERRED_OVERLAP_COLORS:
        if color not in used_colors:
            return int(color)
    raise AssertionError("no spare color available for overlap")


def _stripe_axes(grid: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray]:
    horizontal_rows = np.flatnonzero(np.all(grid != bg, axis=1))
    vertical_cols = np.flatnonzero(np.all(grid != bg, axis=0))
    return horizontal_rows, vertical_cols


def _orientation_colors(
    grid: np.ndarray, bg: int, horizontal_rows: np.ndarray, vertical_cols: np.ndarray
) -> tuple[int, int]:
    colors = [int(color) for color in np.unique(grid) if int(color) != int(bg)]
    assert len(colors) == 2, "expected exactly two stripe colors in the input"

    row_counts = {
        color: int(np.count_nonzero(grid[horizontal_rows] == color)) for color in colors
    }
    col_counts = {
        color: int(np.count_nonzero(grid[:, vertical_cols] == color)) for color in colors
    }

    horizontal_color = max(colors, key=lambda color: (row_counts[color], -col_counts[color]))
    vertical_color = max(colors, key=lambda color: (col_counts[color], -row_counts[color]))

    if horizontal_color == vertical_color:
        horizontal_color = max(colors, key=lambda color: row_counts[color] - col_counts[color])
        vertical_color = next(color for color in colors if color != horizontal_color)

    assert horizontal_color != vertical_color
    return int(horizontal_color), int(vertical_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flatten a woven stripe picture into complete horizontal and vertical stripes.

    Input structure
    - Black background with several one-cell horizontal stripes and several one-cell
      vertical stripes.
    - Every horizontal stripe runs across the full width; every vertical stripe runs
      across the full height.
    - At intersections, the visible color alternates in a weaving pattern.

    Rule
    - Recover the full horizontal stripes and the full vertical stripes.
    - Paint every true overlap cell with a third color.
    - Leave the background unchanged.

    Key insight
    - A horizontal stripe row has no background anywhere, and a vertical stripe
      column has no background anywhere, even though the visible color at
      intersections alternates.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    horizontal_rows, vertical_cols = _stripe_axes(g, bg)
    assert len(horizontal_rows) > 0, "expected at least one horizontal stripe"
    assert len(vertical_cols) > 0, "expected at least one vertical stripe"

    horizontal_color, vertical_color = _orientation_colors(
        g, bg, horizontal_rows, vertical_cols
    )
    overlap_color = _pick_overlap_color({int(bg), int(horizontal_color), int(vertical_color)})

    out = np.full_like(g, int(bg))
    out[horizontal_rows, :] = int(horizontal_color)
    out[:, vertical_cols] = int(vertical_color)
    out[np.ix_(horizontal_rows, vertical_cols)] = int(overlap_color)
    return out


def _sample_non_adjacent(start: int, stop: int, count: int) -> list[int] | None:
    length = int(stop) - int(start) + 1
    if count < 0 or length <= 0 or count > (length + 1) // 2:
        return None
    base = sorted(random.sample(range(length - count + 1), count))
    return [int(start + value + offset) for offset, value in enumerate(base)]


def _count_options(height: int, width: int) -> list[tuple[int, int]]:
    inner_rows = height - 2
    inner_cols = width - 2
    max_h = min(MAX_STRIPES_PER_AXIS, (inner_rows + 1) // 2)
    max_v = min(MAX_STRIPES_PER_AXIS, (inner_cols + 1) // 2)

    options: list[tuple[int, int]] = []
    for h_count in range(MIN_STRIPES_PER_AXIS, max_h + 1):
        for v_count in range(MIN_STRIPES_PER_AXIS, max_v + 1):
            covered = h_count * width + v_count * height - h_count * v_count
            if covered <= int(height * width * MAX_COVERAGE_RATIO):
                options.append((h_count, v_count))
    return options


def _render_input(
    height: int,
    width: int,
    horizontal_rows: list[int],
    vertical_cols: list[int],
    horizontal_color: int,
    vertical_color: int,
    phase: int,
) -> np.ndarray:
    grid = np.full((height, width), BG_COLOR, dtype=int)
    grid[horizontal_rows, :] = int(horizontal_color)
    grid[:, vertical_cols] = int(vertical_color)

    for i, row in enumerate(horizontal_rows):
        for j, col in enumerate(vertical_cols):
            if (i + j + phase) % 2 == 0:
                grid[row, col] = int(horizontal_color)
            else:
                grid[row, col] = int(vertical_color)

    return grid


def _render_output(
    height: int,
    width: int,
    horizontal_rows: list[int],
    vertical_cols: list[int],
    horizontal_color: int,
    vertical_color: int,
) -> np.ndarray:
    overlap_color = _pick_overlap_color(
        {BG_COLOR, int(horizontal_color), int(vertical_color)}
    )
    out = np.full((height, width), BG_COLOR, dtype=int)
    out[horizontal_rows, :] = int(horizontal_color)
    out[:, vertical_cols] = int(vertical_color)
    out[np.ix_(horizontal_rows, vertical_cols)] = int(overlap_color)
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_SIDE_RANGE)
    width = random.randint(*GRID_SIDE_RANGE)
    options = _count_options(height, width)
    if not options:
        return None

    h_count, v_count = random.choice(options)
    horizontal_rows = _sample_non_adjacent(1, height - 2, h_count)
    vertical_cols = _sample_non_adjacent(1, width - 2, v_count)
    if horizontal_rows is None or vertical_cols is None:
        return None

    horizontal_color, vertical_color = random.sample(GEN_COLOR_POOL, 2)
    phase = random.randint(0, 1)

    inp = _render_input(
        height,
        width,
        horizontal_rows,
        vertical_cols,
        horizontal_color,
        vertical_color,
        phase,
    )
    expected = _render_output(
        height,
        width,
        horizontal_rows,
        vertical_cols,
        horizontal_color,
        vertical_color,
    )

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid woven-stripes input")
