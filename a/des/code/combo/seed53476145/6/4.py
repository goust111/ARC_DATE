import random

import numpy as np


_BG = 0
_COLOR_POOL = tuple(range(1, 10))
_MIN_SIDE = 5
_MAX_SIDE = 15
_MAX_MARKERS = 6
_MAX_ATTEMPTS = 300


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _extract_layout(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], dict[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape
    assert height >= 3 and width >= 3, "expected room for both control strips and an interior"

    bg = _dominant_color(g)
    assert int(g[0, 0]) == bg, "expected the shared control-strip corner to stay background"
    assert bool(np.all(g[1:, 1:] == bg)), "expected the interior to be empty before drawing"

    left_markers = [(r, int(g[r, 0])) for r in range(1, height) if int(g[r, 0]) != bg]
    top_markers = [(c, int(g[0, c])) for c in range(1, width) if int(g[0, c]) != bg]
    assert left_markers and top_markers, "expected colored markers on both control strips"

    left_colors = [color for _r, color in left_markers]
    top_colors = [color for _c, color in top_markers]
    assert len(left_colors) == len(set(left_colors)), "left control colors must be unique"
    assert len(top_colors) == len(set(top_colors)), "top control colors must be unique"
    assert set(left_colors) == set(top_colors), "both control strips must contain the same colors"

    top_by_color = {color: c for c, color in top_markers}
    return bg, left_markers, top_by_color


def _solve_from_layout(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    _bg, left_markers, top_by_color = _extract_layout(g)

    out = g.copy()
    for r, color in left_markers:
        c = top_by_color[color]
        out[r, 1:] = color
        out[1:, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Translate matching top/left color markers into crossing row/column instructions.

    Input structure
    - Background is black.
    - The leftmost column and top row are control strips.
    - Each non-background color appears exactly once in the left strip and once in the top strip.
    - The interior is otherwise empty.

    Rule
    - For each color, match its row from the left strip with its column from the top strip.
    - Draw that color across the full interior part of the matched row.
    - Draw that color down the full interior part of the matched column.
    - Process colors in top-to-bottom order of the left strip so later colors overwrite earlier crossings.
    - Keep the control strips unchanged.
    """
    return _solve_from_layout(grid)


def _build_expected(grid: np.ndarray) -> np.ndarray:
    return _solve_from_layout(grid)


def _sample_dimension() -> int:
    return random.randint(_MIN_SIDE, _MAX_SIDE)


def _sample_marker_count(height: int, width: int) -> int:
    return random.randint(2, min(_MAX_MARKERS, height - 1, width - 1, len(_COLOR_POOL)))


def _gen_attempt() -> np.ndarray | None:
    height = _sample_dimension()
    width = _sample_dimension()
    marker_count = _sample_marker_count(height, width)

    rows = sorted(random.sample(range(1, height), marker_count))
    cols = sorted(random.sample(range(1, width), marker_count))
    left_colors = random.sample(_COLOR_POOL, marker_count)
    top_colors = left_colors[:]
    random.shuffle(top_colors)

    grid = np.full((height, width), _BG, dtype=int)
    for r, color in zip(rows, left_colors):
        grid[r, 0] = color
    for c, color in zip(cols, top_colors):
        grid[0, c] = color

    try:
        actual = main(grid)
        expected = _build_expected(grid)
    except AssertionError:
        return None

    if np.array_equal(actual, expected) and np.any(actual != grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
