import random

import numpy as np


PINK = 6
TEAL = 8
ALL_COLORS = list(range(10))
MAX_GENERATION_ATTEMPTS = 300


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def main(grid: np.ndarray) -> np.ndarray:
    """Convert each pink control pixel into a teal horizontal bar based on row-side counts."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    assert bg not in (PINK, TEAL)

    markers = np.argwhere(g == PINK)
    assert len(markers) > 0

    out = np.full(g.shape, bg, dtype=int)
    for r, c in markers:
        assert np.count_nonzero(g[r] == PINK) == 1
        left_count = int(np.count_nonzero(g[r, :c] != bg))
        right_count = int(np.count_nonzero(g[r, c + 1 :] != bg))
        start = int(c) - left_count
        end = int(c) + right_count
        assert 0 <= start <= c <= end < g.shape[1]
        out[int(r), start : end + 1] = TEAL

    return out


def _counter_palette(bg: int) -> list[int]:
    return [color for color in ALL_COLORS if color not in (bg, PINK, TEAL)]


def _sample_row_spec(width: int) -> tuple[int, int, int]:
    pattern = random.choice(("marker_only", "left_only", "right_only", "both"))

    if pattern == "marker_only":
        col = random.randrange(width)
        return col, 0, 0

    if pattern == "left_only":
        left_count = random.randint(1, min(4, width - 1))
        col = random.randint(left_count, width - 1)
        return col, left_count, 0

    if pattern == "right_only":
        right_count = random.randint(1, min(4, width - 1))
        col = random.randint(0, width - right_count - 1)
        return col, 0, right_count

    left_cap = min(4, width - 2)
    left_count = random.randint(1, left_cap)
    right_cap = min(4, width - left_count - 1)
    right_count = random.randint(1, right_cap)
    col = random.randint(left_count, width - right_count - 1)
    return col, left_count, right_count


def _paint_marker_row(grid: np.ndarray, row: int, bg: int, col: int, left_count: int, right_count: int) -> None:
    palette = _counter_palette(bg)
    grid[row, col] = PINK
    if left_count:
        for cc in random.sample(list(range(col)), left_count):
            grid[row, cc] = random.choice(palette)
    if right_count:
        for cc in random.sample(list(range(col + 1, grid.shape[1])), right_count):
            grid[row, cc] = random.choice(palette)


def _paint_distractors(grid: np.ndarray, marker_rows: set[int], bg: int) -> None:
    palette = _counter_palette(bg)
    for row in range(grid.shape[0]):
        if row in marker_rows:
            continue
        max_count = min(3, grid.shape[1])
        count = random.randint(0, max_count)
        if count == 0:
            continue
        for col in random.sample(list(range(grid.shape[1])), count):
            grid[row, col] = random.choice(palette)


def _expected_output(grid: np.ndarray, specs: list[tuple[int, int, int, int]], bg: int) -> np.ndarray:
    out = np.full(grid.shape, bg, dtype=int)
    for row, col, left_count, right_count in specs:
        out[row, col - left_count : col + right_count + 1] = TEAL
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(7, 16)
    width = random.randint(7, 16)
    bg = random.choice([color for color in ALL_COLORS if color not in (PINK, TEAL)])
    grid = np.full((height, width), bg, dtype=int)

    marker_count = random.randint(1, min(6, height))
    marker_rows = sorted(random.sample(list(range(height)), marker_count))
    specs: list[tuple[int, int, int, int]] = []
    for row in marker_rows:
        col, left_count, right_count = _sample_row_spec(width)
        specs.append((row, col, left_count, right_count))
        _paint_marker_row(grid, row, bg, col, left_count, right_count)

    _paint_distractors(grid, set(marker_rows), bg)

    # Keep the background uniquely dominant so the solver can identify it from counts.
    bg_count = int(np.count_nonzero(grid == bg))
    color_counts = [int(np.count_nonzero(grid == color)) for color in ALL_COLORS if color != bg]
    if bg_count <= max(color_counts, default=0):
        return None

    expected = _expected_output(grid, specs, bg)
    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
