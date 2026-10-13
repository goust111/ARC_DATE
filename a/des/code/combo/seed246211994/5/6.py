import random

import numpy as np


BLACK = 0
BLUE = 1
TEAL = 8
MAROON = 9

EDGES = ("top", "bottom", "left", "right")
GRID_SIZE_RANGE = (4, 10)
MAX_GENERATION_ATTEMPTS = 300


def _find_marker(grid: np.ndarray) -> tuple[int, int]:
    cells = np.argwhere(grid == TEAL)
    assert cells.shape == (1, 2), "expected exactly one teal marker"
    return int(cells[0, 0]), int(cells[0, 1])


def _validate_input(grid: np.ndarray, marker_row: int, marker_col: int) -> None:
    assert grid.ndim == 2
    height, width = grid.shape
    assert height >= 3 and width >= 3

    on_top = marker_row == 0
    on_bottom = marker_row == height - 1
    on_left = marker_col == 0
    on_right = marker_col == width - 1
    assert sum([on_top, on_bottom, on_left, on_right]) == 1, "marker must be on one non-corner edge"

    marker_parity = (marker_row + marker_col) % 2
    for r in range(height):
        for c in range(width):
            value = int(grid[r, c])
            if r == marker_row and c == marker_col:
                continue
            expected = BLUE if (r + c) % 2 == marker_parity else BLACK
            assert value == expected, "grid must be a blue/black checkerboard with teal replacing one blue cell"


def _packed_positions(length: int, count: int, toward_start: bool) -> list[int]:
    return list(range(count)) if toward_start else list(range(length - count, length))


def _trail_indices(sources: list[int], final_positions: list[int], marker_index: int) -> set[int]:
    assert len(sources) == len(final_positions)
    final_set = set(final_positions)
    swept = set()
    for src, dst in zip(sources, final_positions):
        if src == marker_index:
            assert dst == marker_index
            continue
        swept.update(range(min(src, dst), max(src, dst) + 1))
    return swept - final_set


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    marker_row, marker_col = _find_marker(g)
    _validate_input(g, marker_row, marker_col)

    height, width = g.shape
    out = np.full_like(g, BLACK)

    if marker_col == 0 or marker_col == width - 1:
        to_left = marker_col == 0
        for r in range(height):
            count = int(np.count_nonzero(g[r, :] == BLUE))
            if r == marker_row:
                count += 1
            out[r, _packed_positions(width, count, to_left)] = BLUE

        sources = [c for c in range(width) if int(g[marker_row, c]) == BLUE or c == marker_col]
        finals = _packed_positions(width, len(sources), to_left)
        for c in _trail_indices(sources, finals, marker_col):
            out[marker_row, c] = MAROON
    else:
        to_top = marker_row == 0
        for c in range(width):
            count = int(np.count_nonzero(g[:, c] == BLUE))
            if c == marker_col:
                count += 1
            out[_packed_positions(height, count, to_top), c] = BLUE

        sources = [r for r in range(height) if int(g[r, marker_col]) == BLUE or r == marker_row]
        finals = _packed_positions(height, len(sources), to_top)
        for r in _trail_indices(sources, finals, marker_row):
            out[r, marker_col] = MAROON

    out[marker_row, marker_col] = TEAL
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Compact the checkerboard's blue parity toward the teal marker edge."""
    return _transform(grid)


def _choose_marker(edge: str, height: int, width: int) -> tuple[int, int]:
    if edge == "top":
        return 0, random.randint(1, width - 2)
    if edge == "bottom":
        return height - 1, random.randint(1, width - 2)
    if edge == "left":
        return random.randint(1, height - 2), 0
    assert edge == "right"
    return random.randint(1, height - 2), width - 1


def _make_checkerboard(height: int, width: int, marker_row: int, marker_col: int) -> np.ndarray:
    marker_parity = (marker_row + marker_col) % 2
    rows, cols = np.indices((height, width))
    grid = np.where((rows + cols) % 2 == marker_parity, BLUE, BLACK).astype(int)
    grid[marker_row, marker_col] = TEAL
    return grid


def _expected_output(grid: np.ndarray, marker_row: int, marker_col: int) -> np.ndarray:
    height, width = grid.shape
    out = np.full_like(grid, BLACK)

    if marker_col == 0 or marker_col == width - 1:
        to_left = marker_col == 0
        counts = np.count_nonzero(grid == BLUE, axis=1)
        counts[marker_row] += 1
        for r in range(height):
            out[r, _packed_positions(width, int(counts[r]), to_left)] = BLUE

        sources = [c for c, value in enumerate(grid[marker_row, :]) if int(value) == BLUE or c == marker_col]
        finals = _packed_positions(width, len(sources), to_left)
        for c in _trail_indices(sources, finals, marker_col):
            out[marker_row, c] = MAROON
    else:
        to_top = marker_row == 0
        counts = np.count_nonzero(grid == BLUE, axis=0)
        counts[marker_col] += 1
        for c in range(width):
            out[_packed_positions(height, int(counts[c]), to_top), c] = BLUE

        sources = [r for r, value in enumerate(grid[:, marker_col]) if int(value) == BLUE or r == marker_row]
        finals = _packed_positions(height, len(sources), to_top)
        for r in _trail_indices(sources, finals, marker_row):
            out[r, marker_col] = MAROON

    out[marker_row, marker_col] = TEAL
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_SIZE_RANGE)
    width = random.randint(*GRID_SIZE_RANGE)
    edge = random.choice(EDGES)
    marker_row, marker_col = _choose_marker(edge, height, width)
    grid = _make_checkerboard(height, width, marker_row, marker_col)

    try:
        _validate_input(grid, marker_row, marker_col)
    except AssertionError:
        return None

    expected = _expected_output(grid, marker_row, marker_col)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
