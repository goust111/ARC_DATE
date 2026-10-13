from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
BOARD_COLOR = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
BOARD_SIZES = (5, 6)
MAX_GENERATION_ATTEMPTS = 500


def _background_color(grid: np.ndarray) -> int:
    corners = [int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]
    return int(Counter(corners).most_common(1)[0][0])


def _contiguous_band(values: np.ndarray, bg: int) -> list[int]:
    pos = np.flatnonzero(values != bg)
    assert 1 <= pos.size <= 3, "band must contain 1-3 colored cells"
    assert pos[-1] == values.size - 1, "band must sit next to the board gap"
    expected = np.arange(pos[0], pos[-1] + 1)
    assert np.array_equal(pos, expected), "band must be a single contiguous run"
    return [int(x) for x in values[pos]]


def _cell_vote(row_band: list[int], col_band: list[int], default_color: int) -> tuple[int, int]:
    row_last_index: dict[int, int] = {}
    for idx, color in enumerate(row_band):
        row_last_index[int(color)] = idx

    col_colors = set(int(color) for color in col_band)
    common = [color for color in row_last_index if color in col_colors]
    if not common:
        return int(default_color), 0
    if len(common) == 1:
        return int(common[0]), 1
    chosen = max(common, key=lambda color: row_last_index[int(color)])
    return int(chosen), 2


def _render_board(row_bands: list[list[int]], col_bands: list[list[int]], default_color: int) -> tuple[np.ndarray, dict[int, int]]:
    n = len(row_bands)
    assert n == len(col_bands), "board must be square"
    board = np.full((n, n), int(default_color), dtype=int)
    counts = {0: 0, 1: 0, 2: 0}
    for r in range(n):
        for c in range(n):
            color, branch = _cell_vote(row_bands[r], col_bands[c], default_color)
            board[r, c] = int(color)
            counts[int(branch)] += 1
    return board, counts


def _valid_board_candidate(grid: np.ndarray, bg: int, color: int, r0: int, c0: int, r1: int, c1: int) -> bool:
    n = r1 - r0 + 1
    if n < 4 or n != c1 - c0 + 1:
        return False
    if r0 < 2 or c0 < 2:
        return False
    if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color):
        return False
    if not np.all(grid[r0 - 1, c0 : c1 + 1] == bg):
        return False
    if not np.all(grid[r0 : r1 + 1, c0 - 1] == bg):
        return False

    try:
        for r in range(r0, r1 + 1):
            band = _contiguous_band(grid[r, : c0 - 1], bg)
            if any(int(x) == int(color) for x in band):
                return False
        for c in range(c0, c1 + 1):
            band = _contiguous_band(grid[: r0 - 1, c], bg)
            if any(int(x) == int(color) for x in band):
                return False
    except AssertionError:
        return False
    return True


def _find_board(grid: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, int, int, int]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        if _valid_board_candidate(grid, bg, color, int(r0), int(c0), int(r1), int(c1)):
            candidates.append((int(r0), int(c0), int(r1 - r0 + 1), color))
    assert len(candidates) == 1, "expected a unique gray board region"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill a central gray board using row/column color-band intersections.

    Each board cell consults its row band and column band. With no shared color
    it stays gray; with one shared color it takes that color; with multiple
    shared colors it takes the shared color closest to the board on the row band.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background_color(g)
    r0, c0, n, board_color = _find_board(g, bg)

    row_bands = [_contiguous_band(g[r0 + i, : c0 - 1], bg) for i in range(n)]
    col_bands = [_contiguous_band(g[: r0 - 1, c0 + j], bg) for j in range(n)]
    board, _ = _render_board(row_bands, col_bands, board_color)

    out = g.copy()
    out[r0 : r0 + n, c0 : c0 + n] = board
    return out


def _sample_band(active_colors: tuple[int, ...]) -> list[int]:
    length = random.randint(1, 3)
    if length == 1:
        return [random.choice(active_colors)]

    mode = random.random()
    if mode < 0.25:
        color = random.choice(active_colors)
        return [color] * length
    if mode < 0.55:
        a, b = random.sample(active_colors, 2)
        band = [a] * length
        band[random.randrange(length)] = b
        if b not in band:
            band[-1] = b
        return band
    if mode < 0.8 and length <= len(active_colors):
        return random.sample(active_colors, length)

    subset = random.sample(active_colors, 2)
    return [random.choice(subset) for _ in range(length)]


def _bands_are_balanced(counts: dict[int, int], total_cells: int) -> bool:
    values = [int(counts.get(branch, 0)) for branch in (0, 1, 2)]
    need = max(2, total_cells // 6)
    return min(values) >= need and max(values) - min(values) <= total_cells // 2


def _has_duplicate_band(bands: list[list[int]]) -> bool:
    return any(len(set(band)) < len(band) for band in bands)


def _build_input(row_bands: list[list[int]], col_bands: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    n = len(row_bands)
    board, _ = _render_board(row_bands, col_bands, BOARD_COLOR)

    max_row_len = max(len(band) for band in row_bands)
    max_col_len = max(len(band) for band in col_bands)
    top_outer = random.randint(0, 3)
    left_outer = random.randint(0, 3)
    bottom_outer = random.randint(1, 4)
    right_outer = random.randint(1, 4)

    r0 = top_outer + 1 + max_col_len
    c0 = left_outer + 1 + max_row_len
    height = r0 + n + bottom_outer
    width = c0 + n + right_outer

    grid = np.full((height, width), BG, dtype=int)
    grid[r0 : r0 + n, c0 : c0 + n] = BOARD_COLOR

    for i, band in enumerate(row_bands):
        start = c0 - 1 - len(band)
        grid[r0 + i, start : c0 - 1] = np.array(band, dtype=int)

    for j, band in enumerate(col_bands):
        start = r0 - 1 - len(band)
        grid[start : r0 - 1, c0 + j] = np.array(band, dtype=int)

    expected = grid.copy()
    expected[r0 : r0 + n, c0 : c0 + n] = board
    return grid, expected


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(BOARD_SIZES)
    active_size = random.randint(3, 5)
    active_colors = tuple(random.sample(COLOR_POOL, active_size))
    require_duplicate = random.random() < 0.5

    row_bands = [_sample_band(active_colors) for _ in range(n)]
    col_bands = [_sample_band(active_colors) for _ in range(n)]
    board, counts = _render_board(row_bands, col_bands, BOARD_COLOR)

    if not _bands_are_balanced(counts, n * n):
        return None
    if require_duplicate and not (_has_duplicate_band(row_bands) or _has_duplicate_band(col_bands)):
        return None
    if len(set(int(x) for x in board.flatten()) - {BOARD_COLOR}) < 2:
        return None

    grid, expected = _build_input(row_bands, col_bands)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid voting-band puzzle")
