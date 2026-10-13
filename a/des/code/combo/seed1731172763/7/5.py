from __future__ import annotations

import random
from functools import lru_cache

import numpy as np


GREY = 5
MAROON = 9
INPUT_COLORS = (1, 2, 3, 4, 6, 7, 8)
GROUP_SIZES = (2, 3, 4)
MIN_SIZE = 5
MAX_SIZE = 12
MAX_ATTEMPTS = 300


def _singleton_index_and_color(values: np.ndarray) -> tuple[int, int]:
    unique, counts = np.unique(values, return_counts=True)
    singletons = unique[counts == 1]
    assert len(singletons) == 1, "each row/column must have exactly one singleton color"
    color = int(singletons[0])
    indices = np.flatnonzero(values == color)
    assert len(indices) == 1, "singleton color must occupy exactly one cell"
    return int(indices[0]), color


def _marker_info(grid: np.ndarray) -> tuple[list[int], list[int], list[int], list[int]]:
    n = grid.shape[0]
    row_cols: list[int] = []
    row_colors: list[int] = []
    for r in range(n):
        c, color = _singleton_index_and_color(grid[r, :])
        row_cols.append(c)
        row_colors.append(color)

    col_rows: list[int] = []
    col_colors: list[int] = []
    for c in range(n):
        r, color = _singleton_index_and_color(grid[:, c])
        col_rows.append(r)
        col_colors.append(color)

    return row_cols, row_colors, col_rows, col_colors


def main(grid: np.ndarray) -> np.ndarray:
    """Mark the intersections between row-singleton and column-singleton cells."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    n, m = g.shape
    assert n == m, "expected an NxN grid"

    row_cols, row_colors, col_rows, col_colors = _marker_info(g)
    row_marker_coords = {(r, c) for r, c in enumerate(row_cols)}

    out = np.full((n, n), GREY, dtype=int)
    for r, c in enumerate(row_cols):
        out[r, c] = row_colors[r]

    for c, r in enumerate(col_rows):
        out[r, c] = MAROON if (r, c) in row_marker_coords else col_colors[c]

    return out


@lru_cache(maxsize=None)
def _partitions(total: int, min_part: int = 2) -> tuple[tuple[int, ...], ...]:
    parts: list[tuple[int, ...]] = []
    for size in GROUP_SIZES:
        if size < min_part or size > total:
            continue
        if size == total:
            parts.append((size,))
            continue
        for rest in _partitions(total - size, size):
            parts.append((size,) + rest)
    return tuple(parts)


@lru_cache(maxsize=None)
def _valid_partitions(total: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        part
        for part in _partitions(total)
        if 2 in part and any(size >= 3 for size in part)
    )


def _row_groups(n: int, sizes: tuple[int, ...]) -> list[list[int]]:
    rows = list(range(n))
    random.shuffle(rows)
    groups: list[list[int]] = []
    start = 0
    for size in sizes:
        group = rows[start : start + size]
        random.shuffle(group)
        groups.append(group)
        start += size
    return groups


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(MIN_SIZE, MAX_SIZE)
    partitions = _valid_partitions(n)
    if not partitions:
        return None

    sizes = random.choice(partitions)
    groups = _row_groups(n, sizes)
    if len(groups) >= len(INPUT_COLORS):
        return None

    base_colors = random.sample(INPUT_COLORS, k=len(groups))
    large_group_ids = [i for i, size in enumerate(sizes) if size >= 3]
    support_pool = [base_colors[i] for i in large_group_ids]
    accent_pool = [color for color in INPUT_COLORS if color not in base_colors]
    if not support_pool or not accent_pool:
        return None

    row_base = {}
    row_size = {}
    for group_id, group in enumerate(groups):
        for row in group:
            row_base[row] = base_colors[group_id]
            row_size[row] = sizes[group_id]

    grid = np.empty((n, n), dtype=int)
    for row in range(n):
        grid[row, :] = row_base[row]

    row_unique_cols = random.sample(range(n), k=n)
    for row, col in enumerate(row_unique_cols):
        if row_size[row] == 2:
            grid[row, col] = random.choice(support_pool)
        else:
            grid[row, col] = random.choice(accent_pool)

    try:
        row_cols, _row_colors, col_rows, _col_colors = _marker_info(grid)
    except AssertionError:
        return None

    overlap_count = sum(int(col_rows[c] == r) for r, c in enumerate(row_cols))
    if not (0 < overlap_count < n):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if int(np.count_nonzero(out == MAROON)) != overlap_count:
        return None
    if not np.any((out != GREY) & (out != MAROON)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
