from __future__ import annotations

import random

import numpy as np


ALL_COLORS = tuple(range(10))
HEIGHT_RANGE = (5, 8)  # random.randrange upper bound is exclusive
WIDTH_RANGE = (5, 10)
MAX_ATTEMPTS = 300


def _row_signature(row: np.ndarray) -> tuple[int, int, int]:
    values, counts = np.unique(row, return_counts=True)
    assert len(values) == 2, "each row must contain exactly one majority color and one outlier color"

    major_idx = np.flatnonzero(counts == row.size - 1)
    minor_idx = np.flatnonzero(counts == 1)
    assert len(major_idx) == 1 and len(minor_idx) == 1, "row must have exactly one minority pixel"

    majority = int(values[int(major_idx[0])])
    outlier = int(values[int(minor_idx[0])])
    col = int(np.flatnonzero(row == outlier)[0])
    return majority, outlier, col


def _first_lower_match(grid: np.ndarray, row: int, col: int, color: int) -> int | None:
    for rr in range(row + 1, grid.shape[0]):
        if int(grid[rr, col]) == color:
            return rr
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each row's unique outlier pixel down to sit just above the nearest same-colored pixel in its column.

    Input structure:
    - The grid contains several visible colors and no dedicated background.
    - Every row has one majority color plus exactly one single-cell outlier of a different color.
    - The outlier color can differ from row to row.

    Rule:
    - For each row, identify the unique outlier pixel and its row-majority color.
    - In the outlier's column, scan downward to the first pixel with the same color as the outlier.
    - Replace the source outlier with the row-majority color, then paint the cell immediately above that lower match with the outlier color.
    - Rows whose outlier has no lower same-colored match are left unchanged.

    Key insight:
    - The anomaly is detected row-wise, but its destination is determined column-wise by the nearest lower pixel of the same color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    moves: list[tuple[int, int, int, int, int]] = []
    for row in range(g.shape[0]):
        majority, outlier, col = _row_signature(g[row])
        support = _first_lower_match(g, row, col, outlier)
        if support is None:
            continue
        moves.append((row, col, majority, outlier, support - 1))

    out = g.copy()
    for row, col, majority, _outlier, _target in moves:
        out[row, col] = majority
    for _row, col, _majority, outlier, target in moves:
        out[target, col] = outlier
    return out


def _choose_moved_rows(height: int) -> list[int]:
    candidates = list(range(height - 2))
    skipped = set()
    if candidates and random.random() < 0.5:
        skipped.add(random.choice(candidates))
    moved = [row for row in candidates if row not in skipped]
    assert len(moved) >= 2
    return moved


def _support_map(height: int, moved_rows: list[int]) -> dict[int, int] | None:
    order = sorted(moved_rows, reverse=True)
    used: set[int] = set()
    mapping: dict[int, int] = {}

    def backtrack(idx: int) -> bool:
        if idx == len(order):
            return True
        row = order[idx]
        candidates = [support for support in range(row + 2, height) if support not in used]
        random.shuffle(candidates)
        for support in candidates:
            mapping[row] = support
            used.add(support)
            if backtrack(idx + 1):
                return True
            used.remove(support)
            del mapping[row]
        return False

    return mapping if backtrack(0) else None


def _column_neighbors(height: int, support_map: dict[int, int]) -> list[set[int]]:
    neighbors = [set() for _ in range(height)]

    def add_edge(a: int, b: int) -> None:
        neighbors[a].add(b)
        neighbors[b].add(a)

    for row in range(height - 1):
        add_edge(row, row + 1)
    for src, support in support_map.items():
        add_edge(src, support)
        add_edge(src, support - 1)
    return neighbors


def _choose_columns(width: int, neighbors: list[set[int]]) -> list[int] | None:
    order = sorted(range(len(neighbors)), key=lambda row: (-len(neighbors[row]), row))
    cols = [-1] * len(neighbors)

    def backtrack(idx: int) -> bool:
        if idx == len(order):
            return True
        row = order[idx]
        options = list(range(width))
        random.shuffle(options)
        for col in options:
            if any(cols[other] == col for other in neighbors[row]):
                continue
            cols[row] = col
            if backtrack(idx + 1):
                return True
            cols[row] = -1
        return False

    return cols if backtrack(0) else None


def _expected_output(
    grid: np.ndarray,
    columns: list[int],
    majorities: list[int],
    anomaly_colors: list[int],
    support_map: dict[int, int],
) -> np.ndarray:
    out = grid.copy()
    for row, support in support_map.items():
        out[row, columns[row]] = majorities[row]
        out[support - 1, columns[row]] = anomaly_colors[row]
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randrange(*HEIGHT_RANGE)
    width = random.randrange(*WIDTH_RANGE)

    majorities = random.sample(ALL_COLORS, height)
    extra_colors = [color for color in ALL_COLORS if color not in majorities]

    moved_rows = _choose_moved_rows(height)
    support_map = _support_map(height, moved_rows)
    if support_map is None:
        return None

    stationary_rows = [row for row in range(height) if row not in moved_rows]
    if len(stationary_rows) > len(extra_colors):
        return None

    anomaly_colors = [0] * height
    for row, support in support_map.items():
        anomaly_colors[row] = majorities[support]

    stationary_colors = random.sample(extra_colors, len(stationary_rows))
    for row, color in zip(stationary_rows, stationary_colors):
        anomaly_colors[row] = color

    neighbors = _column_neighbors(height, support_map)
    columns = _choose_columns(width, neighbors)
    if columns is None:
        return None

    grid = np.repeat(np.array(majorities, dtype=int)[:, None], width, axis=1)
    for row in range(height):
        assert anomaly_colors[row] != majorities[row]
        grid[row, columns[row]] = anomaly_colors[row]

    expected_supports: dict[int, int] = {}
    for row in range(height):
        majority, outlier, col = _row_signature(grid[row])
        assert majority == majorities[row]
        assert outlier == anomaly_colors[row]
        assert col == columns[row]
        support = _first_lower_match(grid, row, col, outlier)
        if support is not None:
            expected_supports[row] = support

    if expected_supports != support_map:
        return None

    expected = _expected_output(grid, columns, majorities, anomaly_colors, support_map)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
