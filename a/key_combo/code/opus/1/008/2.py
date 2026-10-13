from __future__ import annotations

import itertools
from typing import Iterable

import numpy as np


BG = 0
VALID_COLORS = (1, 2, 3, 4)
_RNG = np.random.default_rng()
_GEN_CALL_COUNT = 0
_THREE_COLOR_SETS = (
    (1, 2, 3),
    (1, 2, 4),
    (1, 3, 4),
    (2, 3, 4),
)


def _parse_instance(grid: np.ndarray) -> tuple[list[int], list[int], dict[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height <= 30 and width <= 30, "ARC grids must stay within 30x30"
    assert height >= 4 and width >= 5, "grid is too small for the described layout"

    path_rows = [row for row in range(1, height) if int(g[row, 0]) != BG]
    assert path_rows, "expected a non-empty path in the leftmost column"
    assert path_rows == list(range(1, len(path_rows) + 1)), "path must be a contiguous vertical segment from row 1"
    assert len(path_rows) >= 3, "the puzzle requires at least three path markers"

    path_colors = [int(g[row, 0]) for row in path_rows]
    assert all(color in VALID_COLORS for color in path_colors), "path colors must come from 1..4"
    assert len(set(path_colors)) == len(path_colors), "each path color must appear exactly once"

    top_cols = [int(col) for col in np.flatnonzero(g[0] != BG)]
    assert len(top_cols) == len(path_rows), "path length must match the number of top-row markers"
    assert top_cols == list(range(1, len(top_cols) + 1)), "top-row markers must occupy columns 1..K"

    sources: dict[int, int] = {}
    for col in top_cols:
        color = int(g[0, col])
        assert color in VALID_COLORS, "top-row colors must come from 1..4"
        assert color not in sources, "top-row marker colors must be distinct"
        sources[color] = col

    assert set(sources) == set(path_colors), "path colors and top-row marker colors must match exactly"
    assert width > max(path_colors), "target columns must fit inside the grid"

    allowed = np.zeros_like(g, dtype=bool)
    allowed[0, top_cols] = True
    allowed[path_rows, 0] = True
    assert np.array_equal(g != BG, allowed), "unexpected extra foreground pixels"
    return path_rows, path_colors, sources


def _build_output(shape: tuple[int, int], reversed_colors: Iterable[int]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for row, color in enumerate(reversed_colors, start=1):
        out[row, int(color)] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reverse the left-edge path colors, then use that reversed sequence as row-by-row destination columns for the top markers.

    Input structure:
    - The background is black.
    - Column 0 contains one vertical one-pixel-wide path running contiguously from row 1 downward.
    - Row 0 contains one distinct foreground marker in each of columns 1..K, and the marker-color set matches the path-color set.

    Rule:
    - Read the path colors from the top endpoint to the bottom endpoint.
    - Reverse that color sequence.
    - For row i = 1..K, place the top-row marker whose color equals the i-th reversed path color at column `color`.
    - Erase the original path and the original top-row markers.

    Key insight:
    - The path is not preserved; it is only an instruction string, and reversing that string changes the row-to-column mapping for a separate set of marker pixels.
    """
    g = np.asarray(grid, dtype=int)
    _path_rows, path_colors, _sources = _parse_instance(g)
    return _build_output(g.shape, reversed(path_colors))


def _scheduled_colors() -> tuple[int, ...]:
    global _GEN_CALL_COUNT
    idx = _GEN_CALL_COUNT
    _GEN_CALL_COUNT += 1
    if idx % 2 == 0:
        return VALID_COLORS
    return _THREE_COLOR_SETS[(idx // 2) % len(_THREE_COLOR_SETS)]


def _already_in_target_columns(order: tuple[int, ...]) -> bool:
    return all(color == col for col, color in enumerate(order, start=1))


def _sample_path_order(colors: tuple[int, ...]) -> tuple[int, ...]:
    ascending = tuple(sorted(colors))
    descending = tuple(sorted(colors, reverse=True))
    candidates = [
        perm
        for perm in itertools.permutations(colors)
        if perm not in (ascending, descending)
    ]
    assert candidates, "expected at least one non-monotone path order"
    return candidates[int(_RNG.integers(len(candidates)))]


def _sample_top_order(colors: tuple[int, ...], path_order: tuple[int, ...]) -> tuple[int, ...]:
    candidates = []
    reversed_path = path_order[::-1]
    for perm in itertools.permutations(colors):
        if perm == path_order or perm == reversed_path:
            continue
        if _already_in_target_columns(perm):
            continue
        candidates.append(perm)
    assert candidates, "expected at least one non-degenerate top-row order"
    return candidates[int(_RNG.integers(len(candidates)))]


def generate_input() -> np.ndarray:
    colors = _scheduled_colors()
    path_order = _sample_path_order(colors)
    top_order = _sample_top_order(colors, path_order)

    min_width = max(6, max(colors) + 1)
    width = int(_RNG.integers(min_width, min(30, min_width + 7) + 1))
    min_height = len(colors) + 1
    height = int(_RNG.integers(min_height, min(30, min_height + 7) + 1))

    grid = np.full((height, width), BG, dtype=int)
    for row, color in enumerate(path_order, start=1):
        grid[row, 0] = int(color)
    for col, color in enumerate(top_order, start=1):
        grid[0, col] = int(color)

    _parse_instance(grid)
    return grid
