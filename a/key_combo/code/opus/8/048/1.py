from __future__ import annotations

import random

import numpy as np


BG = 0
TIE_COLOR = 5
BAR_COLORS = (1, 2, 3, 4, 7)
BAR_COUNT_RANGE = (4, 5)  # Palette uniqueness caps the count at 5.
HEIGHT_RANGE = (18, 26)
WIDTH_RANGE = (18, 28)
BAR_HEIGHTS = tuple(range(6, 17, 2))
MAX_GENERATION_ATTEMPTS = 400


def _find_bars(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = grid.shape
    bars: list[tuple[int, int, int, int]] = []
    for col in range(w):
        rows = np.flatnonzero(grid[:, col] != BG)
        if rows.size == 0:
            continue
        assert rows[-1] == h - 1, "each bar must be bottom-anchored"
        assert np.all(np.diff(rows) == 1), "each occupied column must form one contiguous vertical bar"
        colors = np.unique(grid[rows, col])
        assert colors.size == 1, "each bar must be monochrome"
        color = int(colors[0])
        assert color in BAR_COLORS, "unexpected foreground color"
        top = int(rows[0])
        height = int(rows.size)
        bars.append((col, top, height, color))

    assert BAR_COUNT_RANGE[0] <= len(bars) <= BAR_COUNT_RANGE[1], "expected 4-5 bars"
    assert len({color for _col, _top, _height, color in bars}) == len(bars), "bar colors must be unique"

    bars.sort()
    for (col_a, _top_a, _height_a, _color_a), (col_b, _top_b, _height_b, _color_b) in zip(bars, bars[1:]):
        assert col_b - col_a >= 3, "bars must be separated by at least two empty columns"
    return bars


def _seed_segments(bars: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    segments: list[tuple[int, int, int, int]] = []
    for col, top, height, color in bars:
        keep = (height + 1) // 2
        segments.append((col, top, top + keep - 1, color))
    return segments


def _segment_distance(row: int, col: int, seg_col: int, seg_top: int, seg_bottom: int) -> int:
    distance = abs(col - seg_col)
    if row < seg_top:
        distance += seg_top - row
    elif row > seg_bottom:
        distance += row - seg_bottom
    return int(distance)


def _transform(grid: np.ndarray) -> np.ndarray:
    bars = _find_bars(grid)
    segments = _seed_segments(bars)
    out = np.empty_like(grid)
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            best_distance: int | None = None
            nearest_colors: set[int] = set()
            for seg_col, seg_top, seg_bottom, color in segments:
                distance = _segment_distance(row, col, seg_col, seg_top, seg_bottom)
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    nearest_colors = {color}
                elif distance == best_distance:
                    nearest_colors.add(color)
            assert best_distance is not None
            out[row, col] = next(iter(nearest_colors)) if len(nearest_colors) == 1 else TIE_COLOR
    return out


def _oracle(grid: np.ndarray) -> np.ndarray:
    bars = _find_bars(grid)
    seeds: list[tuple[int, int, int]] = []
    for col, top, height, color in bars:
        keep = (height + 1) // 2
        for row in range(top, top + keep):
            seeds.append((row, col, color))

    out = np.empty_like(grid)
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            best_distance: int | None = None
            nearest_colors: set[int] = set()
            for seed_row, seed_col, color in seeds:
                distance = abs(row - seed_row) + abs(col - seed_col)
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    nearest_colors = {color}
                elif distance == best_distance:
                    nearest_colors.add(color)
            assert best_distance is not None
            out[row, col] = next(iter(nearest_colors)) if len(nearest_colors) == 1 else TIE_COLOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Partition the board by Manhattan-nearest color after truncating each bottom-anchored vertical bar to its upper half.

    Input structure:
    - The grid has a black background with 4-5 single-column colored bars.
    - Each bar is monochrome, bottom-anchored, vertically contiguous, and separated from the next bar by at least two empty columns.
    - Bar colors are distinct and drawn from blue, red, green, yellow, and orange.

    Rule:
    - For each bar, erase its lower half and keep only the upper ceil(height / 2) cells as seed pixels.
    - For every cell, compute the Manhattan distance to the remaining seed pixels.
    - Color the cell with the unique nearest seed color.
    - If the nearest distance is tied between different colors, color the cell gray.

    Key insight:
    - The visible input bars are not the final seeds: only their upper halves survive, so the Voronoi boundaries are determined by shortened vertical seed segments rather than by the original full-height bars.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and 0 < g.shape[0] <= 30 and 0 < g.shape[1] <= 30, "expected a 2D grid with size <= 30"
    assert np.all((0 <= g) & (g <= 9)), "ARC colors must stay in 0..9"
    assert not np.any(g == TIE_COLOR), "gray is reserved for output ties"
    return _transform(g)


def _sample_columns(width: int, count: int) -> list[int]:
    compressed = width - 2 * (count - 1)
    assert compressed >= count
    base = sorted(random.sample(range(compressed), count))
    return [value + 2 * idx for idx, value in enumerate(base)]


def _build_input(height: int, width: int, columns: list[int], heights: list[int], colors: list[int]) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for col, bar_height, color in zip(columns, heights, colors):
        top = height - bar_height
        grid[top:, col] = color
    return grid


def _generate_candidate() -> np.ndarray:
    height = random.randint(*HEIGHT_RANGE)
    width = random.randint(*WIDTH_RANGE)
    count = random.randint(*BAR_COUNT_RANGE)
    max_bar_height = min(BAR_HEIGHTS[-1], height - 2)
    choices = [value for value in BAR_HEIGHTS if value <= max_bar_height]
    assert len(choices) >= count
    bar_heights = random.sample(choices, count)
    columns = _sample_columns(width, count)
    colors = random.sample(BAR_COLORS, count)
    return _build_input(height, width, columns, bar_heights, colors)


def generate_input() -> np.ndarray:
    """Generate a valid random input grid for this puzzle."""
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_candidate()
        if np.array_equal(main(grid), _oracle(grid)):
            return grid
    raise RuntimeError("failed to generate a valid input")


if __name__ == "__main__":
    sample = generate_input()
    result = main(sample)
    print(sample.shape)
    print(np.unique(result).tolist())
