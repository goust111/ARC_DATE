from __future__ import annotations

import random
from collections import Counter

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _settle_columns(grid: np.ndarray, bg: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    out = np.full_like(g, bg)
    for c in range(width):
        values = g[:, c][g[:, c] != bg]
        out[height - values.size :, c] = values
    return out


def _sorted_color_counts(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    counts = Counter(map(int, grid[grid != bg].ravel()))
    items = [(int(count), int(color)) for color, count in counts.items()]
    return sorted(items, key=lambda item: (-item[0], item[1]))


def _render_rings(sorted_counts: list[tuple[int, int]], bg: int = 0) -> np.ndarray:
    assert sorted_counts, "expected at least one foreground color"
    max_count = sorted_counts[0][0]
    side = 2 * max_count - 1
    out = np.full((side, side), bg, dtype=int)
    for layer, (_count, color) in enumerate(sorted_counts):
        out[layer : side - layer, layer : side - layer] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Settle sparse colored pixels by gravity, rank colors by their settled pile sizes, and map that order onto concentric square rings.

    Input structure:
    - The grid is a black-background sparse canvas with 3-5 non-black colors.
    - Each color appears a distinct number of times, with only a few pixels per color.
    - In the intended puzzle instances, foreground pixels are scattered across different columns and each column contains at most one foreground pixel.

    Rule:
    - Apply stable vertical gravity independently to each column: read each column's non-black pixels from top to bottom and rewrite them at the bottom in the same order.
    - Count how many settled pixels belong to each non-black color.
    - Sort colors by that count from largest to smallest.
    - Let M = 2 * (largest count) - 1, create an M x M black grid, and paint complete square rings from outside to inside using the sorted colors in order.
    - If there are fewer colors than available rings, leave the remaining inner rings black.

    Key insight:
    - Gravity turns the scattered pixels into an explicit bottom-aligned legend whose per-color pile sizes determine the outer-to-inner ring order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    settled = _settle_columns(g, bg)
    sorted_counts = _sorted_color_counts(settled, bg)
    assert 3 <= len(sorted_counts) <= 5, "expected 3-5 foreground colors"

    only_counts = [count for count, _color in sorted_counts]
    assert len(only_counts) == len(set(only_counts)), "color counts must be distinct"
    return _render_rings(sorted_counts, bg=bg)


# --------------- generator ---------------

GRID_HEIGHT_RANGE = (11, 14)
GRID_WIDTH_HARD_MAX = 24
COLOR_COUNT_RANGE = (3, 5)
PIXELS_PER_COLOR = (2, 6)
GEN_ATTEMPTS = 400


def _sample_counts(num_colors: int) -> list[int]:
    while True:
        counts = random.sample(range(PIXELS_PER_COLOR[0], PIXELS_PER_COLOR[1] + 1), num_colors)
        counts.sort(reverse=True)
        if counts[0] >= num_colors:
            return counts


def _build_scattered_input(
    height: int,
    width: int,
    counts_by_color: list[tuple[int, int]],
) -> np.ndarray | None:
    grid = np.zeros((height, width), dtype=int)

    total_pixels = sum(count for _color, count in counts_by_color)
    if total_pixels > width:
        return None

    used_columns = random.sample(range(width), total_pixels)
    random.shuffle(used_columns)

    unsettled_columns = set(random.sample(used_columns, k=min(2, len(used_columns))))
    bottom_columns = set(random.sample(used_columns, k=max(1, len(used_columns) // 5)))

    idx = 0
    for color, count in counts_by_color:
        cols = used_columns[idx : idx + count]
        idx += count
        for col in cols:
            if col in unsettled_columns:
                row = random.randrange(0, height - 1)
            elif col in bottom_columns:
                row = height - 1
            else:
                row = random.randrange(0, height)
            grid[int(row), int(col)] = int(color)
    return grid


def _validate_generated_input(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    assert max(grid.shape) <= 30

    bg = _get_bg(grid)
    assert bg == 0
    settled = _settle_columns(grid, bg)
    sorted_counts = _sorted_color_counts(settled, bg)
    counts = [count for count, _color in sorted_counts]

    assert 3 <= len(sorted_counts) <= 5
    assert all(PIXELS_PER_COLOR[0] <= count <= PIXELS_PER_COLOR[1] for count in counts)
    assert len(counts) == len(set(counts))

    nonzero_per_column = np.count_nonzero(grid != bg, axis=0)
    assert np.all(nonzero_per_column <= 1)

    floating_columns = 0
    for c in range(grid.shape[1]):
        rows = np.flatnonzero(grid[:, c] != bg)
        if rows.size == 0:
            continue
        if rows[0] != grid.shape[0] - 1:
            floating_columns += 1

    assert floating_columns >= 2

    out = main(grid)
    expected = _render_rings(sorted_counts, bg=bg)
    assert np.array_equal(out, expected)
    assert out.shape[0] == 2 * counts[0] - 1


def _gen_attempt() -> np.ndarray | None:
    num_colors = random.randint(*COLOR_COUNT_RANGE)
    counts = _sample_counts(num_colors)
    colors = random.sample(range(1, 10), num_colors)
    counts_by_color = list(zip(colors, counts))

    height = random.randint(*GRID_HEIGHT_RANGE)
    total_pixels = sum(counts)
    width_min = max(12, total_pixels + 1)
    if width_min > GRID_WIDTH_HARD_MAX:
        return None
    width = random.randint(width_min, GRID_WIDTH_HARD_MAX)

    grid = _build_scattered_input(height, width, counts_by_color)
    if grid is None:
        return None

    try:
        _validate_generated_input(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gravity-to-rings ARC input")
