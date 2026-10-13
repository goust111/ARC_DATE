from __future__ import annotations

import random
from collections import Counter
from itertools import combinations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PALETTE = (BLUE, RED, GREEN, YELLOW)
MAX_GENERATION_ATTEMPTS = 400


def _separator_cross(grid: np.ndarray) -> tuple[int, int]:
    full_rows = np.flatnonzero(np.all(grid == GREY, axis=1))
    full_cols = np.flatnonzero(np.all(grid == GREY, axis=0))
    assert len(full_rows) == 1 and len(full_cols) == 1, "expected one full grey cross"
    sep_r = int(full_rows[0])
    sep_c = int(full_cols[0])
    assert grid.shape[0] == 2 * sep_r + 1, "cross must split rows evenly"
    assert grid.shape[1] == 2 * sep_c + 1, "cross must split cols evenly"
    return sep_r, sep_c


def _region_slices(grid: np.ndarray) -> list[tuple[slice, slice]]:
    sep_r, sep_c = _separator_cross(grid)
    return [
        (slice(0, sep_r), slice(0, sep_c)),
        (slice(0, sep_r), slice(sep_c + 1, grid.shape[1])),
        (slice(sep_r + 1, grid.shape[0]), slice(0, sep_c)),
        (slice(sep_r + 1, grid.shape[0]), slice(sep_c + 1, grid.shape[1])),
    ]


def _extract_region_bars(region: np.ndarray) -> list[int]:
    bars: list[int] = []
    bar_rows: list[int] = []
    for row_index, row in enumerate(region):
        colors = set(map(int, row))
        assert len(colors) == 1, "region rows must be uniform"
        color = next(iter(colors))
        if color == BLACK:
            continue
        assert color in PALETTE, "bars must use ARC primary colors"
        bars.append(color)
        bar_rows.append(int(row_index))
    assert 3 <= len(bars) <= 4, "each region must contain 3-4 bars"
    assert all(b - a >= 2 for a, b in zip(bar_rows, bar_rows[1:])), "bars must be separated"
    counts = Counter(bars)
    assert max(counts.values()) <= 2, "a region may duplicate a color at most once"
    return bars


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder each quadrant's horizontal bars by the colors' global cross-quadrant frequencies.

    Input structure:
    - A black background grid contains one full grey horizontal separator and one full grey vertical separator.
    - The grey cross splits the canvas into four equal rectangular regions.
    - Inside each region, 3-4 full-width one-row bars use colors from blue, red, green, and yellow, with black spacer rows between them.

    Rule:
    - Count how many horizontal bars of each color appear across all four regions combined.
    - Rank colors from highest total count to lowest total count.
    - In each region, keep that region's own multiset of bars but reorder them top-to-bottom by the global frequency rank, then pack them flush against the region's top edge with no gaps; fill the remaining region rows with black.

    Key insight:
    - The sorting key is not local to any single region or tied to color ids; it is the global bar frequency computed across all four quadrants.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"

    region_specs = _region_slices(g)
    regions = [g[row_slice, col_slice] for row_slice, col_slice in region_specs]
    region_shape = regions[0].shape
    assert all(region.shape == region_shape for region in regions), "regions must match"

    region_bars = [_extract_region_bars(region) for region in regions]
    region_color_sets = [set(bars) for bars in region_bars]
    assert len({frozenset(colors) for colors in region_color_sets}) == 1, "regions must share one color set"

    global_counts: Counter[int] = Counter()
    for bars in region_bars:
        global_counts.update(bars)
    totals = list(global_counts.values())
    assert len(totals) == len(set(totals)), "global color frequencies must be unique"

    rank = {
        color: index
        for index, (color, _count) in enumerate(
            sorted(global_counts.items(), key=lambda item: item[1], reverse=True)
        )
    }

    out = np.full_like(g, BLACK)
    sep_r, sep_c = _separator_cross(g)
    out[sep_r, :] = GREY
    out[:, sep_c] = GREY

    for (row_slice, col_slice), bars in zip(region_specs, region_bars):
        ordered = sorted(bars, key=lambda color: rank[color])
        region_out = np.full(region_shape, BLACK, dtype=int)
        for row, color in enumerate(ordered):
            region_out[row, :] = color
        out[row_slice, col_slice] = region_out

    return out


def _sample_region_shape() -> tuple[int, int]:
    height = random.randint(7, 8)
    width = random.randint(4, 8)
    return height, width


def _sample_duplicate_targets(colors: list[int]) -> dict[int, int]:
    pattern = random.choice([(0, 1, 2), (0, 1, 3)])
    shuffled = list(pattern)
    random.shuffle(shuffled)
    return {color: dup_count for color, dup_count in zip(colors, shuffled)}


def _assign_region_duplicates(targets: dict[int, int]) -> list[int | None]:
    duplicate_colors = [color for color, count in targets.items() for _ in range(count)]
    region_indices = list(range(4))
    chosen_regions = random.sample(region_indices, len(duplicate_colors))
    random.shuffle(duplicate_colors)

    per_region: list[int | None] = [None, None, None, None]
    for region_index, color in zip(chosen_regions, duplicate_colors):
        per_region[int(region_index)] = int(color)
    return per_region


def _sample_region_bars(colors: list[int], duplicate_color: int | None) -> list[int]:
    bars = list(colors)
    if duplicate_color is not None:
        bars.append(int(duplicate_color))
    random.shuffle(bars)
    return bars


def _place_region(region_shape: tuple[int, int], bars: list[int]) -> np.ndarray:
    height, width = map(int, region_shape)
    region = np.full((height, width), BLACK, dtype=int)
    candidates = [
        rows
        for rows in combinations(range(height), len(bars))
        if all(b - a >= 2 for a, b in zip(rows, rows[1:]))
    ]
    rows = list(random.choice(candidates))
    for row, color in zip(rows, bars):
        region[row, :] = int(color)
    return region


def _build_grid(region_shape: tuple[int, int], bars_by_region: list[list[int]]) -> np.ndarray:
    region_h, region_w = map(int, region_shape)
    grid = np.full((2 * region_h + 1, 2 * region_w + 1), BLACK, dtype=int)
    grid[region_h, :] = GREY
    grid[:, region_w] = GREY

    for (row_slice, col_slice), bars in zip(_region_slices(grid), bars_by_region):
        grid[row_slice, col_slice] = _place_region(region_shape, bars)
    return grid


def _is_nontrivial(grid: np.ndarray) -> bool:
    return not np.array_equal(main(grid), grid)


def _gen_attempt() -> np.ndarray | None:
    region_shape = _sample_region_shape()
    colors = random.sample(PALETTE, 3)
    duplicate_targets = _sample_duplicate_targets(colors)
    duplicate_choices = _assign_region_duplicates(duplicate_targets)
    bars_by_region = [
        _sample_region_bars(colors, duplicate_color) for duplicate_color in duplicate_choices
    ]

    global_counts: Counter[int] = Counter()
    for bars in bars_by_region:
        global_counts.update(bars)
    if len(set(global_counts.values())) != len(global_counts):
        return None

    grid = _build_grid(region_shape, bars_by_region)

    try:
        solved = main(grid)
    except Exception:
        return None

    if solved.shape != grid.shape:
        return None
    if not _is_nontrivial(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
