from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

SEED_COLORS = (BLUE, RED, GREEN)
MAX_GENERATION_ATTEMPTS = 300


def _find_dividers(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    height, width = grid.shape

    row_lines_by_color: dict[int, list[int]] = {}
    for r in range(height):
        if np.all(grid[r, :] == grid[r, 0]):
            color = int(grid[r, 0])
            row_lines_by_color.setdefault(color, []).append(int(r))

    col_lines_by_color: dict[int, list[int]] = {}
    for c in range(width):
        if np.all(grid[:, c] == grid[0, c]):
            color = int(grid[0, c])
            col_lines_by_color.setdefault(color, []).append(int(c))

    common_colors = sorted(set(row_lines_by_color) & set(col_lines_by_color))
    assert common_colors, "expected a divider color shared by full rows and full columns"

    divider_color = max(
        common_colors,
        key=lambda color: len(row_lines_by_color[color]) + len(col_lines_by_color[color]),
    )
    return (
        int(divider_color),
        sorted(row_lines_by_color[divider_color]),
        sorted(col_lines_by_color[divider_color]),
    )


def _partition_slices(lines: list[int], size: int) -> list[slice]:
    bounds = [-1, *lines, size]
    spans: list[slice] = []
    lengths: list[int] = []
    for left, right in zip(bounds, bounds[1:]):
        assert right - left > 1, "divider lines must separate non-empty regions"
        spans.append(slice(int(left + 1), int(right)))
        lengths.append(int(right - left - 1))

    assert len(set(lengths)) == 1, "regions must have equal size"
    return spans


def _layer_offsets(seed_color: int, layer: int) -> set[tuple[int, int]]:
    layer = int(layer)
    if seed_color == RED:
        offsets = set()
        for dr in range(-layer, layer + 1):
            dc = layer - abs(dr)
            offsets.add((int(dr), int(dc)))
            offsets.add((int(dr), int(-dc)))
        return offsets

    if seed_color == BLUE:
        offsets = set()
        for dc in range(-layer, layer + 1):
            offsets.add((int(-layer), int(dc)))
            offsets.add((int(layer), int(dc)))
        for dr in range(-layer + 1, layer):
            offsets.add((int(dr), int(-layer)))
            offsets.add((int(dr), int(layer)))
        return offsets

    if seed_color == GREEN:
        return {
            (int(-layer), 0),
            (int(layer), 0),
            (0, int(-layer)),
            (0, int(layer)),
        }

    raise AssertionError(f"unexpected seed color: {seed_color}")


def _paint_growth(
    out: np.ndarray,
    row_slice: slice,
    col_slice: slice,
    seed_r: int,
    seed_c: int,
    seed_color: int,
    layers: int,
) -> None:
    for layer in range(int(layers) + 1):
        for dr, dc in _layer_offsets(int(seed_color), int(layer)):
            rr = int(row_slice.start + seed_r + dr)
            cc = int(col_slice.start + seed_c + dc)
            if row_slice.start <= rr < row_slice.stop and col_slice.start <= cc < col_slice.stop:
                out[rr, cc] = int(seed_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a color-dependent pattern inside each partition.

    The partition's yellow marker count controls the number of grown layers:
    red -> diamond shells, blue -> square-frame shells, green -> cross shells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _divider_color, row_lines, col_lines = _find_dividers(g)
    row_regions = _partition_slices(row_lines, g.shape[0])
    col_regions = _partition_slices(col_lines, g.shape[1])

    out = g.copy()
    out[out == YELLOW] = BLACK

    for row_slice in row_regions:
        for col_slice in col_regions:
            region = g[row_slice, col_slice]
            yellow_count = int(np.count_nonzero(region == YELLOW))
            assert 1 <= yellow_count <= 4, "each region must contain 1..4 yellow markers"

            seeds = np.argwhere(np.isin(region, SEED_COLORS))
            assert len(seeds) == 1, "each region must contain exactly one seed pixel"

            seed_r, seed_c = map(int, seeds[0])
            seed_color = int(region[seed_r, seed_c])
            _paint_growth(out, row_slice, col_slice, seed_r, seed_c, seed_color, yellow_count)

    return out


def _layout_options() -> list[tuple[int, int, int, int]]:
    options: list[tuple[int, int, int, int]] = []
    for region_rows in range(2, 5):
        for region_cols in range(2, 5):
            total_regions = region_rows * region_cols
            if total_regions < len(SEED_COLORS):
                continue
            for region_h in range(5, 9):
                for region_w in range(5, 9):
                    height = region_rows * region_h + (region_rows - 1)
                    width = region_cols * region_w + (region_cols - 1)
                    if 12 <= height <= 30 and 12 <= width <= 30:
                        options.append((region_rows, region_cols, region_h, region_w))
    return options


LAYOUT_OPTIONS = _layout_options()


def _balanced_choices(values: list[int], count: int) -> list[int]:
    picked: list[int] = []
    while len(picked) < count:
        block = values[:]
        random.shuffle(block)
        picked.extend(block)
    picked = picked[:count]
    random.shuffle(picked)
    return picked


def _seed_candidates(region_h: int, region_w: int) -> list[tuple[int, int]]:
    center_r = (region_h - 1) / 2.0
    center_c = (region_w - 1) / 2.0
    return [
        (r, c)
        for r in range(region_h)
        for c in range(region_w)
        if abs(r - center_r) <= 1.0 and abs(c - center_c) <= 1.0
    ]


def _pick_markers(region_h: int, region_w: int, seed: tuple[int, int], count: int) -> list[tuple[int, int]]:
    sr, sc = seed
    candidates = [
        (r, c)
        for r in range(region_h)
        for c in range(region_w)
        if (r, c) != (sr, sc) and max(abs(r - sr), abs(c - sc)) >= 2
    ]
    if len(candidates) < count:
        candidates = [
            (r, c)
            for r in range(region_h)
            for c in range(region_w)
            if (r, c) != (sr, sc)
        ]

    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    for candidate in candidates:
        if all(max(abs(candidate[0] - r), abs(candidate[1] - c)) > 1 for r, c in chosen):
            chosen.append(candidate)
            if len(chosen) == count:
                return chosen

    leftovers = [candidate for candidate in candidates if candidate not in chosen]
    random.shuffle(leftovers)
    chosen.extend(leftovers[: count - len(chosen)])
    return chosen


def _region_slices(region_rows: int, region_cols: int, region_h: int, region_w: int) -> list[tuple[slice, slice]]:
    regions: list[tuple[slice, slice]] = []
    row_step = region_h + 1
    col_step = region_w + 1
    for rr in range(region_rows):
        row_slice = slice(rr * row_step, rr * row_step + region_h)
        for cc in range(region_cols):
            col_slice = slice(cc * col_step, cc * col_step + region_w)
            regions.append((row_slice, col_slice))
    return regions


def _gen_attempt() -> np.ndarray | None:
    region_rows, region_cols, region_h, region_w = random.choice(LAYOUT_OPTIONS)
    height = region_rows * region_h + (region_rows - 1)
    width = region_cols * region_w + (region_cols - 1)

    grid = np.full((height, width), BLACK, dtype=int)
    for r in range(region_h, height, region_h + 1):
        grid[r, :] = GREY
    for c in range(region_w, width, region_w + 1):
        grid[:, c] = GREY

    regions = _region_slices(region_rows, region_cols, region_h, region_w)
    seed_colors = _balanced_choices(list(SEED_COLORS), len(regions))
    yellow_counts = _balanced_choices([1, 2, 3, 4], len(regions))

    for (row_slice, col_slice), seed_color, yellow_count in zip(regions, seed_colors, yellow_counts):
        seed = random.choice(_seed_candidates(region_h, region_w))
        markers = _pick_markers(region_h, region_w, seed, int(yellow_count))

        grid[row_slice.start + seed[0], col_slice.start + seed[1]] = int(seed_color)
        for mr, mc in markers:
            grid[row_slice.start + mr, col_slice.start + mc] = YELLOW

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if np.any(out == YELLOW):
        return None
    if not all(np.any(grid == color) for color in SEED_COLORS):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
