from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GRAY = 5

EDGE_NAMES = ("top", "bottom", "left", "right")
COUNT_MODES = ("zero", "partial", "overflow")

REGION_ROW_COUNT_RANGE = (2, 4)
REGION_COL_COUNT_RANGE = (2, 4)
REGION_SIZE_RANGE = (4, 6)
GENERATION_ATTEMPTS = 300


def _separator_indices(g: np.ndarray) -> tuple[list[int], list[int]]:
    sep_rows = [int(r) for r in range(g.shape[0]) if np.all(g[r, :] == GRAY)]
    sep_cols = [int(c) for c in range(g.shape[1]) if np.all(g[:, c] == GRAY)]
    return sep_rows, sep_cols


def _region_ranges(size: int, separators: list[int]) -> list[tuple[int, int]]:
    bounds = [-1] + sorted(separators) + [int(size)]
    ranges: list[tuple[int, int]] = []
    for lo, hi in zip(bounds, bounds[1:]):
        start = int(lo) + 1
        end = int(hi) - 1
        if start <= end:
            ranges.append((start, end))
    return ranges


def _region_boxes(g: np.ndarray) -> list[tuple[int, int, int, int]]:
    sep_rows, sep_cols = _separator_indices(g)
    row_mask = np.zeros(g.shape[0], dtype=bool)
    col_mask = np.zeros(g.shape[1], dtype=bool)
    row_mask[sep_rows] = True
    col_mask[sep_cols] = True

    gray_mask = g == GRAY
    assert np.all(~gray_mask | row_mask[:, None] | col_mask[None, :]), "gray pixels must lie on full separator rows or columns"

    row_ranges = _region_ranges(g.shape[0], sep_rows)
    col_ranges = _region_ranges(g.shape[1], sep_cols)
    assert row_ranges and col_ranges, "expected at least one rectangular region"

    boxes: list[tuple[int, int, int, int]] = []
    for r0, r1 in row_ranges:
        for c0, c1 in col_ranges:
            boxes.append((int(r0), int(c0), int(r1), int(c1)))
    return boxes


def _red_edge(local_r: int, local_c: int, height: int, width: int) -> str:
    hits = []
    if local_r == 0:
        hits.append("top")
    if local_r == height - 1:
        hits.append("bottom")
    if local_c == 0:
        hits.append("left")
    if local_c == width - 1:
        hits.append("right")
    assert len(hits) == 1, "red marker must lie on exactly one region edge"
    return hits[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each partitioned rectangle from its red-marked edge by as many full rows or columns as its blue-dot count.

    Input structure:
    - The background is black.
    - Full grey one-pixel separator rows and columns partition the grid into rectangular regions.
    - Each region contains exactly one red pixel on one of its edges and zero or more blue pixels inside the region.

    Rule:
    - For each region, count its blue pixels and determine which edge contains the red pixel.
    - Remove the region's red and blue pixels.
    - If the red pixel is on the top or bottom edge, fill that many complete rows from that edge inward with green.
    - If the red pixel is on the left or right edge, fill that many complete columns from that edge inward with green, clipping at the region size.
    - Keep all grey separator lines unchanged.

    Key insight:
    - The blue dots encode how much to fill, while the red edge marker independently encodes the fill direction, so both signals must be combined region by region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for r0, c0, r1, c1 in _region_boxes(g):
        region = g[r0 : r1 + 1, c0 : c1 + 1]
        height, width = region.shape

        red_coords = np.argwhere(region == RED)
        assert len(red_coords) == 1, "each region must contain exactly one red marker"
        local_r, local_c = map(int, red_coords[0])
        edge = _red_edge(local_r, local_c, height, width)

        blue_count = int(np.sum(region == BLUE))
        region_out = out[r0 : r1 + 1, c0 : c1 + 1]
        region_out[(region == RED) | (region == BLUE)] = BLACK

        if edge == "top":
            fill = min(blue_count, height)
            if fill:
                region_out[:fill, :] = GREEN
        elif edge == "bottom":
            fill = min(blue_count, height)
            if fill:
                region_out[height - fill :, :] = GREEN
        elif edge == "left":
            fill = min(blue_count, width)
            if fill:
                region_out[:, :fill] = GREEN
        else:
            assert edge == "right"
            fill = min(blue_count, width)
            if fill:
                region_out[:, width - fill :] = GREEN

    return out


def _sample_sizes(count: int) -> list[int]:
    sizes = [random.randint(*REGION_SIZE_RANGE) for _ in range(count)]
    if all(size == REGION_SIZE_RANGE[0] for size in sizes):
        sizes[random.randrange(count)] += 1
    return sizes


def _edge_plans(region_count: int) -> list[str]:
    plan = [EDGE_NAMES[i % len(EDGE_NAMES)] for i in range(region_count)]
    random.shuffle(plan)
    return plan


def _count_plans(region_count: int) -> list[str]:
    plan = [COUNT_MODES[i % len(COUNT_MODES)] for i in range(region_count)]
    random.shuffle(plan)
    return plan


def _blue_capacity(height: int, width: int) -> int:
    return max(0, height - 2) * max(0, width - 2)


def _fill_extent(height: int, width: int, edge: str) -> int:
    return int(height if edge in {"top", "bottom"} else width)


def _choose_blue_count(height: int, width: int, edge: str, preferred_mode: str) -> int:
    capacity = _blue_capacity(height, width)
    extent = _fill_extent(height, width, edge)
    mode_order = [preferred_mode] + [mode for mode in COUNT_MODES if mode != preferred_mode]

    for mode in mode_order:
        if mode == "zero":
            return 0
        if mode == "partial":
            hi = min(capacity, extent - 1)
            if hi >= 1:
                return random.randint(1, hi)
        if mode == "overflow":
            lo = extent + 1
            if capacity >= lo:
                return random.randint(lo, capacity)

    if capacity == 0:
        return 0
    return random.randint(1, min(capacity, extent))


def _place_red(grid: np.ndarray, box: tuple[int, int, int, int], edge: str) -> None:
    r0, c0, r1, c1 = box
    height = r1 - r0 + 1
    width = c1 - c0 + 1

    if edge == "top":
        rr = r0
        cc = c0 + random.randint(1, width - 2)
    elif edge == "bottom":
        rr = r1
        cc = c0 + random.randint(1, width - 2)
    elif edge == "left":
        rr = r0 + random.randint(1, height - 2)
        cc = c0
    else:
        assert edge == "right"
        rr = r0 + random.randint(1, height - 2)
        cc = c1
    grid[rr, cc] = RED


def _place_blues(grid: np.ndarray, box: tuple[int, int, int, int], count: int) -> None:
    if count == 0:
        return

    r0, c0, r1, c1 = box
    interior = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    assert count <= len(interior), "blue count must fit in the region interior"
    for r, c in random.sample(interior, count):
        grid[r, c] = BLUE


def _build_grid(row_sizes: list[int], col_sizes: list[int]) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    height = sum(row_sizes) + len(row_sizes) - 1
    width = sum(col_sizes) + len(col_sizes) - 1
    assert max(height, width) <= 30, "generated grid must fit ARC size limits"

    grid = np.full((height, width), BLACK, dtype=int)

    r = 0
    boxes: list[tuple[int, int, int, int]] = []
    for i, h in enumerate(row_sizes):
        c = 0
        for j, w in enumerate(col_sizes):
            boxes.append((r, c, r + h - 1, c + w - 1))
            c += w
            if j + 1 < len(col_sizes):
                grid[:, c] = GRAY
                c += 1
        r += h
        if i + 1 < len(row_sizes):
            grid[r, :] = GRAY
            r += 1

    return grid, boxes


def _gen_attempt() -> np.ndarray | None:
    row_count = random.randint(*REGION_ROW_COUNT_RANGE)
    col_count = random.randint(*REGION_COL_COUNT_RANGE)
    row_sizes = _sample_sizes(row_count)
    col_sizes = _sample_sizes(col_count)
    grid, boxes = _build_grid(row_sizes, col_sizes)

    edge_plan = _edge_plans(len(boxes))
    count_plan = _count_plans(len(boxes))
    order = list(range(len(boxes)))
    random.shuffle(order)

    has_nonzero = False
    for idx in order:
        r0, c0, r1, c1 = boxes[idx]
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        edge = edge_plan[idx]
        blue_count = _choose_blue_count(height, width, edge, count_plan[idx])
        has_nonzero |= blue_count > 0
        _place_red(grid, boxes[idx], edge)
        _place_blues(grid, boxes[idx], blue_count)

    if not has_nonzero:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    if max(out.shape) > 30:
        return None
    if not np.any(out == GREEN):
        return None
    if not np.array_equal(out[grid == GRAY], grid[grid == GRAY]):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC sample")
