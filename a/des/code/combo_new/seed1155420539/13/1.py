from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GENERATION_RETRIES = 300
GRID_HEIGHT_RANGE = (12, 23)
LINE_COUNT_OPTIONS = (2, 3)
REGION_WIDTH_RANGE = (4, 6)
FRAGMENT_PLACEMENT_TRIES = 100

FRAGMENT_SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2)),
)


def _normalize_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _c in shape)
    min_c = min(c for _r, c in shape)
    return tuple(sorted((r - min_r, c - min_c) for r, c in shape))


def _shape_variants(shape: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    variants: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for mirror in (False, True):
        for rotation in range(4):
            transformed = []
            for r, c in shape:
                rr, cc = r, c
                if mirror:
                    cc = -cc
                for _ in range(rotation):
                    rr, cc = cc, -rr
                transformed.append((rr, cc))
            normalized = _normalize_shape(tuple(transformed))
            if normalized not in seen:
                seen.add(normalized)
                variants.append(normalized)
    return variants


SHAPE_VARIANTS = {shape: _shape_variants(shape) for shape in FRAGMENT_SHAPES}


def _segment_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    if r0 == r1:
        start, end = sorted((c0, c1))
        return [(r0, c) for c in range(start, end + 1)]
    assert c0 == c1, "anchor pairs must share a row or a column"
    start, end = sorted((r0, r1))
    return [(r, c0) for r in range(start, end + 1)]


def _analyze_regions(grid: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape
    assert height <= 30 and width <= 30, "ARC grids must stay within 30x30"
    assert np.all((0 <= g) & (g <= 9)), "expected ARC color ids"

    anchor_colors = [int(color) for color in np.unique(g) if color not in (BLACK, GREY)]
    assert anchor_colors, "expected at least one colored anchor pair"

    line_grid = np.full((height, width), -1, dtype=int)
    for color in anchor_colors:
        coords = np.argwhere(g == color)
        assert len(coords) == 2, "each anchor color must appear exactly twice"
        (r0, c0), (r1, c1) = [tuple(map(int, coord)) for coord in coords]
        assert r0 == r1 or c0 == c1, "anchor pair must be aligned"
        for r, c in _segment_cells(r0, c0, r1, c1):
            existing = int(line_grid[r, c])
            assert existing in (-1, color), "drawn segments must not overlap in different colors"
            line_grid[r, c] = color

    labels = np.zeros((height, width), dtype=int)
    regions: list[dict[str, object]] = []
    next_label = 0

    for start_r in range(height):
        for start_c in range(width):
            if line_grid[start_r, start_c] != -1 or labels[start_r, start_c] != 0:
                continue

            next_label += 1
            queue = deque([(start_r, start_c)])
            labels[start_r, start_c] = next_label
            region_cells: list[tuple[int, int]] = []
            grey_cells: list[tuple[int, int]] = []
            boundary_by_color: dict[int, set[tuple[int, int]]] = defaultdict(set)

            while queue:
                r, c = queue.popleft()
                region_cells.append((r, c))
                if g[r, c] == GREY:
                    grey_cells.append((r, c))

                for dr, dc in NEIGHBORS4:
                    nr = r + dr
                    nc = c + dc
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    if line_grid[nr, nc] == -1:
                        if labels[nr, nc] == 0:
                            labels[nr, nc] = next_label
                            queue.append((nr, nc))
                    else:
                        boundary_by_color[int(line_grid[nr, nc])].add((nr, nc))

            regions.append(
                {
                    "cells": region_cells,
                    "grey_cells": grey_cells,
                    "boundary_counts": {color: len(cells) for color, cells in boundary_by_color.items()},
                }
            )

    return line_grid, regions


def _region_fill_color(boundary_counts: dict[int, int]) -> int | None:
    if len(boundary_counts) == 1:
        return next(iter(boundary_counts))

    if len(boundary_counts) == 2:
        (color_a, count_a), (color_b, count_b) = list(boundary_counts.items())
        if count_a < count_b:
            return color_a
        if count_b < count_a:
            return color_b
        return None

    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw the implied anchor lines, then recolor grey fragments by the color mix on their enclosing boundary.

    Input structure:
    - The background is black.
    - Several non-grey colors appear as anchor pairs, and each such color occurs exactly twice.
    - Every anchor pair is aligned horizontally or vertically, so it implies a straight line segment between the two anchors.
    - Optional grey fragments occupy some of the background regions between these implied segments and the grid edge.

    Rule:
    - For each same-color anchor pair, draw the full horizontal or vertical segment between the anchors, including both endpoints.
    - Treat the drawn segments as barriers and split the remaining cells into 4-connected regions.
    - For every region that contains grey cells, inspect the colors of the line pixels touching that region's boundary.
    - If exactly one line color touches the region, recolor all grey cells in the region to that color.
    - If exactly two line colors touch the region, recolor all grey cells to the color whose boundary contribution has fewer pixels; if the counts tie, leave the grey cells unchanged.
    - If three or more line colors touch the region, leave the grey cells unchanged.

    Key insight:
    - The anchors first have to be completed into full straight walls; only after that do the grey fragments reveal their rule, because each fragment inherits its color from the mix of wall colors bordering its region.
    """
    g = np.asarray(grid, dtype=int)
    line_grid, regions = _analyze_regions(g)

    out = g.copy()
    for region in regions:
        fill_color = _region_fill_color(region["boundary_counts"])  # type: ignore[arg-type]
        if fill_color is None:
            continue
        for r, c in region["grey_cells"]:  # type: ignore[index]
            out[r, c] = fill_color

    line_mask = line_grid != -1
    out[line_mask] = line_grid[line_mask]
    return out


def _apply_transform(grid: np.ndarray, op: int) -> np.ndarray:
    if op == 0:
        return grid.copy()
    if op == 1:
        return np.rot90(grid, 1).copy()
    if op == 2:
        return np.rot90(grid, 2).copy()
    if op == 3:
        return np.rot90(grid, 3).copy()
    if op == 4:
        return np.fliplr(grid).copy()
    if op == 5:
        return np.flipud(grid).copy()
    if op == 6:
        return grid.T.copy()
    if op == 7:
        return np.fliplr(grid.T).copy()
    raise ValueError(f"unknown transform op: {op}")


def _region_intervals(widths: list[int]) -> tuple[int, list[int], list[tuple[int, int]]]:
    line_columns: list[int] = []
    col = widths[0]
    for idx in range(len(widths) - 1):
        line_columns.append(col)
        if idx + 1 < len(widths) - 1:
            col += 1 + widths[idx + 1]

    total_width = sum(widths) + len(line_columns)
    regions = []
    for idx in range(len(widths)):
        start = 0 if idx == 0 else line_columns[idx - 1] + 1
        end = total_width - 1 if idx == len(widths) - 1 else line_columns[idx] - 1
        regions.append((start, end))
    return total_width, line_columns, regions


def _add_edge_segment(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    region: tuple[int, int],
    color: int,
    rng: random.Random,
) -> bool:
    height, _width = grid.shape
    start_col, end_col = region
    region_width = end_col - start_col + 1
    seg_len = rng.randint(2, min(region_width, 4))
    seg_start = rng.randint(start_col, end_col - seg_len + 1)
    seg_end = seg_start + seg_len - 1
    row = 0 if rng.random() < 0.5 else height - 1

    if (row, seg_start) in occupied or (row, seg_end) in occupied:
        return False

    grid[row, seg_start] = color
    grid[row, seg_end] = color
    occupied.add((row, seg_start))
    occupied.add((row, seg_end))
    return True


def _place_fragment(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    region: tuple[int, int],
    rng: random.Random,
) -> bool:
    height, _width = grid.shape
    start_col, end_col = region

    row_lo = 2
    row_hi = height - 3
    col_lo = start_col + 1
    col_hi = end_col - 1
    if col_lo > col_hi:
        col_lo, col_hi = start_col, end_col

    for _ in range(FRAGMENT_PLACEMENT_TRIES):
        base_shape = rng.choice(FRAGMENT_SHAPES)
        shape = rng.choice(SHAPE_VARIANTS[base_shape])
        shape_h = max(r for r, _c in shape) + 1
        shape_w = max(c for _r, c in shape) + 1

        if row_hi - row_lo + 1 < shape_h or col_hi - col_lo + 1 < shape_w:
            continue

        top = rng.randint(row_lo, row_hi - shape_h + 1)
        left = rng.randint(col_lo, col_hi - shape_w + 1)
        cells = [(top + r, left + c) for r, c in shape]
        if any(cell in occupied for cell in cells):
            continue

        for r, c in cells:
            grid[r, c] = GREY
            occupied.add((r, c))
        return True

    return False


def _region_branch(boundary_counts: dict[int, int]) -> str:
    if len(boundary_counts) == 1:
        return "one"
    if len(boundary_counts) == 2 and len(set(boundary_counts.values())) == 2:
        return "two"
    if len(boundary_counts) >= 3:
        return "three"
    return "other"


def _is_good_generated_input(grid: np.ndarray) -> bool:
    line_grid, regions = _analyze_regions(grid)
    assert not np.any((grid == GREY) & (line_grid != -1)), "fragments must not lie on drawn lines"

    branch_hits = {
        _region_branch(region["boundary_counts"])  # type: ignore[arg-type]
        for region in regions
        if region["grey_cells"]  # type: ignore[index]
    }
    return {"one", "two", "three"} <= branch_hits


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    line_count = rng.choice(LINE_COUNT_OPTIONS)
    widths = [rng.randint(*REGION_WIDTH_RANGE) for _ in range(line_count + 1)]
    width, line_columns, regions = _region_intervals(widths)
    height = rng.randint(*GRID_HEIGHT_RANGE)

    grid = np.full((height, width), BLACK, dtype=int)
    occupied: set[tuple[int, int]] = set()

    needed_colors = line_count + 2
    colors = rng.sample(PALETTE, needed_colors)
    line_colors = colors[:line_count]
    two_color_extra = colors[line_count]
    three_color_extra = colors[line_count + 1]

    for col, color in zip(line_columns, line_colors):
        grid[0, col] = color
        grid[height - 1, col] = color
        occupied.add((0, col))
        occupied.add((height - 1, col))

    edge_regions = [0, len(regions) - 1]
    rng.shuffle(edge_regions)
    one_color_region = edge_regions[0]
    two_color_region = edge_regions[1]
    three_color_region = rng.randint(1, len(regions) - 2)

    if not _add_edge_segment(grid, occupied, regions[two_color_region], two_color_extra, rng):
        return None
    if not _add_edge_segment(grid, occupied, regions[three_color_region], three_color_extra, rng):
        return None

    for region_index in (one_color_region, two_color_region, three_color_region):
        fragment_count = rng.randint(1, 2)
        for _ in range(fragment_count):
            if not _place_fragment(grid, occupied, regions[region_index], rng):
                break

    transformed = _apply_transform(grid, rng.randrange(8))
    if not _is_good_generated_input(transformed):
        return None
    return transformed


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        result = _gen_attempt(rng)
        if result is not None:
            return result
    raise RuntimeError(f"failed to generate valid input after {GENERATION_RETRIES} attempts")
