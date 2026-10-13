import random
from collections import Counter, defaultdict, deque

import numpy as np


BLACK = 0
GREY = 5
MARKER_COLORS = tuple(color for color in range(10) if color not in (BLACK, GREY))

GENERATION_ATTEMPTS = 300
REGION_COUNT_OPTIONS = ((2, 2), (2, 3), (3, 2), (2, 4), (4, 2))
BRANCH_MODES = ("unequal", "equal_rows", "equal_cols")
REGION_SIZE_MIN = 2
REGION_SIZE_MAX = 5
MARGIN_MIN = 1
MARGIN_MAX = 3


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _adjacent_equal(values: list[int]) -> bool:
    return any(values[i] == values[i + 1] for i in range(len(values) - 1))


def _sample_band_sizes(rng: random.Random, count: int, require_equal: bool) -> list[int]:
    for _ in range(200):
        sizes = [rng.randint(REGION_SIZE_MIN, REGION_SIZE_MAX) for _ in range(count)]
        has_equal = _adjacent_equal(sizes)
        if has_equal == require_equal:
            return sizes
    raise RuntimeError("failed to sample band sizes")


def _wall_orientations(wall: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = np.zeros_like(wall, dtype=bool)
    right = np.zeros_like(wall, dtype=bool)
    up = np.zeros_like(wall, dtype=bool)
    down = np.zeros_like(wall, dtype=bool)
    left[:, 1:] = wall[:, :-1]
    right[:, :-1] = wall[:, 1:]
    up[1:, :] = wall[:-1, :]
    down[:-1, :] = wall[1:, :]
    horiz = wall & (left | right)
    vert = wall & (up | down)
    return horiz, vert


def _extract_regions(grid: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
    height, width = grid.shape
    walkable = grid != GREY
    seen = np.zeros_like(walkable, dtype=bool)
    region_ids = np.full(grid.shape, -1, dtype=int)
    regions: list[dict[str, object]] = []

    for start_r in range(height):
        for start_c in range(width):
            if not walkable[start_r, start_c] or seen[start_r, start_c]:
                continue

            queue = deque([(start_r, start_c)])
            seen[start_r, start_c] = True
            cells: list[tuple[int, int]] = []
            touches_border = False

            while queue:
                r, c = queue.popleft()
                cells.append((r, c))
                if r in (0, height - 1) or c in (0, width - 1):
                    touches_border = True
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < height and 0 <= nc < width and walkable[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            if touches_border:
                continue

            marker_colors = {
                int(grid[r, c])
                for r, c in cells
                if int(grid[r, c]) not in (BLACK, GREY)
            }
            assert len(marker_colors) == 1, "each enclosed region must contain exactly one marker color"
            region_color = next(iter(marker_colors))
            region_id = len(regions)
            for r, c in cells:
                region_ids[r, c] = region_id
            regions.append(
                {
                    "cells": cells,
                    "area": len(cells),
                    "color": region_color,
                }
            )

    assert regions, "expected at least one enclosed region"
    return region_ids, regions


def _segment_color_horizontal(
    region_ids: np.ndarray,
    regions: list[dict[str, object]],
    row: int,
    col0: int,
    col1: int,
) -> int:
    if col1 - col0 <= 1:
        return GREY

    above_ids: set[int] = set()
    below_ids: set[int] = set()
    height, _ = region_ids.shape
    for col in range(col0 + 1, col1):
        if row - 1 >= 0:
            up_id = int(region_ids[row - 1, col])
            if up_id >= 0:
                above_ids.add(up_id)
        if row + 1 < height:
            down_id = int(region_ids[row + 1, col])
            if down_id >= 0:
                below_ids.add(down_id)

    if len(above_ids) != 1 or len(below_ids) != 1:
        return GREY

    top_id = next(iter(above_ids))
    bottom_id = next(iter(below_ids))
    if top_id == bottom_id:
        return GREY

    top_area = int(regions[top_id]["area"])
    bottom_area = int(regions[bottom_id]["area"])
    if top_area == bottom_area:
        return GREY
    if top_area < bottom_area:
        return int(regions[top_id]["color"])
    return int(regions[bottom_id]["color"])


def _segment_color_vertical(
    region_ids: np.ndarray,
    regions: list[dict[str, object]],
    col: int,
    row0: int,
    row1: int,
) -> int:
    if row1 - row0 <= 1:
        return GREY

    left_ids: set[int] = set()
    right_ids: set[int] = set()
    _, width = region_ids.shape
    for row in range(row0 + 1, row1):
        if col - 1 >= 0:
            left_id = int(region_ids[row, col - 1])
            if left_id >= 0:
                left_ids.add(left_id)
        if col + 1 < width:
            right_id = int(region_ids[row, col + 1])
            if right_id >= 0:
                right_ids.add(right_id)

    if len(left_ids) != 1 or len(right_ids) != 1:
        return GREY

    left_id = next(iter(left_ids))
    right_id = next(iter(right_ids))
    if left_id == right_id:
        return GREY

    left_area = int(regions[left_id]["area"])
    right_area = int(regions[right_id]["area"])
    if left_area == right_area:
        return GREY
    if left_area < right_area:
        return int(regions[left_id]["color"])
    return int(regions[right_id]["color"])


def _majority_color(colors: list[int]) -> int:
    counts = Counter(colors)
    best = counts.most_common()
    if len(best) > 1 and best[1][1] == best[0][1]:
        return GREY
    return int(best[0][0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey-bounded rectangular region with its marker color, then recolor shared boundary segments by the smaller neighboring region.

    Input structure:
    - The background is black.
    - Grey one-pixel horizontal and vertical line segments connect to form
      closed rectangular compartments.
    - Each enclosed compartment contains exactly one non-black, non-grey marker
      pixel, and different compartments use different marker colors.

    Rule:
    - Find every enclosed non-grey region and flood-fill that whole region with
      its unique marker color.
    - For each straight grey boundary segment, compare the two enclosed regions
      on its opposite sides; recolor the segment to the smaller region's fill
      color, and keep it grey if the two region areas are equal or the segment
      is not between two enclosed regions.
    - When a grey junction belongs to multiple segments, color that junction by
      the majority color among its incident segment colors, breaking ties by
      keeping grey.

    Key insight:
    - The marker fixes each compartment's fill color, but the boundary colors can
      only be decided after measuring the filled regions and comparing areas
      across each separating line segment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    region_ids, regions = _extract_regions(g)
    for region in regions:
        for r, c in region["cells"]:
            out[r, c] = int(region["color"])

    wall = g == GREY
    horiz, vert = _wall_orientations(wall)
    votes: dict[tuple[int, int], list[int]] = defaultdict(list)
    height, width = g.shape

    for row in range(height):
        col = 0
        while col < width:
            if not horiz[row, col]:
                col += 1
                continue
            start = col
            while col + 1 < width and horiz[row, col + 1]:
                col += 1
            end = col
            terminals = [start]
            terminals.extend(pos for pos in range(start + 1, end) if vert[row, pos])
            terminals.append(end)
            terminals = sorted(set(terminals))
            for left, right in zip(terminals, terminals[1:]):
                color = _segment_color_horizontal(region_ids, regions, row, left, right)
                for x in range(left, right + 1):
                    votes[(row, x)].append(color)
            col += 1

    for col in range(width):
        row = 0
        while row < height:
            if not vert[row, col]:
                row += 1
                continue
            start = row
            while row + 1 < height and vert[row + 1, col]:
                row += 1
            end = row
            terminals = [start]
            terminals.extend(pos for pos in range(start + 1, end) if horiz[pos, col])
            terminals.append(end)
            terminals = sorted(set(terminals))
            for top, bottom in zip(terminals, terminals[1:]):
                color = _segment_color_vertical(region_ids, regions, col, top, bottom)
                for y in range(top, bottom + 1):
                    votes[(y, col)].append(color)
            row += 1

    for (r, c), colors in votes.items():
        out[r, c] = _majority_color(colors)

    return out


def _build_partition_grid(
    rng: random.Random,
    n_rows: int,
    n_cols: int,
    row_heights: list[int],
    col_widths: list[int],
) -> np.ndarray:
    frame_height = sum(row_heights) + n_rows + 1
    frame_width = sum(col_widths) + n_cols + 1
    max_top = min(MARGIN_MAX, 30 - frame_height - MARGIN_MIN)
    max_left = min(MARGIN_MAX, 30 - frame_width - MARGIN_MIN)
    if max_top < MARGIN_MIN or max_left < MARGIN_MIN:
        raise RuntimeError("frame does not fit within ARC size limit")

    top_margin = rng.randint(MARGIN_MIN, max_top)
    bottom_margin = rng.randint(MARGIN_MIN, min(MARGIN_MAX, 30 - frame_height - top_margin))
    left_margin = rng.randint(MARGIN_MIN, max_left)
    right_margin = rng.randint(MARGIN_MIN, min(MARGIN_MAX, 30 - frame_width - left_margin))

    height = frame_height + top_margin + bottom_margin
    width = frame_width + left_margin + right_margin
    grid = np.full((height, width), BLACK, dtype=int)

    row_lines = [top_margin]
    for band in row_heights:
        row_lines.append(row_lines[-1] + band + 1)
    col_lines = [left_margin]
    for band in col_widths:
        col_lines.append(col_lines[-1] + band + 1)

    grid[row_lines[0] : row_lines[-1] + 1, col_lines] = GREY
    grid[row_lines, col_lines[0] : col_lines[-1] + 1] = GREY

    region_colors = rng.sample(list(MARKER_COLORS), n_rows * n_cols)
    color_idx = 0
    for row_idx in range(n_rows):
        r0 = row_lines[row_idx] + 1
        r1 = row_lines[row_idx + 1]
        for col_idx in range(n_cols):
            c0 = col_lines[col_idx] + 1
            c1 = col_lines[col_idx + 1]
            marker_r = rng.randint(r0, r1 - 1)
            marker_c = rng.randint(c0, c1 - 1)
            grid[marker_r, marker_c] = region_colors[color_idx]
            color_idx += 1

    return grid


def _generated_grid_is_valid(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    try:
        out = main(grid)
    except AssertionError:
        return False

    if out.shape != grid.shape:
        return False

    return bool(np.count_nonzero(out != grid) > 0)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        n_rows, n_cols = rng.choice(REGION_COUNT_OPTIONS)
        mode = rng.choice(BRANCH_MODES)
        row_heights = _sample_band_sizes(rng, n_rows, require_equal=(mode == "equal_rows"))
        col_widths = _sample_band_sizes(rng, n_cols, require_equal=(mode == "equal_cols"))

        grid = _build_partition_grid(rng, n_rows, n_cols, row_heights, col_widths)
        if _generated_grid_is_valid(grid):
            return grid

    raise RuntimeError("failed to generate a valid input")
