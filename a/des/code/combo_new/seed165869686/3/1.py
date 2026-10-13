from __future__ import annotations

from collections import deque
import random

import numpy as np


# Fixed visible color cycle from the puzzle description:
# red -> green -> blue -> yellow -> maroon -> red.
CYCLE = (2, 3, 1, 4, 9)
NEXT_COLOR = {2: 3, 3: 1, 1: 4, 4: 9, 9: 2}

# Generator configuration.
GREY = 5
BACKGROUND_CHOICES = (0, 6, 7, 8)
BRANCHES = ("horizontal", "vertical", "grid")
MAX_SIZE = 30
MAX_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            comp: list[tuple[int, int]] = []
            queue = deque([(r, c)])
            seen[r, c] = True

            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            comps.append(comp)

    return comps


def _touches_boundary(comp: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)


def _describe_regions(
    grid: np.ndarray, wall_color: int, background_color: int
) -> list[tuple[list[tuple[int, int]], int]] | None:
    mask = grid != wall_color
    regions: list[tuple[list[tuple[int, int]], int]] = []
    boundary_components = 0

    for comp in _components(mask):
        values = [int(grid[r, c]) for r, c in comp]
        markers = [(r, c) for r, c in comp if int(grid[r, c]) in NEXT_COLOR]
        touches_boundary = _touches_boundary(comp, grid.shape)

        if touches_boundary:
            boundary_components += 1
            if any(v != background_color for v in values):
                return None
            continue

        if len(markers) != 1:
            return None

        marker_r, marker_c = markers[0]
        marker_color = int(grid[marker_r, marker_c])
        for r, c in comp:
            value = int(grid[r, c])
            if (r, c) == (marker_r, marker_c):
                continue
            if value != background_color:
                return None
        regions.append((comp, marker_color))

    if boundary_components == 0 or not regions:
        return None

    return regions


def _infer_roles(grid: np.ndarray) -> tuple[int, int, list[tuple[list[tuple[int, int]], int]]]:
    colors = [int(v) for v in np.unique(grid)]
    non_cycle = [color for color in colors if color not in NEXT_COLOR]
    assert len(non_cycle) == 2, "expected exactly one background color and one wall color outside the marker cycle"

    candidates: list[tuple[int, int, list[tuple[list[tuple[int, int]], int]]]] = []
    for wall_color in non_cycle:
        background_color = next(color for color in non_cycle if color != wall_color)
        regions = _describe_regions(grid, wall_color, background_color)
        if regions is not None:
            candidates.append((wall_color, background_color, regions))

    assert candidates, "failed to infer wall/background roles"
    if len(candidates) > 1:
        # Prefer the candidate that explains the most enclosed regions.
        candidates.sort(key=lambda item: len(item[2]), reverse=True)
        assert len(candidates[0][2]) > len(candidates[1][2]), "role inference is ambiguous"

    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed rectangle with the next color in a fixed marker color cycle.

    Input structure:
    - The grid has one background color and one line color that draws horizontal
      and vertical walls.
    - Those wall segments create several closed rectangular regions, while the
      outside background remains connected to the grid boundary.
    - Each enclosed region contains exactly one marker pixel colored from the
      visible cycle red, green, blue, yellow, maroon.

    Rule:
    - Identify the wall color and all non-wall connected regions.
    - Ignore every non-wall component that touches the grid boundary; these are
      outside background, not closed rooms.
    - For each enclosed component, read its unique marker color and replace the
      entire component, including the marker itself, with the next color in the
      cycle red -> green -> blue -> yellow -> maroon -> red.
    - Keep all wall pixels unchanged.

    Key insight:
    - The marker is only a pointer into a visible color ring; the actual fill
      color is its successor, applied to the whole enclosed region.
    """
    g = np.asarray(grid, dtype=int)
    wall_color, _background_color, regions = _infer_roles(g)

    out = g.copy()
    for comp, marker_color in regions:
        fill_color = NEXT_COLOR[marker_color]
        for r, c in comp:
            out[r, c] = fill_color

    # Preserve walls explicitly for clarity.
    out[g == wall_color] = wall_color
    return out


def _span_range(count: int) -> tuple[int, int]:
    if count == 1:
        return 4, 10
    if count == 2:
        return 3, 8
    if count == 3:
        return 3, 6
    if count == 4:
        return 2, 5
    if count == 5:
        return 2, 4
    raise ValueError(f"unsupported segment count: {count}")


def _sample_spans(count: int, rng: random.Random) -> list[int]:
    lo, hi = _span_range(count)
    return [rng.randint(lo, hi) for _ in range(count)]


def _sample_margin(remaining: int, rng: random.Random) -> tuple[int, int] | None:
    if remaining < 2:
        return None
    max_side = min(4, remaining - 1)
    first = rng.randint(1, max_side)
    max_second = min(4, remaining - first)
    if max_second < 1:
        return None
    second = rng.randint(1, max_second)
    return first, second


def _choose_layout(rng: random.Random) -> tuple[int, int]:
    branch = rng.choice(BRANCHES)
    if branch == "horizontal":
        return 1, rng.randint(2, 5)
    if branch == "vertical":
        return rng.randint(2, 5), 1
    return 2, 2


def _paint_layout(
    background_color: int,
    row_heights: list[int],
    col_widths: list[int],
    margins: tuple[int, int, int, int],
    marker_colors: list[int],
    rng: random.Random,
) -> np.ndarray:
    top, bottom, left, right = margins
    rows = len(row_heights)
    cols = len(col_widths)

    structure_h = sum(row_heights) + rows + 1
    structure_w = sum(col_widths) + cols + 1
    grid_h = top + structure_h + bottom
    grid_w = left + structure_w + right
    grid = np.full((grid_h, grid_w), background_color, dtype=int)

    frame_top = top
    frame_left = left
    frame_bottom = top + structure_h - 1
    frame_right = left + structure_w - 1

    grid[frame_top, frame_left : frame_right + 1] = GREY
    grid[frame_bottom, frame_left : frame_right + 1] = GREY
    grid[frame_top : frame_bottom + 1, frame_left] = GREY
    grid[frame_top : frame_bottom + 1, frame_right] = GREY

    row_starts: list[int] = []
    current_r = frame_top + 1
    for idx, height in enumerate(row_heights):
        row_starts.append(current_r)
        current_r += height
        if idx != rows - 1:
            grid[current_r, frame_left : frame_right + 1] = GREY
            current_r += 1

    col_starts: list[int] = []
    current_c = frame_left + 1
    for idx, width in enumerate(col_widths):
        col_starts.append(current_c)
        current_c += width
        if idx != cols - 1:
            grid[frame_top : frame_bottom + 1, current_c] = GREY
            current_c += 1

    color_idx = 0
    for r_idx, region_top in enumerate(row_starts):
        for c_idx, region_left in enumerate(col_starts):
            region_bottom = region_top + row_heights[r_idx] - 1
            region_right = region_left + col_widths[c_idx] - 1
            marker_r = rng.randint(region_top, region_bottom)
            marker_c = rng.randint(region_left, region_right)
            grid[marker_r, marker_c] = marker_colors[color_idx]
            color_idx += 1

    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    background_color = rng.choice(BACKGROUND_CHOICES)
    rows, cols = _choose_layout(rng)
    region_count = rows * cols

    row_heights = _sample_spans(rows, rng)
    col_widths = _sample_spans(cols, rng)

    structure_h = sum(row_heights) + rows + 1
    structure_w = sum(col_widths) + cols + 1

    vertical_margin = _sample_margin(MAX_SIZE - structure_h, rng)
    horizontal_margin = _sample_margin(MAX_SIZE - structure_w, rng)
    if vertical_margin is None or horizontal_margin is None:
        return None

    top, bottom = vertical_margin
    left, right = horizontal_margin
    marker_colors = rng.sample(list(CYCLE), region_count)
    grid = _paint_layout(
        background_color=background_color,
        row_heights=row_heights,
        col_widths=col_widths,
        margins=(top, bottom, left, right),
        marker_colors=marker_colors,
        rng=rng,
    )

    try:
        wall_color, inferred_background, regions = _infer_roles(grid)
    except AssertionError:
        return None

    if wall_color != GREY or inferred_background != background_color:
        return None
    if len(regions) != region_count:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
