from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
LEGEND_COLOR = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
MIN_COLOR_COUNT = 3
MAX_COLOR_COUNT = 6
LEGEND_HEIGHT = 5
LEGEND_EXTRA_COLUMNS = (0, 2)
BAND_HEIGHT_RANGE = (2, 4)
TOP_GAP_RANGE = (1, 3)
INTER_GAP_RANGE = (0, 2)
BOTTOM_MARGIN_RANGE = (0, 2)
GRID_WIDTH_RANGE = (12, 30)
GENERATION_ATTEMPTS = 500


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _path_cells(
    top_point: tuple[int, int],
    bottom_point: tuple[int, int],
    horizontal_first: bool,
) -> set[tuple[int, int]]:
    (r0, c0), (r1, c1) = top_point, bottom_point
    cells: set[tuple[int, int]] = set()
    if horizontal_first:
        for c in range(min(c0, c1), max(c0, c1) + 1):
            cells.add((r0, c))
        for r in range(r0, r1 + 1):
            cells.add((r, c1))
    else:
        for r in range(r0, r1 + 1):
            cells.add((r, c0))
        for c in range(min(c0, c1), max(c0, c1) + 1):
            cells.add((r1, c))
    return cells


def _find_legend(grid: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], int, dict[int, bool]]:
    candidates: list[tuple[tuple[int, int], tuple[int, int, int, int], int, dict[int, bool]]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for comp in _components4(grid == color):
            if len(comp) < 10:
                continue
            rows = [r for r, _c in comp]
            cols = [c for _r, c in comp]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            if r1 - r0 + 1 != LEGEND_HEIGHT or c1 - c0 + 1 < 3:
                continue
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            if not (
                np.all(sub[0, :] == color)
                and np.all(sub[2, :] == color)
                and np.all(sub[4, :] == color)
                and np.all(sub[:, 0] == color)
                and np.all(sub[:, -1] == color)
            ):
                continue

            other_mask = (sub != bg) & (sub != color)
            allowed = np.zeros_like(other_mask, dtype=bool)
            allowed[1, 1:-1] = True
            allowed[3, 1:-1] = True
            if np.any(other_mask & ~allowed):
                continue

            mapping: dict[int, bool] = {}
            valid = True
            for local_r, horizontal_first in ((1, True), (3, False)):
                for local_c in range(1, sub.shape[1] - 1):
                    value = int(sub[local_r, local_c])
                    if value == bg:
                        continue
                    if value == color or value in mapping:
                        valid = False
                        break
                    mapping[value] = horizontal_first
                if not valid:
                    break
            if not valid or not mapping:
                continue
            candidates.append(((r0, c0), (r0, c0, r1, c1), color, mapping))

    assert candidates, "failed to locate the legend frame"
    candidates.sort(key=lambda item: (item[0][0], item[0][1], -len(item[3])))
    _anchor, bbox, legend_color, mapping = candidates[0]
    return bbox, legend_color, mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-colored endpoint pair with a one-bend L path decoded from the top-left legend.

    Input structure:
    - The background is a uniform dark color, and a small framed legend sits in the
      top-left area of the grid.
    - The legend is outlined by a single gray rectangular border and contains a gray
      horizontal divider, leaving one interior row above and one below that divider.
    - Each non-gray legend color appears exactly once inside the legend and exactly
      twice elsewhere in the grid as a separated endpoint pair.

    Rule:
    - Read the legend row containing each color: colors in the upper interior row use
      a horizontal-then-vertical L path, while colors in the lower interior row use a
      vertical-then-horizontal L path.
    - For every outside same-colored pair, choose the endpoint with the smaller row as
      the start and draw the corresponding one-pixel-wide L path including both endpoints.
    - Leave the legend border, divider, and all legend contents unchanged.

    Key insight:
    - The legend does not name destinations directly; it encodes the bend orientation
      for each color by whether that color is placed above or below the divider.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    (lr0, lc0, lr1, lc1), _legend_color, mapping = _find_legend(g, bg)

    out = g.copy()
    for color, horizontal_first in mapping.items():
        positions = [
            (int(r), int(c))
            for r, c in np.argwhere(g == color)
            if not (lr0 <= int(r) <= lr1 and lc0 <= int(c) <= lc1)
        ]
        assert len(positions) == 2, f"color {color} must have exactly two non-legend endpoints"
        positions.sort()
        (r0, c0), (r1, c1) = positions
        assert r0 != r1 and c0 != c1, "paired endpoints must differ in both row and column"

        for r, c in _path_cells((r0, c0), (r1, c1), horizontal_first):
            out[r, c] = color

    out[lr0 : lr1 + 1, lc0 : lc1 + 1] = g[lr0 : lr1 + 1, lc0 : lc1 + 1]
    return out


def _gen_attempt() -> np.ndarray | None:
    color_count = random.randint(MIN_COLOR_COUNT, MAX_COLOR_COUNT)
    colors = random.sample(COLOR_POOL, color_count)
    random.shuffle(colors)

    top_count = random.randint(1, color_count - 1)
    horizontal_colors = set(colors[:top_count])

    extra_cols = random.randint(*LEGEND_EXTRA_COLUMNS)
    inner_width = color_count + extra_cols
    legend_width = inner_width + 2

    band_heights = [random.randint(*BAND_HEIGHT_RANGE) for _ in range(color_count)]
    top_gap = random.randint(*TOP_GAP_RANGE)
    inter_gaps = [random.randint(*INTER_GAP_RANGE) for _ in range(color_count - 1)]
    bottom_margin = random.randint(*BOTTOM_MARGIN_RANGE)
    height = (
        LEGEND_HEIGHT
        + top_gap
        + sum(band_heights)
        + sum(inter_gaps)
        + bottom_margin
    )
    if height > 30:
        return None

    min_width = max(GRID_WIDTH_RANGE[0], legend_width)
    if min_width > GRID_WIDTH_RANGE[1]:
        return None
    width = random.randint(min_width, GRID_WIDTH_RANGE[1])

    grid = np.full((height, width), BG, dtype=int)
    expected = grid.copy()

    grid[0, :legend_width] = LEGEND_COLOR
    grid[2, :legend_width] = LEGEND_COLOR
    grid[4, :legend_width] = LEGEND_COLOR
    grid[:LEGEND_HEIGHT, 0] = LEGEND_COLOR
    grid[:LEGEND_HEIGHT, legend_width - 1] = LEGEND_COLOR
    expected[:LEGEND_HEIGHT, :legend_width] = grid[:LEGEND_HEIGHT, :legend_width]

    legend_columns = random.sample(range(1, legend_width - 1), color_count)
    for color, col in zip(colors, legend_columns):
        row = 1 if color in horizontal_colors else 3
        grid[row, col] = color
        expected[row, col] = color

    occupied = np.zeros_like(grid, dtype=bool)
    occupied[:LEGEND_HEIGHT, :legend_width] = True

    cursor = LEGEND_HEIGHT + top_gap
    band_order = random.sample(colors, len(colors))
    for index, color in enumerate(band_order):
        band_height = band_heights[index]
        band_top = cursor
        band_bottom = band_top + band_height - 1
        if band_bottom >= height:
            return None

        r0 = random.randint(band_top, band_bottom - 1)
        r1 = random.randint(r0 + 1, band_bottom)
        c0, c1 = random.sample(range(width), 2)

        cells = _path_cells((r0, c0), (r1, c1), color in horizontal_colors)
        if any(occupied[r, c] for r, c in cells):
            return None

        grid[r0, c0] = color
        grid[r1, c1] = color
        for r, c in cells:
            occupied[r, c] = True
            expected[r, c] = color

        cursor = band_bottom + 1
        if index < len(inter_gaps):
            cursor += inter_gaps[index]

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
