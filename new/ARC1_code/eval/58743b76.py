from __future__ import annotations

from collections import Counter

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg
import random


def _unique_divider_color(g: np.ndarray, bg: int) -> int:
    counts = Counter(map(int, g.ravel()))
    foreground = [color for color in counts if color != bg]
    assert foreground, "expected non-background divider and legend colors"
    max_count = max(counts[color] for color in foreground)
    candidates = [color for color in foreground if counts[color] == max_count]
    assert len(candidates) == 1, "expected a unique most common non-background divider color"
    return int(candidates[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the data pixels in each quadrant using the matching entry of the 2x2 legend.

    Input structure:
    - The grid has a dominant background, a thick divider color, one 2x2 corner legend,
      and an opposite rectangular data region.
    - The data region contains background plus one foreground data color.

    Rule:
    - Split the data region into four equal quadrants.
    - For every data-colored pixel, replace it with the color in the same top/bottom and
      left/right position of the 2x2 legend.
    - Keep the background, divider, and legend unchanged.

    Key insight:
    - The 2x2 legend is not moved or copied as a shape; its cells are a quadrant color
      key for the separated data field.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    bg = get_bg(g)
    divider = _unique_divider_color(g, bg)

    non_field_mask = (g != bg) & (g != divider)
    legend_boxes = []
    for coords in components_mask(non_field_mask, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(coords)
        is_corner = (r0 == 0 or r1 == h - 1) and (c0 == 0 or c1 == w - 1)
        if coords.shape[0] == 4 and (r1 - r0 + 1, c1 - c0 + 1) == (2, 2) and is_corner:
            legend_boxes.append((r0, c0, r1, c1))
    assert len(legend_boxes) == 1, "expected exactly one 2x2 legend"
    lr0, lc0, lr1, lc1 = legend_boxes[0]
    legend = g[lr0 : lr1 + 1, lc0 : lc1 + 1]
    assert np.all((legend != bg) & (legend != divider)), "legend should contain only paint colors"

    at_top = lr0 == 0
    at_bottom = lr1 == h - 1
    at_left = lc0 == 0
    at_right = lc1 == w - 1
    assert at_top != at_bottom and at_left != at_right, "legend must sit in one corner"

    data_r0, data_r1 = (lr1 + 1, h - 1) if at_top else (0, lr0 - 1)
    data_c0, data_c1 = (lc1 + 1, w - 1) if at_left else (0, lc0 - 1)
    assert data_r0 <= data_r1 and data_c0 <= data_c1, "expected a nonempty data region"

    assert np.all(g[lr0 : lr1 + 1, data_c0 : data_c1 + 1] == divider)
    assert np.all(g[data_r0 : data_r1 + 1, lc0 : lc1 + 1] == divider)

    data = g[data_r0 : data_r1 + 1, data_c0 : data_c1 + 1]
    data_colors = [int(color) for color in set(map(int, data.ravel())) if color != bg]
    assert len(data_colors) == 1, "data region should contain one foreground color"
    data_color = data_colors[0]
    assert data_color != divider, "divider should not appear inside the data region"

    data_h, data_w = data.shape
    assert data_h % 2 == 0 and data_w % 2 == 0, "data region must split into four equal quadrants"
    half_h, half_w = data_h // 2, data_w // 2

    out = g.copy()
    for qr in range(2):
        for qc in range(2):
            rs = slice(data_r0 + qr * half_h, data_r0 + (qr + 1) * half_h)
            cs = slice(data_c0 + qc * half_w, data_c0 + (qc + 1) * half_w)
            mask = g[rs, cs] == data_color
            block = out[rs, cs]
            block[mask] = int(legend[qr, qc])
    return out

# --------------- generator (o5x) ---------------

GRID_SIZES = (10, 12, 14)
BACKGROUND_COLOR = 0
DIVIDER_COLOR = 8
PAINT_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
LEGEND_SIZE = 2
QUADRANT_AXIS_COUNT = 2
QUADRANT_POSITIONS = ((0, 0), (0, 1), (1, 0), (1, 1))
LEGEND_CELL_COUNT = LEGEND_SIZE * LEGEND_SIZE
MIN_DENSITY_PERCENT = 6
MAX_DENSITY_PERCENT = 25
MIN_PIXELS_PER_QUADRANT = 1
MAX_PIXELS_PER_QUADRANT = 5
MAX_TOTAL_DATA_PIXELS = 20
GENERATION_RETRIES = 300


def _gen_regions(n: int, corner: int) -> tuple[slice, slice, slice, slice]:
    top = corner < QUADRANT_AXIS_COUNT
    left = corner % QUADRANT_AXIS_COUNT == 0
    legend_rs = slice(0, LEGEND_SIZE) if top else slice(n - LEGEND_SIZE, n)
    legend_cs = slice(0, LEGEND_SIZE) if left else slice(n - LEGEND_SIZE, n)
    data_rs = slice(LEGEND_SIZE, n) if top else slice(0, n - LEGEND_SIZE)
    data_cs = slice(LEGEND_SIZE, n) if left else slice(0, n - LEGEND_SIZE)
    return legend_rs, legend_cs, data_rs, data_cs


def _gen_expected_output(inp: np.ndarray, legend: np.ndarray, data_color: int, data_rs: slice, data_cs: slice) -> np.ndarray:
    out = inp.copy()
    half_h = (data_rs.stop - data_rs.start) // QUADRANT_AXIS_COUNT
    half_w = (data_cs.stop - data_cs.start) // QUADRANT_AXIS_COUNT
    for qr, qc in QUADRANT_POSITIONS:
        rs = slice(data_rs.start + qr * half_h, data_rs.start + (qr + 1) * half_h)
        cs = slice(data_cs.start + qc * half_w, data_cs.start + (qc + 1) * half_w)
        block = out[rs, cs]
        block[block == data_color] = int(legend[qr, qc])
    return out


def _gen_quadrant_counts(data_area: int) -> list[int]:
    min_total = max(LEGEND_CELL_COUNT * MIN_PIXELS_PER_QUADRANT, (MIN_DENSITY_PERCENT * data_area + 99) // 100)
    max_total = min(MAX_TOTAL_DATA_PIXELS, (MAX_DENSITY_PERCENT * data_area) // 100)
    total = random.randint(min_total, max_total)
    counts = [MIN_PIXELS_PER_QUADRANT] * LEGEND_CELL_COUNT
    for _ in range(total - LEGEND_CELL_COUNT):
        choices = [i for i, count in enumerate(counts) if count < MAX_PIXELS_PER_QUADRANT]
        counts[random.choice(choices)] += 1
    random.shuffle(counts)
    return counts


def _paint_data_region(grid: np.ndarray, data_rs: slice, data_cs: slice, data_color: int) -> None:
    """Place data-color pixels in all quadrants while avoiding the outer data corner."""
    data_h = data_rs.stop - data_rs.start
    data_w = data_cs.stop - data_cs.start
    half_h = data_h // QUADRANT_AXIS_COUNT
    half_w = data_w // QUADRANT_AXIS_COUNT
    outer_r = 0 if data_rs.start == 0 else data_h - 1
    outer_c = 0 if data_cs.start == 0 else data_w - 1
    forbidden_corner_cell = (outer_r, outer_c)
    quadrant_counts = _gen_quadrant_counts(data_h * data_w)
    for q_index, (qr, qc) in enumerate(QUADRANT_POSITIONS):
        r0 = qr * half_h
        c0 = qc * half_w
        cells = [(r0 + dr, c0 + dc) for dr in range(half_h) for dc in range(half_w)]
        if forbidden_corner_cell in cells:
            cells.remove(forbidden_corner_cell)
        for rr, cc in random.sample(cells, quadrant_counts[q_index]):
            grid[data_rs.start + rr, data_cs.start + cc] = data_color


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(GRID_SIZES)
    corner = random.randrange(LEGEND_CELL_COUNT)
    legend_rs, legend_cs, data_rs, data_cs = _gen_regions(n, corner)
    grid = np.full((n, n), BACKGROUND_COLOR, dtype=int)
    grid[legend_rs, data_cs] = DIVIDER_COLOR
    grid[data_rs, legend_cs] = DIVIDER_COLOR
    colors = random.sample(PAINT_COLORS, LEGEND_CELL_COUNT)
    legend = np.array(colors, dtype=int).reshape(LEGEND_SIZE, LEGEND_SIZE)
    data_color = random.choice(colors)
    grid[legend_rs, legend_cs] = legend

    _paint_data_region(grid, data_rs, data_cs, data_color)

    expected = _gen_expected_output(grid, legend, data_color, data_rs, data_cs)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 58743b76 input")
