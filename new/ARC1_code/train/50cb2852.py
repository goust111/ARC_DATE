from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox_h, bbox_w, find_components, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the one-pixel-inset interior of every solid foreground rectangle to teal.

    Input structure
    - The grid has one dominant background color.
    - Non-background pixels form separated 4-connected monochrome objects.
    - Each foreground object is a filled axis-aligned rectangle at least 3 cells tall and wide.

    Rule
    - Preserve every original pixel except the strict interior of each rectangle.
    - For every foreground rectangle, replace all cells not on its one-pixel border with teal.

    Key insight
    - The output marks rectangle interiors, not whole objects: the original colors remain only as borders.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert min(g.shape) > 0, "expected a nonempty grid"

    bg = get_bg(g)
    assert bg != Color.TEAL, "expected teal to be the added interior color, not background"
    assert not np.any(g == Color.TEAL), "expected no existing teal pixels"

    objects = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert objects, "expected at least one foreground rectangle"

    out = g.copy()
    for obj in objects:
        box = obj["bbox"]
        height = bbox_h(box)
        width = bbox_w(box)
        assert height >= 3 and width >= 3, "expected each rectangle to have an interior"
        assert int(obj["area"]) == height * width, "expected each object to fill its bbox"

        r0, c0, r1, c1 = map(int, box)
        out[r0 + 1 : r1, c0 + 1 : c1] = Color.TEAL

    return out

# --------------- generator (o5x) ---------------

# Canvas and rectangle bounds use np.random.randint's exclusive upper endpoint.
GRID_H_RANGE, GRID_W_RANGE = (12, 15), (11, 16)
RECT_COUNT_RANGE, RECT_H_RANGE, RECT_W_RANGE = (2, 5), (3, 8), (3, 9)
# Rectangle colors and expected interior recolor.
BACKGROUND_COLOR, INTERIOR_COLOR = 0, int(Color.TEAL)
FOREGROUND_COLORS = np.array([1, 2, 3], dtype=int)
DISTINCT_COLOR_MIN, DISTINCT_COLOR_MAX = 2, 3
COLOR_BIN_COUNT = 10
# Layout spacing and retry controls.
RECT_GAP, RECT_BORDER = 1, 1
GEN_MAX_ATTEMPTS = 300

def _gen_feasible_locations(occupied, height, width):
    rows, cols = occupied.shape
    locations = []
    for r in range(rows - height + 1):
        for c in range(cols - width + 1):
            r0 = max(0, r - RECT_GAP)
            c0 = max(0, c - RECT_GAP)
            r1 = min(rows, r + height + RECT_GAP)
            c1 = min(cols, c + width + RECT_GAP)
            if not occupied[r0:r1, c0:c1].any():
                locations.append((r, c))
    return locations

def _gen_pick_colors(rects):
    palette = np.random.permutation(FOREGROUND_COLORS).tolist()
    distinct = int(np.random.randint(DISTINCT_COLOR_MIN, min(DISTINCT_COLOR_MAX, len(rects)) + 1))
    active = palette[:distinct]
    totals = {color: 0 for color in active}
    colors = [None] * len(rects)

    order = sorted(range(len(rects)), key=lambda i: rects[i][0] * rects[i][1], reverse=True)
    for j, idx in enumerate(order):
        area = rects[idx][0] * rects[idx][1]
        if j < distinct:
            color = active[j]
        else:
            least = min(totals.values())
            choices = [color for color in active if totals[color] == least]
            color = int(np.random.choice(choices))
        colors[idx] = color
        totals[color] += area
    return colors

def _gen_place_rectangles(shape, rects):
    """Return top-left placements in rect order, or None if one cannot fit."""
    occupied = np.zeros(shape, dtype=bool)
    placements = [None] * len(rects)
    order = sorted(range(len(rects)), key=lambda i: rects[i][0] * rects[i][1], reverse=True)
    for idx in order:
        rect_h, rect_w = rects[idx]
        locations = _gen_feasible_locations(occupied, rect_h, rect_w)
        if not locations:
            return None
        loc_i = int(np.random.randint(0, len(locations)))
        r, c = locations[loc_i]
        occupied[r : r + rect_h, c : c + rect_w] = True
        placements[idx] = (r, c)
    return placements

def _gen_render_case(shape, rects, colors, placements):
    """Render the input grid and the oracle output expected from main()."""
    grid = np.full(shape, BACKGROUND_COLOR, dtype=int)
    expected = grid.copy()
    for (rect_h, rect_w), color, (r, c) in zip(rects, colors, placements):
        grid[r : r + rect_h, c : c + rect_w] = color
        expected[r : r + rect_h, c : c + rect_w] = color
        expected[r + RECT_BORDER : r + rect_h - RECT_BORDER, c + RECT_BORDER : c + rect_w - RECT_BORDER] = INTERIOR_COLOR
    return grid, expected

def _gen_attempt():
    height = int(np.random.randint(*GRID_H_RANGE))
    width = int(np.random.randint(*GRID_W_RANGE))
    rect_count = int(np.random.randint(*RECT_COUNT_RANGE))
    rects = [
        (int(np.random.randint(*RECT_H_RANGE)), int(np.random.randint(*RECT_W_RANGE)))
        for _ in range(rect_count)
    ]
    colors = _gen_pick_colors(rects)
    shape = (height, width)
    placements = _gen_place_rectangles(shape, rects)
    if placements is None:
        return None
    grid, expected = _gen_render_case(shape, rects, colors, placements)
    counts = np.bincount(grid.ravel(), minlength=COLOR_BIN_COUNT)
    if counts[BACKGROUND_COLOR] <= max(counts[color] for color in FOREGROUND_COLORS):
        return None
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid

def generate_input():
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 50cb2852 input")
