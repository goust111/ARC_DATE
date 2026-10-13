from collections import Counter

import numpy as np

from arc_utils import in_bounds, neighbors4
import random as _gen_random


def _unique_least_frequent_color(g: np.ndarray) -> int:
    colors, counts = np.unique(g, return_counts=True)
    assert colors.size >= 2, "expected marker pixels embedded in colored regions"

    min_count = int(counts.min())
    candidates = [int(color) for color, count in zip(colors, counts) if int(count) == min_count]
    assert len(candidates) == 1, "expected a unique rare marker color"
    return candidates[0]


def _marker_region_color(g: np.ndarray, marker: int, r: int, c: int) -> int:
    adjacent = [
        int(g[rr, cc])
        for rr, cc in neighbors4(r, c)
        if in_bounds(g.shape, rr, cc) and int(g[rr, cc]) != marker
    ]
    assert adjacent, "each marker must touch its surrounding region color"

    ranked = Counter(adjacent).most_common()
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], (
        "expected each marker's surrounding region color to be unique"
    )
    return int(ranked[0][0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Return the color of the rectangular region that contains the most rare marker pixels.

    Input structure:
    - The grid is partitioned into large, mostly solid rectangular color regions.
    - One unique rare color appears only as marker pixels embedded inside those regions.
    - Each marker's 4-neighborhood identifies the base color of its containing region.

    Rule:
    - Find the unique least frequent color; these cells are the markers.
    - Count how many marker pixels belong to each surrounding region color.
    - Output a 1x1 grid containing the unique region color with the largest marker count.

    Key insight:
    - The markers are not output directly; they vote for the region they sit inside,
      and the winning region's base color is the label to return.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    marker = _unique_least_frequent_color(g)
    marker_positions = np.argwhere(g == marker)
    assert marker_positions.size > 0, "expected at least one marker pixel"

    marker_counts_by_region: Counter[int] = Counter()
    for r, c in marker_positions:
        region_color = _marker_region_color(g, marker, int(r), int(c))
        marker_counts_by_region[region_color] += 1

    ranked_regions = marker_counts_by_region.most_common()
    assert len(ranked_regions) == 1 or ranked_regions[0][1] > ranked_regions[1][1], (
        "expected a unique region with the most markers"
    )

    return np.array([[int(ranked_regions[0][0])]], dtype=int)

# --------------- generator (o5x) ---------------

# Canvas and BSP partition bounds.
GRID_ROWS = (16, 18)
GRID_COLS = (13, 19)
REGION_COUNTS = (3, 4)
MIN_REGION_SIDE = 4

# Marker count and placement bounds.
MARKER_COUNTS = (3, 11)
MARKER_BORDER_PAD = 1

# Palette and retry limits.
COLOR_POOL = range(1, 10)
GENERATION_RETRIES = 300


def _gen_rect_area(rect):
    r0, c0, r1, c1 = rect
    return (r1 - r0) * (c1 - c0)


def _gen_partition(rows, cols, n_regions):
    """Return half-open BSP rectangles that tile the grid, or None if unsplittable."""
    leaves = [(0, 0, rows, cols)]

    for _ in range(n_regions - 1):
        options = []
        for i, (r0, c0, r1, c1) in enumerate(leaves):
            if r1 - r0 >= 2 * MIN_REGION_SIDE:
                options.append((i, "h"))
            if c1 - c0 >= 2 * MIN_REGION_SIDE:
                options.append((i, "v"))
        if not options:
            return None

        leaf_i, axis = _gen_random.choice(options)
        r0, c0, r1, c1 = leaves.pop(leaf_i)
        if axis == "h":
            cut = _gen_random.randint(r0 + MIN_REGION_SIDE, r1 - MIN_REGION_SIDE)
            split = [(r0, c0, cut, c1), (cut, c0, r1, c1)]
        else:
            cut = _gen_random.randint(c0 + MIN_REGION_SIDE, c1 - MIN_REGION_SIDE)
            split = [(r0, c0, r1, cut), (r0, cut, r1, c1)]

        if _gen_random.random() < 0.5:
            split.reverse()
        leaves.extend(split)

    return leaves


def _gen_marker_capacity(rect):
    r0, c0, r1, c1 = rect
    interior_h = max(0, r1 - r0 - 2 * MARKER_BORDER_PAD)
    interior_w = max(0, c1 - c0 - 2 * MARKER_BORDER_PAD)
    return (interior_h * interior_w + 1) // 2


def _gen_marker_allocations(caps, total):
    n = len(caps)
    allocations = []

    def rec(i, remaining, current):
        if i == n:
            if remaining != 0 or sum(v > 0 for v in current) < 2:
                return
            top = max(current)
            if current.count(top) == 1:
                allocations.append(tuple(current))
            return

        for count in range(min(caps[i], remaining) + 1):
            current.append(count)
            rec(i + 1, remaining - count, current)
            current.pop()

    rec(0, total, [])
    return allocations


def _gen_choose_marker_counts(rects):
    """Choose counts with >=2 marked regions and a unique winning region."""
    by_total = {}
    capacities = [_gen_marker_capacity(rect) for rect in rects]
    areas = [_gen_rect_area(rect) for rect in rects]

    for total in range(MARKER_COUNTS[0], MARKER_COUNTS[1] + 1):
        caps = [max(0, min(cap, area - total - 1)) for cap, area in zip(capacities, areas)]
        allocations = _gen_marker_allocations(caps, total)
        if not allocations:
            return None
        by_total[total] = allocations

    total = _gen_random.randint(*MARKER_COUNTS)
    return _gen_random.choice(by_total[total])


def _gen_marker_cells(rect, count):
    if count == 0:
        return []

    r0, c0, r1, c1 = rect
    by_parity = [[], []]
    for r in range(r0 + MARKER_BORDER_PAD, r1 - MARKER_BORDER_PAD):
        for c in range(c0 + MARKER_BORDER_PAD, c1 - MARKER_BORDER_PAD):
            by_parity[(r + c) & 1].append((r, c))

    choices = [cells for cells in by_parity if len(cells) >= count]
    if not choices:
        return None
    return _gen_random.sample(_gen_random.choice(choices), count)


def _gen_render_regions(rows, cols, rects, region_colors):
    grid = np.zeros((rows, cols), dtype=int)
    for rect, color in zip(rects, region_colors):
        r0, c0, r1, c1 = rect
        grid[r0:r1, c0:c1] = color
    return grid


def _gen_place_markers(grid, rects, marker_counts, marker_color):
    for rect, count in zip(rects, marker_counts):
        cells = _gen_marker_cells(rect, count)
        if cells is None:
            return False
        for r, c in cells:
            grid[r, c] = marker_color
    return True


def _gen_oracle_checked_grid(grid, expected):
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def _gen_attempt():
    rows = _gen_random.randint(*GRID_ROWS)
    cols = _gen_random.randint(*GRID_COLS)
    n_regions = _gen_random.randint(*REGION_COUNTS)

    rects = _gen_partition(rows, cols, n_regions)
    if rects is None:
        return None

    marker_counts = _gen_choose_marker_counts(rects)
    if marker_counts is None:
        return None

    colors = _gen_random.sample(COLOR_POOL, n_regions + 1)
    marker_color = colors[0]
    region_colors = colors[1:]
    winner_i = max(range(n_regions), key=lambda i: marker_counts[i])
    expected = np.array([[region_colors[winner_i]]], dtype=int)
    grid = _gen_render_regions(rows, cols, rects, region_colors)

    if not _gen_place_markers(grid, rects, marker_counts, marker_color):
        return None

    return _gen_oracle_checked_grid(grid, expected)


def generate_input():
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid de1cd16c input")
