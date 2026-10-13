from __future__ import annotations

import random

import numpy as np


HorizontalSegment = tuple[int, int, int, int]  # color, row, c0, c1
VerticalSegment = tuple[int, int, int, int]  # color, col, r0, r1


def _bbox_from_points(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in points]
    cols = [c for _r, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _collect_segments(grid: np.ndarray) -> tuple[list[HorizontalSegment], list[VerticalSegment]]:
    g = np.asarray(grid, dtype=int)
    horizontal_owner = np.full(g.shape, -1, dtype=int)
    vertical_owner = np.full(g.shape, -1, dtype=int)
    horizontals: list[HorizontalSegment] = []
    verticals: list[VerticalSegment] = []

    for color in sorted(int(c) for c in np.unique(g) if int(c) != 0):
        coords = np.argwhere(g == color)
        assert coords.shape == (2, 2), "each non-black color must appear exactly twice"
        (r0, c0), (r1, c1) = coords.tolist()

        if r0 == r1:
            c0, c1 = sorted((int(c0), int(c1)))
            assert c0 < c1, "horizontal markers must be distinct"
            row = int(r0)
            for c in range(c0, c1 + 1):
                assert horizontal_owner[row, c] == -1, "horizontal segments must not overlap"
                horizontal_owner[row, c] = color
            horizontals.append((color, row, c0, c1))
        else:
            assert c0 == c1, "same-color markers must share a row or a column"
            r0, r1 = sorted((int(r0), int(r1)))
            assert r0 < r1, "vertical markers must be distinct"
            col = int(c0)
            for r in range(r0, r1 + 1):
                assert vertical_owner[r, col] == -1, "vertical segments must not overlap"
                vertical_owner[r, col] = color
            verticals.append((color, col, r0, r1))

    assert horizontals and verticals, "expected at least one horizontal and one vertical pair"
    return horizontals, verticals


def _render_lines(shape: tuple[int, int], horizontals: list[HorizontalSegment], verticals: list[VerticalSegment]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for color, row, c0, c1 in horizontals:
        out[row, c0 : c1 + 1] = color
    for color, col, r0, r1 in verticals:
        out[r0 : r1 + 1, col] = color
    return out


def _intersection_points(
    horizontals: list[HorizontalSegment], verticals: list[VerticalSegment]
) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for _h_color, row, c0, c1 in horizontals:
        for _v_color, col, r0, r1 in verticals:
            if c0 <= col <= c1 and r0 <= row <= r1:
                points.append((row, col))
    return points


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw aligned same-color segments, keep the vertical-colored crossings, then crop to the crossings' bounding box.

    Input structure:
    - A black background contains 4-6 non-black colors, each appearing in exactly two cells.
    - Each same-color pair is aligned either horizontally or vertically, so it defines one straight segment.
    - Horizontal pairs may cross vertical pairs, but same-orientation segments do not overlap.

    Rule:
    - For each same-color pair, draw the inclusive horizontal or vertical segment between its two markers.
    - Resolve every horizontal-vertical crossing by keeping the vertical segment's color at that cell.
    - Collect all crossing coordinates, compute their tight bounding box, and crop the fully drawn line image to that box.
    - Inside the crop, preserve every drawn line pixel and leave all remaining cells black.

    Key insight:
    - The crop is determined only by emergent crossing cells, not by the original endpoints and not by the union of all segments, so non-crossing lines can disappear completely.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    horizontals, verticals = _collect_segments(g)
    points = _intersection_points(horizontals, verticals)
    assert points, "expected at least one horizontal-vertical crossing"

    drawn = _render_lines(g.shape, horizontals, verticals)
    r0, c0, r1, c1 = _bbox_from_points(points)
    return drawn[r0 : r1 + 1, c0 : c1 + 1]


GRID_RANGE = (10, 16)
SEGMENT_COUNT_CHOICES = ((2, 2), (2, 3), (3, 2), (3, 3))
COLOR_POOL = tuple(range(1, 10))
SPAN_MIN = 4
GEN_ATTEMPTS = 500


def _random_span(limit: int, min_len: int = SPAN_MIN) -> tuple[int, int]:
    length = random.randint(min_len, limit)
    start = random.randint(0, limit - length)
    return start, start + length - 1


def _build_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_RANGE)
    width = random.randint(*GRID_RANGE)
    n_h, n_v = random.choice(SEGMENT_COUNT_CHOICES)
    isolated_orientation = random.choice(("h", "v"))

    core_h = n_h - (1 if isolated_orientation == "h" else 0)
    core_v = n_v - (1 if isolated_orientation == "v" else 0)
    if core_h < 1 or core_v < 1 or core_h * core_v < 2:
        return None

    if height < 8 or width < 8:
        return None

    row_pool = list(range(2, height - 2))
    col_pool = list(range(2, width - 2))
    if len(row_pool) < n_h or len(col_pool) < n_v:
        return None

    core_rows = sorted(random.sample(row_pool, core_h))
    core_cols = sorted(random.sample(col_pool, core_v))
    top = core_rows[0] - 1
    bottom = core_rows[-1] + 1
    left = core_cols[0] - 1
    right = core_cols[-1] + 1

    horizontals: list[HorizontalSegment] = []
    verticals: list[VerticalSegment] = []
    colors = random.sample(COLOR_POOL, n_h + n_v)
    color_i = 0

    for row in core_rows:
        horizontals.append((colors[color_i], row, left, right))
        color_i += 1
    for col in core_cols:
        verticals.append((colors[color_i], col, top, bottom))
        color_i += 1

    if isolated_orientation == "h":
        isolated_rows = [r for r in range(height) if r < top or r > bottom]
        if not isolated_rows:
            return None
        row = random.choice(isolated_rows)
        c0, c1 = _random_span(width)
        horizontals.append((colors[color_i], row, c0, c1))
    else:
        isolated_cols = [c for c in range(width) if c < left or c > right]
        if not isolated_cols:
            return None
        col = random.choice(isolated_cols)
        r0, r1 = _random_span(height)
        verticals.append((colors[color_i], col, r0, r1))

    grid = np.zeros((height, width), dtype=int)
    used_endpoints: set[tuple[int, int]] = set()

    for color, row, c0, c1 in horizontals:
        endpoints = ((row, c0), (row, c1))
        if endpoints[0] in used_endpoints or endpoints[1] in used_endpoints:
            return None
        used_endpoints.update(endpoints)
        grid[row, c0] = color
        grid[row, c1] = color

    for color, col, r0, r1 in verticals:
        endpoints = ((r0, col), (r1, col))
        if endpoints[0] in used_endpoints or endpoints[1] in used_endpoints:
            return None
        used_endpoints.update(endpoints)
        grid[r0, col] = color
        grid[r1, col] = color

    intersections = _intersection_points(horizontals, verticals)
    if len(intersections) < 2:
        return None

    per_segment_crossings: list[int] = []
    for _color, row, c0, c1 in horizontals:
        count = sum(c0 <= col <= c1 and r0 <= row <= r1 for _vc, col, r0, r1 in verticals)
        per_segment_crossings.append(count)
    for _color, col, r0, r1 in verticals:
        count = sum(c0 <= col <= c1 and r0 <= row <= r1 for _hc, row, c0, c1 in horizontals)
        per_segment_crossings.append(count)
    if 0 not in per_segment_crossings:
        return None

    r0, c0, r1, c1 = _bbox_from_points(intersections)
    drawn = _render_lines(grid.shape, horizontals, verticals)

    isolated_visible = False
    for color, row, hc0, hc1 in horizontals:
        if sum(hc0 <= col <= hc1 and vr0 <= row <= vr1 for _vc, col, vr0, vr1 in verticals) == 0:
            if not (r0 <= row <= r1):
                isolated_visible = True
    for color, col, vr0, vr1 in verticals:
        if sum(hc0 <= col <= hc1 and vr0 <= row <= vr1 for _hc, row, hc0, hc1 in horizontals) == 0:
            if not (c0 <= col <= c1):
                isolated_visible = True
    if not isolated_visible:
        return None

    try:
        solved = main(grid)
    except AssertionError:
        return None

    expected = drawn[r0 : r1 + 1, c0 : c1 + 1]
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
