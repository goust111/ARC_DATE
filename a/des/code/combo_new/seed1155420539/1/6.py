from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
MARKER_COLORS = (2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 300


def _stripe_span(row: np.ndarray) -> tuple[int, int, list[tuple[int, int]]]:
    """Return the continuous stripe span and its non-blue marker crossings."""
    blue_cols = np.flatnonzero(row == BLUE)
    assert blue_cols.size > 0, "stripe rows must contain blue pixels"

    non_bg = np.flatnonzero(row != BG)
    c0 = int(non_bg[0])
    c1 = int(non_bg[-1])
    assert c1 > c0, "stripe must have positive width"
    assert bool(np.all(row[c0 : c1 + 1] != BG)), "stripe span must be continuous"

    markers: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        value = int(row[c])
        if value == BLUE:
            continue
        assert value != BG, "marker crossing cannot be background"
        markers.append((c, value))

    assert markers, "each stripe must be crossed by at least one marker line"
    return c0, c1, markers


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor horizontal blue stripes by the colors of their vertical marker crossings.

    Input structure:
    - The background is black.
    - One or more one-pixel-high blue horizontal stripes run across most of the
      grid.
    - Non-blue vertical marker segments cross every stripe; at each crossing the
      marker color overwrites the blue stripe pixel.

    Rule:
    - For each blue stripe row, order its marker crossings from left to right.
    - Paint the segment before the first marker with the first marker's color,
      and the segment after the last marker with the last marker's color.
    - Between two adjacent markers, paint the intervening stripe cells with that
      marker color only when the two marker colors match; otherwise leave those
      cells blue.
    - Marker crossing pixels themselves, and all pixels away from stripe rows,
      remain unchanged.

    Key insight:
    - The vertical markers are color injectors for stripe intervals, and the
      interior intervals require a same-color test on the two neighboring
      markers instead of copying either marker unconditionally.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    stripe_rows = [int(r) for r in np.flatnonzero(np.any(g == BLUE, axis=1))]
    assert stripe_rows, "expected at least one blue stripe row"

    out = g.copy()
    for r in stripe_rows:
        c0, c1, markers = _stripe_span(g[r])
        markers = sorted(markers)

        first_c, first_color = markers[0]
        if c0 < first_c:
            out[r, c0:first_c] = first_color

        for (left_c, left_color), (right_c, right_color) in zip(markers, markers[1:]):
            if left_c + 1 >= right_c:
                continue
            fill = left_color if left_color == right_color else BLUE
            out[r, left_c + 1 : right_c] = fill

        last_c, last_color = markers[-1]
        if last_c < c1:
            out[r, last_c + 1 : c1 + 1] = last_color

    return out


def _sample_band_rows(height: int, count: int) -> list[int] | None:
    choices = list(range(2, height - 2))
    if len(choices) < count:
        return None

    for _ in range(100):
        rows = sorted(random.sample(choices, count))
        if all(b - a >= 2 for a, b in zip(rows, rows[1:])):
            return rows
    return None


def _sample_marker_columns(c0: int, c1: int, count: int) -> list[int] | None:
    choices = list(range(c0 + 1, c1))
    if len(choices) < count:
        return None

    for _ in range(100):
        cols = sorted(random.sample(choices, count))
        if all(b - a >= 2 for a, b in zip(cols, cols[1:])):
            return cols
    return None


def _sample_marker_colors(count: int) -> list[int]:
    for _ in range(100):
        colors = [random.choice(MARKER_COLORS)]
        for _i in range(count - 1):
            if random.random() < 0.5:
                colors.append(colors[-1])
            else:
                choices = [color for color in MARKER_COLORS if color != colors[-1]]
                colors.append(random.choice(choices))

        has_same = any(a == b for a, b in zip(colors, colors[1:]))
        has_diff = any(a != b for a, b in zip(colors, colors[1:]))
        if has_same and has_diff:
            return colors

    a, b = random.sample(MARKER_COLORS, 2)
    colors = [a, a, b]
    while len(colors) < count:
        colors.append(random.choice(MARKER_COLORS))
    return colors


def _marker_extent(height: int, top_row: int, bottom_row: int) -> tuple[int, int]:
    mode = random.choice(("top", "bottom", "both"))
    top_space = top_row
    bottom_space = height - 1 - bottom_row

    if mode == "top":
        above = random.randint(1, min(3, top_space))
        below = random.randint(0, min(1, bottom_space))
    elif mode == "bottom":
        above = random.randint(0, min(1, top_space))
        below = random.randint(1, min(3, bottom_space))
    else:
        above = random.randint(1, min(3, top_space))
        below = random.randint(1, min(3, bottom_space))

    return top_row - above, bottom_row + below


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(10, 24)
    width = random.randint(14, 30)

    band_count = random.randint(2, min(5, max(2, (height - 3) // 2)))
    band_rows = _sample_band_rows(height, band_count)
    if band_rows is None:
        return None

    left_margin = random.randint(0, min(3, width - 10))
    right_margin = random.randint(0, min(3, width - left_margin - 10))
    c0 = left_margin
    c1 = width - 1 - right_margin
    if c1 - c0 + 1 < 10:
        return None

    max_markers = min(6, max(3, (c1 - c0) // 2))
    marker_count = random.randint(3, max_markers)
    marker_cols = _sample_marker_columns(c0, c1, marker_count)
    if marker_cols is None:
        return None

    marker_colors = _sample_marker_colors(marker_count)
    grid = np.full((height, width), BG, dtype=int)

    for r in band_rows:
        grid[r, c0 : c1 + 1] = BLUE

    top_row = int(band_rows[0])
    bottom_row = int(band_rows[-1])
    for c, color in zip(marker_cols, marker_colors):
        r0, r1 = _marker_extent(height, top_row, bottom_row)
        grid[r0 : r1 + 1, c] = color

    same_edges = sum(a == b for a, b in zip(marker_colors, marker_colors[1:]))
    diff_edges = (marker_count - 1) - same_edges
    if same_edges == 0 or diff_edges == 0:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for seed1155420539/1/6")
