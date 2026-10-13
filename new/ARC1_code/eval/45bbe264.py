from __future__ import annotations

import numpy as np

from arc_utils import Color, get_bg
import itertools
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each colored marker through its full row and column, marking crossings between different markers red.

    Input structure:
    - The grid has a dominant background color and two or more isolated non-background marker pixels.
    - Marker pixels have distinct rows, distinct columns, and distinct non-red colors.

    Rule:
    - For each marker, paint its entire row and entire column with the marker color.
    - Where the row or column from one marker crosses the row or column from another marker, paint red.
    - Preserve the original marker pixels in their own colors.

    Key insight:
    - The sparse marker pixels are anchors for full-grid horizontal and vertical guide lines; only cross-marker intersections get the special red color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    bg_count = int(np.count_nonzero(g == bg))
    assert bg_count > g.size // 2, "expected a dominant background"

    coords = np.argwhere(g != bg)
    markers = [(int(r), int(c), int(g[r, c])) for r, c in coords]
    assert len(markers) >= 2, "expected at least two markers"

    rows = [r for r, _c, _color in markers]
    cols = [c for _r, c, _color in markers]
    colors = [color for _r, _c, color in markers]
    assert len(set(rows)) == len(rows), "marker rows must be unique"
    assert len(set(cols)) == len(cols), "marker columns must be unique"
    assert len(set(colors)) == len(colors), "marker colors must be unique"
    assert Color.RED not in colors, "red is reserved for crossings"
    marker_positions = {(r, c) for r, c, _color in markers}

    out = np.full_like(g, bg)
    for r, c, color in markers:
        out[r, :] = color
        out[:, c] = color
        out[r, c] = color

    for r1, _c1, _color1 in markers:
        for _r2, c2, _color2 in markers:
            if (r1, c2) not in marker_positions:
                out[r1, c2] = Color.RED

    for r, c, color in markers:
        out[r, c] = color

    return out

# --------------- generator (o5x) ---------------

# Canvas bounds and marker geometry.
_GEN_GRID_SIZE_RANGE = (13, 16)
_GEN_EDGE_PADDING = 1
_GEN_MIN_MARKER_GAP = 2
_GEN_MARKER_COUNT_RANGE = (2, 3)
_GEN_MARKER_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)
_GEN_CROSSING_COLOR = 2
_GEN_MAX_ATTEMPTS = 300


def _gen_spaced_positions(size: int, count: int) -> tuple[int, ...]:
    """Sample sorted interior coordinates with the configured pairwise gap."""
    positions = range(_GEN_EDGE_PADDING, size - _GEN_EDGE_PADDING)
    choices = [
        combo
        for combo in itertools.combinations(positions, count)
        if all(b - a >= _GEN_MIN_MARKER_GAP for a, b in zip(combo, combo[1:]))
    ]
    return random.choice(choices)


def _gen_markers(size: int, count: int) -> list[tuple[int, int, int]]:
    colors = random.sample(_GEN_MARKER_COLORS, count)
    rows = list(_gen_spaced_positions(size, count))
    cols = list(_gen_spaced_positions(size, count))
    random.shuffle(rows)
    random.shuffle(cols)
    return list(zip(rows, cols, colors))


def _gen_expected_output(
    size: int, markers: list[tuple[int, int, int]]
) -> np.ndarray:
    out = np.zeros((size, size), dtype=int)
    marker_positions = {(r, c) for r, c, _color in markers}

    for r, c, color in markers:
        out[r, :] = color
        out[:, c] = color
        out[r, c] = color

    for r1, _c1, _color1 in markers:
        for _r2, c2, _color2 in markers:
            if (r1, c2) not in marker_positions:
                out[r1, c2] = _GEN_CROSSING_COLOR

    for r, c, color in markers:
        out[r, c] = color

    return out


def _gen_attempt() -> np.ndarray | None:
    size = random.randint(*_GEN_GRID_SIZE_RANGE)
    marker_count = random.randint(*_GEN_MARKER_COUNT_RANGE)
    markers = _gen_markers(size, marker_count)
    grid = np.zeros((size, size), dtype=int)
    for r, c, color in markers:
        grid[r, c] = color

    expected = _gen_expected_output(size, markers)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 45bbe264 input")
