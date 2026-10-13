from __future__ import annotations

import numpy as np

from arc_utils import Color, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Add a one-step inward miniature of the three colored square-corner markers plus a grey center pixel.

    Input structure:
    - The background is the dominant color.
    - Exactly three non-background pixels appear at three corners of one axis-aligned square.
    - The square has an integer center and enough span that the inward miniature is separate from the original markers.

    Rule:
    - Keep the input unchanged.
    - Paint the square center grey.
    - For each colored corner marker, paint the pixel one diagonal step from the center toward that marker with the marker's color.

    Key insight:
    The three markers define both the missing fourth corner and the square center; the output draws a scaled-down copy of the observed corner pattern around that center.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    coords = np.argwhere(g != bg)
    assert coords.shape[0] == 3, "expected exactly three foreground marker pixels"

    rows = sorted(set(map(int, coords[:, 0])))
    cols = sorted(set(map(int, coords[:, 1])))
    assert len(rows) == 2 and len(cols) == 2, "markers must occupy two rows and two columns"

    r0, r1 = rows
    c0, c1 = cols
    side = r1 - r0
    assert side == c1 - c0, "marker bbox must be square"
    assert side >= 4, "square must have room for an inward miniature"
    assert (r0 + r1) % 2 == 0 and (c0 + c1) % 2 == 0, "square center must be a grid cell"

    marker_positions = {(int(r), int(c)) for r, c in coords}
    valid_corners = {(r, c) for r in rows for c in cols}
    assert marker_positions < valid_corners, "markers must be exactly three of the square corners"

    center_r = (r0 + r1) // 2
    center_c = (c0 + c1) // 2
    out = g.copy()
    assert out[center_r, center_c] == bg, "square center must start as background"
    out[center_r, center_c] = Color.GREY

    for r, c in marker_positions:
        dr = -1 if r < center_r else 1
        dc = -1 if c < center_c else 1
        target_r = center_r + dr
        target_c = center_c + dc
        assert out[target_r, target_c] == bg, "inward copy target must start as background"
        out[target_r, target_c] = int(g[r, c])

    return out

# --------------- generator (o5x) ---------------

# Canvas, square, and retry configuration; range upper bounds are randint-exclusive.
_GEN_GRID_RANGES = ((10, 15), (10, 15))
_GEN_SQUARE_SIDES = (6, 8, 10)
_GEN_EDGE_MARGIN = 1
_GEN_MAX_ATTEMPTS = 300
# Marker palette and oracle constants.
_GEN_MARKER_COLORS = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
_GEN_CENTER_COLOR = 5

def _sample_layout():
    """Return one valid grid shape, center, and four square corners."""
    h = int(np.random.randint(*_GEN_GRID_RANGES[0]))
    w = int(np.random.randint(*_GEN_GRID_RANGES[1]))
    min_room = 2 * _GEN_EDGE_MARGIN + 1
    valid_sides = [s for s in _GEN_SQUARE_SIDES if s <= h - min_room and s <= w - min_room]
    if not valid_sides:
        return None
    side = int(np.random.choice(valid_sides))
    r0 = int(np.random.randint(_GEN_EDGE_MARGIN, h - side - _GEN_EDGE_MARGIN))
    c0 = int(np.random.randint(_GEN_EDGE_MARGIN, w - side - _GEN_EDGE_MARGIN))
    corners = ((r0, c0), (r0, c0 + side), (r0 + side, c0), (r0 + side, c0 + side))
    return h, w, (r0 + side // 2, c0 + side // 2), corners

def _render_case(layout):
    """Build the sampled input and its expected output oracle."""
    h, w, (center_r, center_c), corners = layout
    missing_i = int(np.random.randint(0, len(corners)))
    marker_positions = [pos for i, pos in enumerate(corners) if i != missing_i]
    marker_colors = np.random.choice(_GEN_MARKER_COLORS, size=len(marker_positions), replace=False)

    grid = np.zeros((h, w), dtype=int)
    expected = grid.copy()
    for (r, c), color in zip(marker_positions, marker_colors):
        color = int(color)
        grid[r, c] = color
        expected[r, c] = color

        dr = -1 if r < center_r else 1
        dc = -1 if c < center_c else 1
        expected[center_r + dr, center_c + dc] = color

    expected[center_r, center_c] = _GEN_CENTER_COLOR
    return grid, expected

def _gen_attempt() -> np.ndarray | None:
    layout = _sample_layout()
    if layout is None:
        return None
    grid, expected = _render_case(layout)
    try:
        solved = main(grid)
    except AssertionError:
        return None
    return grid if np.array_equal(solved, expected) else None

def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 11e1fe23 input")
