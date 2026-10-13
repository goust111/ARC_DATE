from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present, get_bg, in_bounds
import random


NEIGHBOR_COLORS = (
    (-1, 0, Color.RED),
    (0, -1, Color.ORANGE),
    (0, 1, Color.PINK),
    (1, 0, Color.TEAL),
)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Add a clipped direction-colored four-neighbor cross around each blue marker pixel.

    Input structure
    - The grid has a black background and one foreground color: blue marker pixels.
    - Marker pixels are isolated enough that their four-neighbor stamps do not overlap or touch another marker.

    Rule
    - Keep every original blue marker pixel unchanged.
    - For each marker, recolor its in-bounds background neighbor above to red, left to orange, right to pink, and below to teal.
    - Clip the stamp at the grid boundary.

    Key insight
    - The marker position is an anchor; the added color is determined solely by compass direction from that anchor.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == Color.BLACK, "expected black background"

    marker_colors = colors_present(g, exclude={bg})
    assert marker_colors == [Color.BLUE], "expected only blue marker pixels"
    markers = np.argwhere(g == Color.BLUE)
    assert len(markers) > 0, "expected at least one marker"

    out = g.copy()
    for r_raw, c_raw in markers:
        r = int(r_raw)
        c = int(c_raw)
        for dr, dc, color in NEIGHBOR_COLORS:
            rr = r + dr
            cc = c + dc
            if not in_bounds(g.shape, rr, cc):
                continue
            assert int(g[rr, cc]) == bg, "marker stamps must target background cells"
            assert int(out[rr, cc]) == bg, "marker stamps must not overlap"
            out[rr, cc] = color

    return out

# --------------- generator (o5x) ---------------

# Canvas and colors.
GEN_GRID_SHAPE = (10, 10)
GEN_BG_COLOR = Color.BLACK
GEN_MARKER_COLOR = Color.BLUE
GEN_NEIGHBORS = (
    (-1, 0, Color.RED), (0, -1, Color.ORANGE),
    (0, 1, Color.PINK), (1, 0, Color.TEAL),
)

# Marker counts and search policy.
GEN_MARKER_COUNT, GEN_EDGE_COUNT = (5, 7), (2, 4)
GEN_RETRIES = 300

def _gen_marker_footprint_d364b489(r, c):
    cells = [(r, c)]
    for dr, dc, _ in GEN_NEIGHBORS:
        rr = r + dr
        cc = c + dc
        if in_bounds(GEN_GRID_SHAPE, rr, cc):
            cells.append((rr, cc))
    return frozenset(cells)

def _gen_expected_output_d364b489(grid):
    out = grid.copy()
    for r_raw, c_raw in np.argwhere(grid == GEN_MARKER_COLOR):
        r = int(r_raw)
        c = int(c_raw)
        for dr, dc, color in GEN_NEIGHBORS:
            rr = r + dr
            cc = c + dc
            if in_bounds(grid.shape, rr, cc):
                out[rr, cc] = color
    return out

def _gen_sample_roles_d364b489():
    marker_count = random.randint(*GEN_MARKER_COUNT)
    edge_count = random.randint(GEN_EDGE_COUNT[0], min(GEN_EDGE_COUNT[1], marker_count))
    roles = [True] * edge_count + [False] * (marker_count - edge_count)
    random.shuffle(roles)
    return roles

def _gen_positions_by_role_d364b489():
    h, w = GEN_GRID_SHAPE
    positions_by_role = {True: [], False: []}
    for r in range(h):
        for c in range(w):
            is_edge = r in (0, h - 1) or c in (0, w - 1)
            positions_by_role[is_edge].append((r, c, _gen_marker_footprint_d364b489(r, c)))
    return positions_by_role

GEN_POSITIONS_BY_ROLE = _gen_positions_by_role_d364b489()

def _gen_place_markers_d364b489(roles):
    def place_from(i, occupied, placed):
        if i == len(roles):
            return placed

        candidates = [item for item in GEN_POSITIONS_BY_ROLE[roles[i]] if occupied.isdisjoint(item[2])]
        random.shuffle(candidates)
        for r, c, footprint in candidates:
            result = place_from(i + 1, occupied | footprint, placed + [(r, c)])
            if result is not None:
                return result
        return None

    return place_from(0, frozenset(), [])

def _gen_attempt():
    roles = _gen_sample_roles_d364b489()
    markers = _gen_place_markers_d364b489(roles)
    if markers is None:
        return None

    grid = np.full(GEN_GRID_SHAPE, GEN_BG_COLOR, dtype=int)
    for r, c in markers:
        grid[r, c] = GEN_MARKER_COLOR
    return grid

def generate_input():
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is None:
            continue
        try:
            expected = _gen_expected_output_d364b489(grid)
            if np.array_equal(main(grid), expected):
                return grid
        except Exception:
            continue
    raise RuntimeError("failed to generate valid d364b489 input")
