import numpy as np

from arc_utils import Color, find_components, get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the red segment one quarter-turn clockwise around the grey marker, recoloring its original cells green.

    Input structure:
    - The grid has a dominant background, exactly one grey marker, and one red
      4-connected segment.
    - The red segment is a straight horizontal or vertical bar attached to the
      grey marker at exactly one end.
    - The clockwise-rotated red segment fits on the same canvas.

    Rule:
    - Locate the grey marker and the adjacent red cell, which gives the bar's
      direction away from the marker.
    - Recolor every input red cell green.
    - Draw a red segment of the same length from the marker in the direction
      obtained by rotating the original direction clockwise.

    Key insight:
    - The grey marker is the pivot, and the side occupied by the red bar tells
      which perpendicular side receives the new red bar.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    assert set(map(int, np.unique(g))) == {bg, Color.RED, Color.GREY}

    marker_coords = np.argwhere(g == Color.GREY)
    assert marker_coords.shape == (1, 2)
    pivot_r, pivot_c = map(int, marker_coords[0])

    red_components = find_components(
        g, bg=bg, connectivity=4, monochromatic=True, colors=[Color.RED]
    )
    assert len(red_components) == 1
    red_coords = red_components[0]["coords"]
    red_set = {tuple(map(int, coord)) for coord in red_coords}
    length = len(red_set)

    adjacent = [
        (r, c)
        for r, c in red_set
        if abs(r - pivot_r) + abs(c - pivot_c) == 1
    ]
    assert len(adjacent) == 1
    near_r, near_c = adjacent[0]
    dr = near_r - pivot_r
    dc = near_c - pivot_c
    assert abs(dr) + abs(dc) == 1

    expected_source = {
        (pivot_r + k * dr, pivot_c + k * dc)
        for k in range(1, length + 1)
    }
    assert red_set == expected_source

    out = g.copy()
    out[g == Color.RED] = Color.GREEN

    rotated_dr, rotated_dc = dc, -dr
    H, W = g.shape
    for k in range(1, length + 1):
        r = pivot_r + k * rotated_dr
        c = pivot_c + k * rotated_dc
        assert 0 <= r < H and 0 <= c < W
        assert int(g[r, c]) == bg
        out[r, c] = Color.RED

    return out

# --------------- generator (o5x) ---------------

# Canvas and object size ranges.
GRID_ROWS = (3, 11)
GRID_COLS = (3, 11)
BAR_LENGTH = (1, 4)
DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

# Retry budget for oracle-checked generation.
GENERATION_RETRIES = 300


def _gen_paint_bar(grid, pivot_r, pivot_c, dr, dc, length, color):
    for k in range(1, length + 1):
        grid[pivot_r + k * dr, pivot_c + k * dc] = color


def _gen_attempt():
    H = random.randint(*GRID_ROWS)
    W = random.randint(*GRID_COLS)
    max_len = min(BAR_LENGTH[1], H - 1, W - 1)
    length = random.randint(BAR_LENGTH[0], max_len)
    dr, dc = random.choice(DIRECTIONS)
    rot_dr, rot_dc = dc, -dr

    min_r = 0
    max_r = H - 1
    min_c = 0
    max_c = W - 1
    for vr, vc in ((dr, dc), (rot_dr, rot_dc)):
        if vr < 0:
            min_r = max(min_r, length)
        elif vr > 0:
            max_r = min(max_r, H - 1 - length)
        if vc < 0:
            min_c = max(min_c, length)
        elif vc > 0:
            max_c = min(max_c, W - 1 - length)

    if min_r > max_r or min_c > max_c:
        return None

    pivot_r = random.randint(min_r, max_r)
    pivot_c = random.randint(min_c, max_c)
    inp = np.zeros((H, W), dtype=int)
    inp[pivot_r, pivot_c] = Color.GREY
    _gen_paint_bar(inp, pivot_r, pivot_c, dr, dc, length, Color.RED)

    expected = inp.copy()
    expected[inp == Color.RED] = Color.GREEN
    _gen_paint_bar(expected, pivot_r, pivot_c, rot_dr, rot_dc, length, Color.RED)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input():
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid c074846d input")
