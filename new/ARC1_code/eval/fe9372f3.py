from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox, colors_present, get_bg, in_bounds
import random


def _red_plus_center(g: np.ndarray, bg: int) -> tuple[int, int]:
    colors = colors_present(g, exclude={bg})
    assert colors == [Color.RED], "expected only one red foreground object"

    red = g == Color.RED
    coords = np.argwhere(red)
    assert coords.shape == (5, 2), "expected a five-pixel red plus"

    b = bbox(red)
    assert b is not None, "expected red foreground"
    r0, c0, r1, c1 = b
    assert (r1 - r0, c1 - c0) == (2, 2), "red plus must have a 3x3 bounding box"

    cr = r0 + 1
    cc = c0 + 1
    expected = {
        (cr, cc),
        (cr - 1, cc),
        (cr + 1, cc),
        (cr, cc - 1),
        (cr, cc + 1),
    }
    assert set(map(tuple, coords)) == expected, "red foreground must be a plus shape"
    return cr, cc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the red plus into an eight-direction star with blue diagonal rays and repeating teal-teal-yellow cardinal rays.

    Input structure
    - The canvas background is black.
    - The only foreground object is one 4-connected five-pixel red plus.
    - The plus supplies a center, four diagonal gaps, and four cardinal arms.

    Rule
    - Preserve the red plus.
    - From each diagonal neighbor of the plus center, paint a blue ray to the grid edge.
    - From each cardinal arm, continue outward to the grid edge with the repeating color pattern teal, teal, yellow.

    Key insight
    - The red plus is a compact ray emitter: its diagonal gaps start the uniform blue diagonals, while each arm starts the same three-cell cardinal stripe pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)
    assert bg == Color.BLACK, "expected black background"

    cr, cc = _red_plus_center(g, bg)
    out = g.copy()

    for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        r = cr + dr
        c = cc + dc
        while in_bounds(g.shape, r, c):
            assert int(out[r, c]) == bg, "diagonal ray must cross only background"
            out[r, c] = Color.BLUE
            r += dr
            c += dc

    cardinal_pattern = (Color.TEAL, Color.TEAL, Color.YELLOW)
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        r = cr + 2 * dr
        c = cc + 2 * dc
        k = 0
        while in_bounds(g.shape, r, c):
            assert int(out[r, c]) == bg, "cardinal ray must cross only background"
            out[r, c] = cardinal_pattern[k % len(cardinal_pattern)]
            k += 1
            r += dr
            c += dc

    return out

# --------------- generator (o5x) ---------------

# Canvas size bounds.
GRID_HEIGHT_RANGE = (7, 30)
GRID_WIDTH_RANGE = (10, 17)

# Red plus geometry and placement.
PLUS_OFFSETS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
PLUS_CENTER_EDGE_MARGIN = 3

# Output oracle ray definitions.
DIAGONAL_DIRECTIONS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
CARDINAL_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
CARDINAL_PATTERN = (Color.TEAL, Color.TEAL, Color.YELLOW)

# Validation retry limit.
GEN_RETRY_LIMIT = 300


def _gen_expected_output(inp: np.ndarray) -> np.ndarray:
    out = inp.copy()
    h, w = out.shape
    coords = np.argwhere(out == Color.RED)
    cr = int(coords[:, 0].mean())
    cc = int(coords[:, 1].mean())

    for dr, dc in DIAGONAL_DIRECTIONS:
        r = cr + dr
        c = cc + dc
        while 0 <= r < h and 0 <= c < w:
            out[r, c] = Color.BLUE
            r += dr
            c += dc

    for dr, dc in CARDINAL_DIRECTIONS:
        r = cr + 2 * dr
        c = cc + 2 * dc
        k = 0
        while 0 <= r < h and 0 <= c < w:
            out[r, c] = CARDINAL_PATTERN[k % len(CARDINAL_PATTERN)]
            k += 1
            r += dr
            c += dc

    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)

    cr = random.randint(PLUS_CENTER_EDGE_MARGIN, h - PLUS_CENTER_EDGE_MARGIN - 1)
    cc = random.randint(PLUS_CENTER_EDGE_MARGIN, w - PLUS_CENTER_EDGE_MARGIN - 1)

    g = np.full((h, w), Color.BLACK, dtype=int)
    for dr, dc in PLUS_OFFSETS:
        g[cr + dr, cc + dc] = Color.RED
    return g


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        inp = _gen_attempt()
        if inp is None:
            continue
        try:
            out = main(inp)
        except AssertionError:
            continue
        if np.array_equal(out, _gen_expected_output(inp)):
            return inp
    raise RuntimeError("failed to generate a valid fe9372f3 input")
