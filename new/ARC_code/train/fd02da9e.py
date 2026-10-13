from __future__ import annotations

import numpy as np

from arc_utils import get_bg
import random


_CORNER_STENCILS = {
    "top_left": ((1, 1), (1, 2), (2, 1), (2, 2)),
    "top_right": ((1, -3), (1, -2), (2, -3), (2, -2)),
    "bottom_left": ((-4, 2), (-3, 2), (-2, 3)),
    "bottom_right": ((-4, -3), (-3, -3), (-2, -4)),
}


def _coord(shape: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    height, width = shape
    rr = r if r >= 0 else height + r
    cc = c if c >= 0 else width + c
    assert 0 <= rr < height and 0 <= cc < width, "stencil cell must be in bounds"
    return int(rr), int(cc)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each colored corner marker with its corner-specific inward stencil in the same color.

    Input structure
    - The grid is 8x8 with a single dominant background color.
    - Foreground pixels are marker pixels, each located at one of the four corners.
    - A marker's color is carried to the stencil selected by its corner position.

    Rule
    - Erase each corner marker back to the background.
    - Top corner markers become an inward 2x2 block one cell away from the top and side edges.
    - Bottom corner markers become the mirrored three-pixel inward bent stencil shown by the bottom-corner examples.

    Key insight
    - The marker position selects the fixed stencil; the marker color only supplies the paint color, so multiple corners apply independently.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape == (8, 8), "expected the fixed 8x8 task grid"

    bg = get_bg(g)
    assert np.count_nonzero(g == bg) > g.size // 2, "expected a dominant background color"

    height, width = map(int, g.shape)
    corners = {
        "top_left": (0, 0),
        "top_right": (0, width - 1),
        "bottom_left": (height - 1, 0),
        "bottom_right": (height - 1, width - 1),
    }
    corner_cells = set(corners.values())

    foreground = [tuple(map(int, rc)) for rc in np.argwhere(g != bg)]
    assert foreground, "expected at least one corner marker"
    assert all(rc in corner_cells for rc in foreground), "all foreground pixels must be corner markers"

    out = g.copy()
    planned: dict[tuple[int, int], int] = {}
    for name, (r, c) in corners.items():
        color = int(g[r, c])
        if color == bg:
            continue

        out[r, c] = bg
        for sr, sc in _CORNER_STENCILS[name]:
            rr, cc = _coord(g.shape, sr, sc)
            assert int(g[rr, cc]) == bg, "stencil targets must start as background"
            previous = planned.get((rr, cc))
            assert previous is None, "corner stencils must not overlap"
            planned[(rr, cc)] = color

    for (r, c), color in planned.items():
        out[r, c] = color

    return out

# --------------- generator (o5x) ---------------
# Canvas and palette.
_GEN_SHAPE = (8, 8)
_GEN_COLOR_COUNT = 10
# Marker sampling and retry limits.
_GEN_MARKER_COUNT_MIN = 1
_GEN_MARKER_COUNT_MAX = 4
_GEN_RETRY_LIMIT = 300
_GEN_CORNER_SPECS = (
    ("top_left", (0, 0)),
    ("top_right", (0, _GEN_SHAPE[1] - 1)),
    ("bottom_left", (_GEN_SHAPE[0] - 1, 0)),
    ("bottom_right", (_GEN_SHAPE[0] - 1, _GEN_SHAPE[1] - 1)),
)
_GEN_CORNERS = tuple(corner for _, corner in _GEN_CORNER_SPECS)
_GEN_STENCIL_SPECS = tuple(
    (corner, tuple(_coord(_GEN_SHAPE, sr, sc) for sr, sc in _CORNER_STENCILS[name]))
    for name, corner in _GEN_CORNER_SPECS
)


def _gen_expected_output(inp: np.ndarray, bg: int) -> np.ndarray:
    out = inp.copy()
    for (r, c), cells in _GEN_STENCIL_SPECS:
        color = int(inp[r, c])
        if color == bg:
            continue
        out[r, c] = bg
        for rr, cc in cells:
            out[rr, cc] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    bg = random.randrange(_GEN_COLOR_COUNT)
    grid = np.full(_GEN_SHAPE, bg, dtype=int)

    marker_count = random.randint(_GEN_MARKER_COUNT_MIN, _GEN_MARKER_COUNT_MAX)
    chosen = random.sample(_GEN_CORNERS, marker_count)
    colors = random.sample([color for color in range(_GEN_COLOR_COUNT) if color != bg], marker_count)
    for (r, c), color in zip(chosen, colors):
        grid[r, c] = color

    expected = _gen_expected_output(grid, bg)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid fd02da9e input")
