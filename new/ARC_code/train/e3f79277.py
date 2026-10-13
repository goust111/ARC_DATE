import numpy as np

from arc_utils import colors_present, get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace a corner L with a same-color right-triangle outline in the corresponding corner of an enlarged canvas.

    Input structure:
    - The input is a square grid with one background color and one foreground color.
    - The foreground is a one-pixel-thick L inside a square bbox.
    - The L touches exactly one horizontal grid edge and one vertical grid edge, identifying a corner.

    Rule:
    - Fill a square output canvas of side 3n-2, where n is the input side length, with the background color.
    - Double the L leg length to get the triangle side length.
    - Draw the two perpendicular triangle legs on the same output corner edges and connect their free endpoints with the diagonal.

    Key insight:
    - The input L encodes the corner, color, and side length of the triangle outline; the missing side is the diagonal between the L's two free endpoints.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height == width, "expected a square input grid"

    bg = get_bg(g)
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 1, "expected exactly one foreground color"
    fg = fg_colors[0]

    mask = g == fg
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a foreground L"
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    leg_len = r1 - r0 + 1
    assert leg_len == c1 - c0 + 1, "expected the L bbox to be square"
    assert leg_len > 1, "expected a nontrivial L"

    touches_top = r0 == 0
    touches_bottom = r1 == height - 1
    touches_left = c0 == 0
    touches_right = c1 == width - 1
    assert sum((touches_top, touches_bottom)) == 1, "expected the L to touch one horizontal grid edge"
    assert sum((touches_left, touches_right)) == 1, "expected the L to touch one vertical grid edge"

    expected = np.zeros_like(mask, dtype=bool)
    edge_row = r0 if touches_top else r1
    edge_col = c0 if touches_left else c1
    expected[edge_row, c0 : c1 + 1] = True
    expected[r0 : r1 + 1, edge_col] = True
    assert np.array_equal(mask, expected), "expected foreground to be exactly a corner L"

    out_size = 3 * height - 2
    side = 2 * leg_len
    out = np.full((out_size, out_size), bg, dtype=int)
    top = touches_top
    left = touches_left
    rr0 = 0 if top else out_size - side
    cc0 = 0 if left else out_size - side
    out_edge_row = rr0 if top else rr0 + side - 1
    out_edge_col = cc0 if left else cc0 + side - 1

    for i in range(side):
        out[out_edge_row, cc0 + i] = fg
        out[rr0 + i, out_edge_col] = fg
        diag_col = side - 1 - i if top == left else i
        out[rr0 + i, cc0 + diag_col] = fg

    return out

# --------------- generator (o5x) ---------------
from arc_utils import bbox

# Sampling bounds.
GRID_SIZE_MIN = 4
GRID_SIZE_MAX = 8
LEG_LEN_MIN = 2
COLOR_VALUES = range(10)
COLOR_PAIR_SIZE = 2
CORNER_ANCHORS = ((True, True), (True, False), (False, True), (False, False))

# Output geometry and validation.
OUTPUT_SCALE = 3
OUTPUT_OFFSET = 2
TRIANGLE_SIDE_SCALE = 2
MAX_ATTEMPTS = 300


def _gen_expected_output(grid: np.ndarray) -> np.ndarray:
    bg = get_bg(grid)
    fg_values = colors_present(grid, exclude={bg})
    if len(fg_values) != 1:
        raise ValueError("expected exactly one foreground color")
    fg = fg_values[0]

    box = bbox(grid == fg)
    if box is None:
        raise ValueError("expected a foreground L")
    r0, c0, r1, c1 = box
    leg_len = r1 - r0 + 1
    side = TRIANGLE_SIDE_SCALE * leg_len
    out_size = OUTPUT_SCALE * grid.shape[0] - OUTPUT_OFFSET
    out = np.full((out_size, out_size), bg, dtype=int)

    top = r0 == 0
    left = c0 == 0
    rr0 = 0 if top else out_size - side
    cc0 = 0 if left else out_size - side
    edge_row = rr0 if top else rr0 + side - 1
    edge_col = cc0 if left else cc0 + side - 1

    idx = np.arange(side)
    out[edge_row, cc0 + idx] = fg
    out[rr0 + idx, edge_col] = fg
    diag_cols = side - 1 - idx if top == left else idx
    out[rr0 + idx, cc0 + diag_cols] = fg
    return out


def _draw_corner_l(n: int, bg: int, fg: int, leg_len: int, corner: int) -> np.ndarray:
    grid = np.full((n, n), bg, dtype=int)
    top, left = CORNER_ANCHORS[corner]
    row = 0 if top else n - 1
    col = 0 if left else n - 1
    row_cols = slice(0, leg_len) if left else slice(n - leg_len, n)
    col_rows = slice(0, leg_len) if top else slice(n - leg_len, n)
    grid[row, row_cols] = fg
    grid[col_rows, col] = fg
    return grid


def _gen_attempt() -> np.ndarray:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    bg, fg = random.sample(COLOR_VALUES, COLOR_PAIR_SIZE)
    leg_len = random.randint(LEG_LEN_MIN, n - 1)
    corner = random.randrange(len(CORNER_ANCHORS))
    return _draw_corner_l(n, bg, fg, leg_len, corner)


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        candidate = _gen_attempt()
        expected = _gen_expected_output(candidate)
        try:
            actual = main(candidate)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return candidate
    raise RuntimeError(f"generate_input: failed after {MAX_ATTEMPTS} attempts")
