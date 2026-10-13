from __future__ import annotations

import numpy as np

from arc_utils import get_bg
import random


PIVOT_COLOR = 2
CROSS_COLOR = 5


def _unique_extreme(
    controls: list[tuple[int, int, int]],
    axis: int,
    extreme_fn,
    name: str,
) -> tuple[int, int, int]:
    value = extreme_fn(control[axis] for control in controls)
    matches = [control for control in controls if control[axis] == value]
    assert len(matches) == 1, f"expected a unique {name} control pixel"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw the colored rectangle defined by four control pixels and connect its interior to the red pivot with a grey cross.

    Input structure
    - The background grid contains exactly one red pivot pixel.
    - Four other single-pixel controls have unique topmost, bottommost, leftmost, and rightmost positions.
    - The red pivot lies strictly inside the rectangle bounded by those extreme row and column positions.

    Rule
    - Use the topmost control's row and color for the rectangle top side, the bottommost for the bottom side, the leftmost for the left side, and the rightmost for the right side.
    - Paint the top and bottom sides across the full left-to-right span, then paint the left and right sides only between them.
    - Paint grey from the pivot to each side along the pivot's row and column, preserving the red pivot itself.

    Key insight
    - The four controls are not directly joined to each other; their row and column extrema define a rectangular frame around the pivot.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)

    pivot_coords = np.argwhere(g == PIVOT_COLOR)
    assert len(pivot_coords) == 1, "expected exactly one red pivot pixel"
    pivot_r, pivot_c = map(int, pivot_coords[0])

    controls = [
        (int(r), int(c), int(g[r, c]))
        for r, c in np.argwhere((g != bg) & (g != PIVOT_COLOR))
    ]
    assert len(controls) == 4, "expected four non-pivot control pixels"

    top = _unique_extreme(controls, 0, min, "top")
    bottom = _unique_extreme(controls, 0, max, "bottom")
    left = _unique_extreme(controls, 1, min, "left")
    right = _unique_extreme(controls, 1, max, "right")
    role_positions = {(r, c) for r, c, _ in (top, bottom, left, right)}
    assert len(role_positions) == 4, "expected four distinct side controls"

    top_r = top[0]
    bottom_r = bottom[0]
    left_c = left[1]
    right_c = right[1]
    assert top_r < pivot_r < bottom_r, "pivot must be vertically inside the control rectangle"
    assert left_c < pivot_c < right_c, "pivot must be horizontally inside the control rectangle"

    for r, c, _ in controls:
        assert top_r <= r <= bottom_r, "control row must lie within the rectangle span"
        assert left_c <= c <= right_c, "control column must lie within the rectangle span"

    out = g.copy()
    out[top_r, left_c : right_c + 1] = top[2]
    out[bottom_r, left_c : right_c + 1] = bottom[2]
    out[top_r + 1 : bottom_r, left_c] = left[2]
    out[top_r + 1 : bottom_r, right_c] = right[2]
    out[top_r + 1 : bottom_r, pivot_c] = CROSS_COLOR
    out[pivot_r, left_c + 1 : right_c] = CROSS_COLOR
    out[pivot_r, pivot_c] = PIVOT_COLOR
    return out

# --------------- generator (o5x) ---------------

# Canvas and rectangle bounds.
_GEN_GRID_ROWS, _GEN_GRID_COLS = (14, 16), (14, 16)
_GEN_RECT_ROW_STARTS, _GEN_RECT_COL_STARTS = (1, 3), (1, 3)
_GEN_RECT_HEIGHTS, _GEN_RECT_WIDTHS = (11, 13), (8, 11)
_GEN_RECT_MARGIN = 1

# Pivot/control placement and colors.
_GEN_CONTROL_COLORS = (1, 3, 4, 6, 7, 8, 9)
_GEN_PIVOT_COLOR, _GEN_CROSS_COLOR = 2, 5
_GEN_PIVOT_CLEARANCE = 2

# Retry budget.
_GEN_RETRY_LIMIT = 300


def _gen_valid_pairs(size: int, start_range: tuple[int, int], span_range: tuple[int, int]):
    """Return inclusive side pairs matching the configured span and margin."""
    start_min, start_max = start_range
    span_min, span_max = span_range
    pairs = []
    for start in range(start_min, start_max + 1):
        last_max = min(start + span_max - 1, size - 1 - _GEN_RECT_MARGIN)
        for last in range(start + span_min - 1, last_max + 1):
            pairs.append((start, last))
    return pairs


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*_GEN_GRID_ROWS)
    cols = random.randint(*_GEN_GRID_COLS)

    row_pairs = _gen_valid_pairs(rows, _GEN_RECT_ROW_STARTS, _GEN_RECT_HEIGHTS)
    col_pairs = _gen_valid_pairs(cols, _GEN_RECT_COL_STARTS, _GEN_RECT_WIDTHS)
    if not row_pairs or not col_pairs:
        return None

    top_r, bottom_r = random.choice(row_pairs)
    left_c, right_c = random.choice(col_pairs)

    pivot_r = random.randint(top_r + _GEN_PIVOT_CLEARANCE, bottom_r - _GEN_PIVOT_CLEARANCE)
    pivot_c = random.randint(left_c + _GEN_PIVOT_CLEARANCE, right_c - _GEN_PIVOT_CLEARANCE)

    top_cols = [c for c in range(left_c + 1, right_c) if c != pivot_c]
    side_rows = [r for r in range(top_r + 1, bottom_r) if r != pivot_r]
    if not top_cols or not side_rows:
        return None

    top_c = random.choice(top_cols)
    bottom_c = random.choice(top_cols)
    left_r = random.choice(side_rows)
    right_r = random.choice(side_rows)
    top_color, bottom_color, left_color, right_color = random.sample(_GEN_CONTROL_COLORS, 4)

    inp = np.zeros((rows, cols), dtype=int)
    inp[pivot_r, pivot_c] = _GEN_PIVOT_COLOR
    inp[top_r, top_c] = top_color
    inp[bottom_r, bottom_c] = bottom_color
    inp[left_r, left_c] = left_color
    inp[right_r, right_c] = right_color

    expected = inp.copy()
    expected[top_r, left_c : right_c + 1] = top_color
    expected[bottom_r, left_c : right_c + 1] = bottom_color
    expected[top_r + 1 : bottom_r, left_c] = left_color
    expected[top_r + 1 : bottom_r, right_c] = right_color
    expected[top_r + 1 : bottom_r, pivot_c] = _GEN_CROSS_COLOR
    expected[pivot_r, left_c + 1 : right_c] = _GEN_CROSS_COLOR
    expected[pivot_r, pivot_c] = _GEN_PIVOT_COLOR

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e4075551 input")
