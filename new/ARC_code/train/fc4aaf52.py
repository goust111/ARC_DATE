from __future__ import annotations

import numpy as np

from arc_utils import colors_present, find_components, get_bg
import random


def _swapped_values(values: np.ndarray, color_a: int, color_b: int) -> np.ndarray:
    out = values.copy()
    out[values == int(color_a)] = int(color_b)
    out[values == int(color_b)] = int(color_a)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split the two-color object horizontally, swap its colors, and slide the upper half to the right.

    Input structure:
    - The grid has one background color and exactly two foreground colors.
    - All foreground pixels form one 4-connected object with an even-height bbox.
    - The object has foreground pixels on both rows adjacent to its horizontal midpoint.

    Rule:
    - Split the object into upper and lower halves at the midpoint of its bbox.
    - Swap the two foreground colors in both halves.
    - Keep the lower half in place and shift the upper half right until its lower-left
      boundary pixel is diagonally adjacent to the lower half's upper-right boundary pixel.

    Key insight:
    - The object is not copied or reflected; it is broken at the horizontal waist,
      recolored by exchanging the two visible object colors, then rejoined diagonally.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    colors = colors_present(g, exclude={bg})
    assert len(colors) == 2, "expected exactly two foreground colors"
    color_a, color_b = colors

    objects = find_components(g, bg=bg, connectivity=4, monochromatic=False)
    assert len(objects) == 1, "expected one connected foreground object"
    r0, _c0, r1, _c1 = objects[0]["bbox"]
    height = int(r1 - r0 + 1)
    assert height % 2 == 0, "object bbox height must split evenly"

    split_row = int(r0 + height // 2)
    row_ids = np.arange(g.shape[0])[:, None]
    fg = g != int(bg)
    upper = fg & (row_ids < split_row)
    lower = fg & (row_ids >= split_row)
    assert np.any(upper), "expected a nonempty upper half"
    assert np.any(lower), "expected a nonempty lower half"
    assert int(upper.sum()) == int(lower.sum()), "expected equal upper and lower areas"

    upper_boundary_cols = np.flatnonzero(upper[split_row - 1])
    lower_boundary_cols = np.flatnonzero(lower[split_row])
    assert upper_boundary_cols.size > 0, "upper half must touch the split boundary"
    assert lower_boundary_cols.size > 0, "lower half must touch the split boundary"

    shift = int(lower_boundary_cols.max() + 1 - upper_boundary_cols.min())
    assert shift > 0, "upper half must shift right"

    upper_rows, upper_cols = np.where(upper)
    lower_rows, lower_cols = np.where(lower)
    shifted_upper_cols = upper_cols + shift
    assert int(shifted_upper_cols.max()) < g.shape[1], "shifted upper half must fit in the grid"

    out = np.full_like(g, int(bg))
    out[lower_rows, lower_cols] = _swapped_values(g[lower_rows, lower_cols], color_a, color_b)
    out[upper_rows, shifted_upper_cols] = _swapped_values(g[upper_rows, upper_cols], color_a, color_b)
    return out

# --------------- generator (o5x) ---------------

from arc_utils import paste as _gen_paste
# Canvas and palette.
GEN_BG, GEN_GRID_SIZE = 8, 16
GEN_COLOR_POOL = [c for c in range(10) if c != GEN_BG]

# Symmetric object shape.
GEN_HALF_HEIGHTS, GEN_MAX_WIDTHS = [3, 4], [5, 7]
GEN_MIN_ROW_WIDTH, GEN_WIDTH_STEP = 1, 2
GEN_ROW_SWAP_PROB, GEN_ROW_SWAP_HEIGHT = 0.25, 4
GEN_ROW_SWAP_INDICES = (1, 2)

# Placement and validation.
GEN_MARGIN, GEN_RETRIES = 1, 300


def _gen_fc4aaf52_swap(arr: np.ndarray, color_a: int, color_b: int) -> np.ndarray:
    out = arr.copy()
    out[arr == color_a] = color_b
    out[arr == color_b] = color_a
    return out


def _gen_fc4aaf52_upper(half_h: int, border_color: int, fill_color: int) -> np.ndarray:
    max_w = random.choice(GEN_MAX_WIDTHS)
    widths = [max_w]
    for _ in range(half_h - 1):
        prev_w = widths[-1]
        widths.append(random.choice([max(GEN_MIN_ROW_WIDTH, prev_w - GEN_WIDTH_STEP), prev_w]))
    widths = list(reversed(widths))
    if random.random() < GEN_ROW_SWAP_PROB and half_h == GEN_ROW_SWAP_HEIGHT:
        i, j = GEN_ROW_SWAP_INDICES
        widths[i], widths[j] = widths[j], widths[i]

    bottom_lefts = [0]
    bottom_widths = list(reversed(widths))
    for i, width in enumerate(bottom_widths[1:], start=1):
        prev_l = bottom_lefts[-1]
        prev_r = prev_l + bottom_widths[i - 1] - 1
        choices = [left for left in range(max_w - width + 1) if max(left, prev_l) <= min(left + width - 1, prev_r)]
        bottom_lefts.append(random.choice(choices))
    lefts = list(reversed(bottom_lefts))

    upper = np.full((half_h, max_w), GEN_BG, dtype=int)
    for r, (left, width) in enumerate(zip(lefts, widths)):
        right = left + width
        upper[r, left:right] = fill_color
        upper[r, left] = border_color
        upper[r, right - 1] = border_color
    return upper


def _gen_fc4aaf52_placement(obj_h: int, obj_w: int, shift: int) -> tuple[int, int] | None:
    max_row = GEN_GRID_SIZE - GEN_MARGIN - obj_h
    max_col = GEN_GRID_SIZE - GEN_MARGIN - obj_w - shift
    return None if max_col < 0 else (random.randint(GEN_MARGIN, max_row), random.randint(GEN_MARGIN, max_col))


def _gen_fc4aaf52_render(upper, lower, row, col, shift, border_color, fill_color, *, swapped):
    out = np.full((GEN_GRID_SIZE, GEN_GRID_SIZE), GEN_BG, dtype=int)
    upper_col = col
    if swapped:
        upper = _gen_fc4aaf52_swap(upper, border_color, fill_color)
        lower = _gen_fc4aaf52_swap(lower, border_color, fill_color)
        upper_col += shift
    _gen_paste(out, upper, row, upper_col, transparent=GEN_BG)
    _gen_paste(out, lower, row + upper.shape[0], col, transparent=GEN_BG)
    return out


def _gen_attempt() -> np.ndarray | None:
    border_color, fill_color = random.sample(GEN_COLOR_POOL, 2)
    half_h = random.choice(GEN_HALF_HEIGHTS)
    upper_in = _gen_fc4aaf52_upper(half_h, border_color, fill_color)
    lower_in = np.flipud(upper_in)

    obj_h, obj_w = half_h * 2, upper_in.shape[1]
    shift = int(np.count_nonzero(upper_in[-1] != GEN_BG))
    placement = _gen_fc4aaf52_placement(obj_h, obj_w, shift)
    if placement is None:
        return None
    obj_row, obj_col = placement

    expected = _gen_fc4aaf52_render(upper_in, lower_in, obj_row, obj_col, shift, border_color, fill_color, swapped=True)
    inp = _gen_fc4aaf52_render(upper_in, lower_in, obj_row, obj_col, shift, border_color, fill_color, swapped=False)

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid fc4aaf52 input")
