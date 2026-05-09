# Combo variant of 136b0064 (index 2)
# combo_spec: D1/0+D4/0
# Name: 行主序加底部生长折线

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior + trace:
#    The left half still holds a 2-by-N bank of 3x3 glyphs at fixed tile slots, separated from the
#    right half by the full-height divider at column 7. The fused rule changes two stages together:
#    glyphs are decoded in row-major order, and the resulting polyline grows upward from a gray
#    start on the bottom row. For N=2 the tile slots are:
#      row 0: A at (0,0), B at (0,4)
#      row 1: C at (4,0), D at (4,4)
#    Original task order/path frame: A, C, B, D; start at top; before each segment move down.
#    Combo order/path frame:         A, B, C, D; start at bottom; before each segment move up.
#    Example if row-major areas are [5, 4, 7, 6] and shifted start is (6, 3):
#      glyph A area 5 -> move to (5,3), draw right len 3 ending at (5,5)
#      glyph B area 4 -> move to (4,5), then continue up to (3,5)
#      glyph C area 7 -> move to (2,5), draw left len 2 ending at (2,4)
#      glyph D area 6 -> move to (1,4), draw left len 4 ending at (1,1)
#    The differences are therefore not cosmetic: both the instruction permutation and the vertical
#    reference frame change the visible output.
#
# b) Complexity calibration:
#    Only two low-cardinality controls are changed. Shape->color and shape->segment bindings remain
#    fixed, so the puzzle still has repeated evidence for the four instruction types. To prevent the
#    D1 change from becoming invisible, generation rejects inputs whose row-major upward output is
#    identical to the column-major upward output on the same glyph bank.
#
# c) Implicit-assumption audit:
#    - Divider detection by "full-height constant non-background column" still holds.
#    - Gray-start shifting still holds, but the solver now assumes the gray pixel can legally sit on
#      the bottom row; generation enforces exactly that.
#    - Component extraction of 3x3 monochromatic glyphs still holds because layout geometry is
#      unchanged.
#    - The old `sort(key=(c0,r0))` assumption is invalid under D1 and must become `(r0,c0)`.
#    - The old `r_cur += 1` / downward vertical extension assumptions are invalid under D4 and must
#      become upward row motion with top-bound checks.
#    - No conflict is introduced by combining them because ordering is determined entirely from tile
#      positions while growth direction is determined entirely from the start row and row updates.
#
# d) Placement legality:
#    The generator samples the output walk first from `(H-1, start_col)` using deltas
#    {4:(-2,0), 5:(-1,+2), 6:(-1,-3), 7:(-1,-1)}. Each candidate is kept only if the resulting
#    endpoint stays inside rows `0..H-1` and cols `0..6`. Glyphs are then written into fixed non-
#    overlapping 3x3 tile slots, so no component collisions can occur in the input.
#
# e) Visual observability:
#    Humans can see both active controls directly: the gray start is on the bottom row in every
#    sample, and paired glyphs within the same tile row must be read together because generation
#    rejects cases where row-major vs column-major would look the same. No hidden ordering by color
#    value or object id is used.
#
# f) Visual cleanliness:
#    The glyph bank stays isolated on the left of the divider; the right side contains only the gray
#    start. The output is still a single colored trace plus the gray pixel, with no extra markers or
#    overlay clutter introduced by the fusion.
#
# g) Change inferability:
#    Colors remain shape-bound with only four recurring segment types, so there is no new color
#    inference burden. The only new reasoning load is "read row-wise" plus "grow upward from the
#    bottom", both supported by repeated spatial evidence.
#
# New solver assumptions guaranteed by generation:
# - Exactly one gray cell exists and it is on the bottom row of the right half.
# - Every intended glyph is a 3x3 monochromatic 8-connected component.
# - The row-major decoding order changes the produced output relative to the same input read in
#   column-major order.
# - The sampled upward path never exits the 7-column left canvas or the top boundary.


def find_full_height_constant_col(grid: np.ndarray, *, bg: int | None = None) -> tuple[int | None, int | None]:
    """
    Local helper (only used by this solver): find a column where all values are identical and != bg.
    Returns (col_idx, color) or (None, None).
    """
    _h, w = grid.shape
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    for c in range(int(w)):
        col = grid[:, int(c)]
        if (col == col[0]).all() and int(col[0]) != bg:
            return int(c), int(col[0])
    return None, None


def _collect_glyphs(left: np.ndarray, *, bg: int, div_color: int | None) -> list[dict[str, int]]:
    ignore = {int(bg), GRAY}
    if div_color is not None:
        ignore.add(int(div_color))

    glyphs: list[dict[str, int]] = []
    for comp in find_components(left, bg=bg, connectivity=8, monochromatic=True, return_format="dict"):
        color = int(comp["color"])
        if color in ignore:
            continue
        b = comp["bbox"]
        if bbox_h(b) == 3 and bbox_w(b) == 3:
            r0, c0, _r1, _c1 = map(int, b)
            glyphs.append({"color": color, "r0": r0, "c0": c0, "area": int(comp["area"])})
    return glyphs


AREA_TO_LEN = {4: 2, 5: 3, 6: 4, 7: 2}
AREA_TO_HDIR = {5: 1, 6: -1, 7: -1}


def _solve_with_controls(grid: np.ndarray, *, row_major: bool, grow_up: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    h, _w = g.shape
    bg = int(get_bg(g))

    div_c, div_color = find_full_height_constant_col(g, bg=bg)
    assert div_c is not None

    out = np.full((h, int(div_c)), bg, dtype=int)

    gray = np.argwhere(g == GRAY)
    if gray.size == 0:
        return out
    r_cur, c_cur = map(int, gray[0])
    c_cur = int(c_cur - (int(div_c) + 1))
    if not (0 <= c_cur < int(div_c)):
        return out
    out[int(r_cur), int(c_cur)] = GRAY

    glyphs = _collect_glyphs(g[:, : int(div_c)], bg=bg, div_color=div_color)
    if row_major:
        glyphs.sort(key=lambda d: (int(d["r0"]), int(d["c0"])))
    else:
        glyphs.sort(key=lambda d: (int(d["c0"]), int(d["r0"])))

    row_step = -1 if grow_up else 1
    for glyph in glyphs:
        color = int(glyph["color"])
        area = int(glyph["area"])
        seg_len = int(AREA_TO_LEN[area])

        r_cur = int(r_cur + row_step)
        if grow_up:
            assert r_cur >= 0, f"segment exits top: r_cur={r_cur}"
        else:
            assert r_cur < h, f"segment exits bottom: r_cur={r_cur}, H={h}"
        out[r_cur, int(c_cur)] = color

        if area == 4:
            end_r = int(r_cur + row_step * (seg_len - 1))
            if grow_up:
                assert end_r >= 0, f"vertical segment exits top: end_r={end_r}"
            else:
                assert end_r < h, f"vertical segment exits bottom: end_r={end_r}, H={h}"
            for _ in range(seg_len - 1):
                r_cur = int(r_cur + row_step)
                out[r_cur, int(c_cur)] = color
        else:
            dir_h = int(AREA_TO_HDIR[area])
            end_c = int(c_cur) + dir_h * (seg_len - 1)
            assert 0 <= end_c < int(div_c), (
                f"horizontal segment exits canvas: c_cur={c_cur}, dir={dir_h}, "
                f"seg_len={seg_len}, div_c={div_c}"
            )
            for _ in range(seg_len - 1):
                c_cur = int(c_cur + dir_h)
                out[r_cur, c_cur] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode tiled 3x3 glyphs into a colored polyline that rises across the left canvas.

    Input structure:
    - Left of the full-height divider: monochromatic 3x3 glyphs arranged in two tile columns on the background.
    - Right of the divider: exactly one gray (5) start pixel, placed on the bottom row.

    Rule:
    - Remove the divider and shift the gray start into the 7-column left canvas.
    - Read the glyph bank in row-major tile order: for each tile row, process the left glyph first and the right glyph second.
    - Convert glyph area into the segment instruction: area 4 -> up 2, area 5 -> right 3, area 6 -> left 4, area 7 -> left 2.
    - Before each segment, move one row upward, color that cell, then extend the rest of the segment in the decoded direction.

    Key insight:
    - The glyph shape still encodes the segment type, but the instruction sequence now comes from row-wise pairing while the whole trace is anchored at the bottom and constructed upward.
    """
    return _solve_with_controls(grid, row_major=True, grow_up=True)


N_MIN, N_MAX = 2, 5
GRID_W = 15
DIVIDER_COL = 7
DIVIDER_COLOR = 4
LEFT_W = 7
TILE_COLS = (0, 4)
TILE_ROW_STRIDE = 4
GRAY = 5

_TEMPLATES = {
    4: np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
    5: np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int),
    6: np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=int),
    7: np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=int),
}
_AREA_TO_COLOR = {4: 6, 5: 1, 6: 3, 7: 2}
_AREA_DELTA_UP = {4: (-2, 0), 5: (-1, 2), 6: (-1, -3), 7: (-1, -1)}
_AREAS = (4, 5, 6, 7)

MIN_DIRECTION_CHANGES = 2
MAX_ATTEMPTS = 500


def _has_instruction_coverage(areas: list[int]) -> bool:
    """Require visible evidence for vertical, rightward, and leftward behaviors."""
    seen = set(areas)
    return 4 in seen and 5 in seen and (6 in seen or 7 in seen) and len(seen) >= 3


def _sample_path(n: int, start_col: int) -> list[int] | None:
    """Sample a valid upward-growing row-major instruction list of length 2*n."""
    h = 4 * n - 1
    r, c = h - 1, start_col
    areas: list[int] = []
    for _ in range(2 * n):
        opts = [
            area
            for area in _AREAS
            if 0 <= r + _AREA_DELTA_UP[area][0] < h and 0 <= c + _AREA_DELTA_UP[area][1] < LEFT_W
        ]
        if not opts:
            return None
        area = random.choice(opts)
        areas.append(area)
        dr, dc = _AREA_DELTA_UP[area]
        r += dr
        c += dc

    types = [area == 4 for area in areas]
    changes = sum(1 for i in range(1, len(types)) if types[i] != types[i - 1])
    if changes < MIN_DIRECTION_CHANGES:
        return None
    if not _has_instruction_coverage(areas):
        return None
    return areas


def _render_grid(n: int, start_col: int, areas: list[int]) -> np.ndarray:
    """Paint the bottom gray start, divider, and glyph bank in row-major tile order."""
    h = 4 * n - 1
    grid = np.zeros((h, GRID_W), dtype=int)
    grid[:, DIVIDER_COL] = DIVIDER_COLOR
    grid[h - 1, DIVIDER_COL + 1 + start_col] = GRAY

    for i, area in enumerate(areas):
        tile_col = TILE_COLS[i % 2]
        r0 = TILE_ROW_STRIDE * (i // 2)
        grid[r0 : r0 + 3, tile_col : tile_col + 3] = _TEMPLATES[area] * _AREA_TO_COLOR[area]
    return grid


def _column_major_permutation(areas: list[int]) -> list[int]:
    n = len(areas) // 2
    left_col = [areas[2 * r] for r in range(n)]
    right_col = [areas[2 * r + 1] for r in range(n)]
    return left_col + right_col


def _variant_observable(inp: np.ndarray, areas: list[int]) -> bool:
    if areas == _column_major_permutation(areas):
        return False

    combo_out = _solve_with_controls(inp, row_major=True, grow_up=True)
    col_up_out = _solve_with_controls(inp, row_major=False, grow_up=True)
    if np.array_equal(combo_out, col_up_out):
        return False

    for row_major, grow_up in ((True, False), (False, False)):
        try:
            alt_out = _solve_with_controls(inp, row_major=row_major, grow_up=grow_up)
        except Exception:
            continue
        if np.array_equal(combo_out, alt_out):
            return False

    gray = np.argwhere(inp == GRAY)
    return gray.shape == (1, 2) and int(gray[0, 0]) == inp.shape[0] - 1


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(N_MIN, N_MAX)
    start_col = random.randint(0, LEFT_W - 1)
    areas = _sample_path(n, start_col)
    if areas is None:
        return None

    inp = _render_grid(n, start_col, areas)
    if not _variant_observable(inp, areas):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            inp = _gen_attempt()
            if inp is None:
                continue
            out = main(inp)
            if out.shape != (inp.shape[0], LEFT_W):
                continue
            return inp
        except Exception:
            continue
    raise RuntimeError(f"generate_input: failed after {MAX_ATTEMPTS} attempts")
