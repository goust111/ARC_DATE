# Combo variant of 136b0064 (index 18)
# combo_spec: D6/0+D9/1
# Name: 水平分区与宽字形布局

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior + trace:
#    The original task uses a tall left canvas: glyphs fill a 2-by-N bank on the left of a vertical
#    divider, the gray start sits on the right, and each glyph first steps one row downward before
#    drawing its segment on an Hx7 output. This fusion rotates the usable geometry into a short-wide
#    frame. The input becomes 15 x (4*N-1): rows 0..6 hold a 2-row x N-column glyph bank, row 7 is
#    the divider, and rows 8..14 hold the gray start. The output is the lower 7-row canvas.
#    For N=3 the visible tile slots are:
#      top row glyphs:    A at (0,0), C at (0,4), E at (0,8)
#      bottom row glyphs: B at (4,0), D at (4,4), F at (4,8)
#    Column-major wide-layout order is A,B,C,D,E,F.
#    We also rotate the walk itself so the long dimension is now horizontal. Example with areas
#    [5,4,7,6,4,5] and shifted gray start at (4,0):
#      A area 5 -> step to (4,1), draw up len 3, end at (2,1)
#      B area 4 -> step to (2,2), draw right len 2, end at (2,3)
#      C area 7 -> step to (2,4), draw down len 2, end at (3,4)
#      D area 6 -> step to (3,5), draw down len 4, end at (6,5)
#      ...
#    The fused differences are therefore structural, not cosmetic: divider orientation, glyph-bank
#    packing, decoding scan path, and trace frame all move into a horizontal composition.
#
# b) Complexity calibration:
#    The task still uses only four recurring glyph types with the original strict shape->color and
#    shape->segment bindings, so the instruction alphabet remains low-complexity and well-supported.
#    To keep the D9/1 change visible, generation rejects samples whose output would be identical if
#    the same wide glyph bank were misread row-major instead of column-major. N is sampled in 3..5
#    so the top bank is visibly wider than a trivial 2-column square.
#
# c) Implicit-assumption audit:
#    - "Divider = full-height constant column" is invalid; solver must search for a full-width
#      constant non-background row instead.
#    - "Output width equals divider column index" is invalid; output is now the lower block with
#      height `H - div_r - 1` and the full input width.
#    - "Gray shift happens in columns" is invalid; the gray point now shifts upward by removing the
#      divider row and the whole glyph bank above it.
#    - "Glyph bank lives in `g[:, :div_c]` and sorts by (c0,r0)" is only half-valid. The bank is
#      now `g[:div_r, :]`, but the fused wide layout still uses visible column-major order, so the
#      sorting key remains `(c0, r0)` on the top-half components.
#    - "Each instruction begins with `r_cur += 1`" is invalid after rotation. The new primitive
#      begins with `c_cur += 1`; area 4 extends horizontally, while areas 5/6/7 extend vertically.
#    These changes do not conflict: D6/0 defines the new frame geometry, and D9/1 supplies the new
#    glyph placement that the same `(c0,r0)` ordering reads naturally as columns of a 2-row bank.
#
# d) Placement legality:
#    The generator samples the lower-canvas walk first on a 7 x (4*N-1) board from `(start_row,0)`.
#    It uses total deltas {4:(0,+2), 5:(-2,+1), 6:(+3,+1), 7:(+1,+1)}, which already encode the
#    mandatory "step right once, then draw the segment" rule. A candidate area is accepted only if
#    the resulting endpoint stays inside rows 0..6 and columns 0..W-1. Glyphs are then stamped into
#    fixed 3x3 slots `(r0 in {0,4}, c0 = 0,4,8,...)`, so they never overlap and always fit.
#
# e) Visual observability:
#    A human sees the horizontal split directly: all instructions are above the yellow divider and
#    the lone gray start is below it. The bank is obviously N columns by 2 rows, and the output is a
#    short-wide 7-row strip whose path advances across columns rather than down rows. Because the
#    generator rejects row-major degeneracies, the top/bottom pairing inside each column carries real
#    information in the examples.
#
# f) Visual cleanliness:
#    The top and bottom roles are separated by a single solid divider row, leaving no region where
#    glyphs and traced segments can visually interfere. The bottom half contains only one gray seed,
#    and the output contains only that seed plus a single colored polyline, so the scene stays easy
#    to parse.
#
# g) Change inferability:
#    No new color mapping is introduced: the same four glyph silhouettes still imply the same colors
#    and segment types. The new inference burden is purely spatial and repeated across every sample:
#    "read the wide bank column by column" and "build the path on the lower horizontal canvas by
#    stepping right before each segment."
#
# New solver assumptions guaranteed by generation:
# - Exactly one gray cell exists in the lower half, always on input column 0.
# - The divider is a full-width row of color 4 at row 7.
# - Every intended glyph is a monochromatic 3x3 8-connected component in the top half.
# - Column-major decoding of the wide 2-row bank differs from row-major decoding on the same input.
# - The sampled rightward path never exits the 7-row output or the right boundary.


def find_full_width_constant_row(grid: np.ndarray, *, bg: int | None = None) -> tuple[int | None, int | None]:
    """Return `(row_idx, color)` for a full-width constant non-background row, else `(None, None)`."""
    h, _w = grid.shape
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    for r in range(int(h)):
        row = grid[int(r), :]
        if (row == row[0]).all() and int(row[0]) != bg:
            return int(r), int(row[0])
    return None, None


def _collect_glyphs(top: np.ndarray, *, bg: int, div_color: int | None) -> list[dict[str, int]]:
    ignore = {int(bg), GRAY}
    if div_color is not None:
        ignore.add(int(div_color))

    glyphs: list[dict[str, int]] = []
    for comp in find_components(top, bg=bg, connectivity=8, monochromatic=True, return_format="dict"):
        color = int(comp["color"])
        if color in ignore:
            continue
        b = comp["bbox"]
        if bbox_h(b) == 3 and bbox_w(b) == 3:
            r0, c0, _r1, _c1 = map(int, b)
            glyphs.append({"color": color, "r0": r0, "c0": c0, "area": int(comp["area"])})
    return glyphs


AREA_TO_LEN = {4: 2, 5: 3, 6: 4, 7: 2}
AREA_TO_VDIR = {5: -1, 6: 1, 7: 1}


def _solve_horizontal_wide(grid: np.ndarray, *, read_row_major: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))

    div_r, div_color = find_full_width_constant_row(g, bg=bg)
    assert div_r is not None

    out_h = int(h - (int(div_r) + 1))
    out = np.full((out_h, w), bg, dtype=int)

    gray = np.argwhere(g == GRAY)
    if gray.size == 0:
        return out
    r_cur, c_cur = map(int, gray[0])
    r_cur = int(r_cur - (int(div_r) + 1))
    if not (0 <= r_cur < out_h and 0 <= c_cur < w):
        return out
    out[r_cur, c_cur] = GRAY

    glyphs = _collect_glyphs(g[: int(div_r), :], bg=bg, div_color=div_color)
    if read_row_major:
        glyphs.sort(key=lambda d: (int(d["r0"]), int(d["c0"])))
    else:
        glyphs.sort(key=lambda d: (int(d["c0"]), int(d["r0"])))

    for glyph in glyphs:
        color = int(glyph["color"])
        area = int(glyph["area"])
        seg_len = int(AREA_TO_LEN[area])

        c_cur = int(c_cur + 1)
        assert c_cur < w, f"segment exits right edge: c_cur={c_cur}, W={w}"
        out[int(r_cur), c_cur] = color

        if area == 4:
            end_c = int(c_cur + (seg_len - 1))
            assert end_c < w, f"horizontal segment exits right edge: end_c={end_c}, W={w}"
            for _ in range(seg_len - 1):
                c_cur = int(c_cur + 1)
                out[int(r_cur), c_cur] = color
        else:
            dir_v = int(AREA_TO_VDIR[area])
            end_r = int(r_cur + dir_v * (seg_len - 1))
            assert 0 <= end_r < out_h, f"vertical segment exits canvas: end_r={end_r}, out_h={out_h}"
            for _ in range(seg_len - 1):
                r_cur = int(r_cur + dir_v)
                out[r_cur, int(c_cur)] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a wide 3x3 glyph bank into a colored polyline on the lower horizontal canvas.

    Input structure:
    - Above a full-width horizontal divider: monochromatic 3x3 glyphs arranged as N columns x 2 rows on the background.
    - Below the divider: exactly one gray (5) start pixel, with the rest of the lower half empty.

    Rule:
    - Remove the divider and the upper glyph bank, shifting the gray start into lower-canvas coordinates.
    - Read the top glyph bank in visible column-major order: within each glyph column, process the top tile before the bottom tile.
    - Convert glyph area into a segment instruction on the lower canvas: area 4 -> right 2, area 5 -> up 3, area 6 -> down 4, area 7 -> down 2.
    - Before each segment, move one column to the right, color that cell, then extend the remaining cells of the segment in the decoded direction.

    Key insight:
    - The glyph silhouettes still encode the instruction types, but the whole execution frame has been rotated: the cues live in an upper wide bank and the trace unfolds across the lower wide strip from left to right.
    """
    return _solve_horizontal_wide(grid, read_row_major=False)


INPUT_H = 15
TOP_H = 7
BOTTOM_H = 7
DIVIDER_ROW = 7
DIVIDER_COLOR = 4
GRAY = 5
N_MIN, N_MAX = 3, 5
TILE_ROW_ORIGINS = (0, 4)
TILE_COL_STRIDE = 4

_TEMPLATES = {
    4: np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
    5: np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int),
    6: np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=int),
    7: np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=int),
}
_AREA_TO_COLOR = {4: 6, 5: 1, 6: 3, 7: 2}
_AREA_DELTA = {4: (0, 2), 5: (-2, 1), 6: (3, 1), 7: (1, 1)}
_AREAS = (4, 5, 6, 7)
REQUIRED_AREAS = frozenset(_AREAS)

MIN_DIRECTION_CHANGES = 2
MAX_ATTEMPTS = 600


def _sample_path(n_cols: int, start_row: int) -> list[int] | None:
    """Sample a valid right-growing instruction list of length `2*n_cols` on the 7-row output."""
    out_w = 4 * n_cols - 1
    r, c = int(start_row), 0
    areas: list[int] = []
    used: set[int] = set()

    for step in range(2 * n_cols):
        opts = [
            area
            for area in _AREAS
            if 0 <= r + _AREA_DELTA[area][0] < BOTTOM_H and c + _AREA_DELTA[area][1] < out_w
        ]
        if not opts:
            return None

        remaining = 2 * n_cols - step - 1
        must_cover = REQUIRED_AREAS - used
        if must_cover:
            covering_opts = [
                area for area in opts if len(REQUIRED_AREAS - (used | {area})) <= remaining
            ]
            if covering_opts:
                opts = covering_opts

        prev_is_horizontal = areas[-1] == 4 if areas else None
        weights: list[int] = []
        for area in opts:
            weight = 1
            if area not in used:
                weight += 4
            if area == 6:
                weight += 2
            if prev_is_horizontal is not None and (area == 4) != prev_is_horizontal:
                weight += 1
            weights.append(weight)

        area = random.choices(opts, weights=weights, k=1)[0]
        areas.append(area)
        used.add(area)
        dr, dc = _AREA_DELTA[area]
        r += dr
        c += dc

    if used != REQUIRED_AREAS:
        return None

    segment_is_horizontal = [area == 4 for area in areas]
    changes = sum(
        1
        for i in range(1, len(segment_is_horizontal))
        if segment_is_horizontal[i] != segment_is_horizontal[i - 1]
    )
    if changes < MIN_DIRECTION_CHANGES:
        return None
    return areas


def _render_input(n_cols: int, start_row: int, areas: list[int]) -> np.ndarray:
    """Paint the upper glyph bank, divider row, and lower gray start into the fused input grid."""
    out_w = 4 * n_cols - 1
    grid = np.zeros((INPUT_H, out_w), dtype=int)
    grid[DIVIDER_ROW, :] = DIVIDER_COLOR
    grid[DIVIDER_ROW + 1 + start_row, 0] = GRAY

    # Wide column-major layout: top glyph then bottom glyph for each glyph column.
    for i, area in enumerate(areas):
        glyph_col = i // 2
        glyph_row = i % 2
        r0 = TILE_ROW_ORIGINS[glyph_row]
        c0 = TILE_COL_STRIDE * glyph_col
        grid[r0 : r0 + 3, c0 : c0 + 3] = _TEMPLATES[area] * _AREA_TO_COLOR[area]
    return grid


def _gen_attempt() -> np.ndarray | None:
    n_cols = random.randint(N_MIN, N_MAX)
    start_row = random.randint(0, BOTTOM_H - 1)
    areas = _sample_path(n_cols, start_row)
    if areas is None:
        return None

    inp = _render_input(n_cols, start_row, areas)
    out = _solve_horizontal_wide(inp, read_row_major=False)
    wrong_order_out = _solve_horizontal_wide(inp, read_row_major=True)
    if np.array_equal(out, wrong_order_out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            inp = _gen_attempt()
            if inp is None:
                continue
            out = main(inp)
            expected_w = inp.shape[1]
            if out.shape != (BOTTOM_H, expected_w):
                continue
            return inp
        except Exception:
            continue
    raise RuntimeError(f"generate_input: failed after {MAX_ATTEMPTS} attempts")
