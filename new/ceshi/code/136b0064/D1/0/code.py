# Variant of 136b0064
# Dimension: D1 — 字形处理顺序——排序规则 (match_cond)
# Change: 按列主序（先按 tile 列 c0 从左到右，再按 tile 行 r0 从上到下）依次处理字形... → 行主序处理（先上到下，再左到右）

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg

# Design analysis for the D1 variant
# ==================================
# a) Concrete behavior:
#    The input geometry stays the same as the original task: a 7-column left canvas
#    with 2 glyph columns (c0=0 and c0=4), a full-height divider at col 7, and a
#    single gray start pixel on the right. The visible change is only the order in
#    which the 3x3 glyphs are interpreted. Original behavior decoded all glyphs in
#    the left tile column first, then all glyphs in the right tile column. This
#    variant decodes row by row: at each tile row, first the left glyph, then the
#    right glyph, then move to the next tile row. Example with N=3:
#      tile slots on the left:
#        row 0: A at (0,0), B at (0,4)
#        row 1: C at (4,0), D at (4,4)
#        row 2: E at (8,0), F at (8,4)
#      original order: A, C, E, B, D, F
#      new order:      A, B, C, D, E, F
#    The drawn polyline therefore changes even though the glyph set, shapes, colors,
#    start marker, and canvas layout are unchanged.
#
# b) Complexity calibration:
#    The rule change is small but can easily become invisible if the generator emits
#    sequences where row-major and column-major happen to produce the same decoded
#    walk. To avoid that degeneration, generation rejects samples whose row-major
#    output equals the old column-major output on the same input. This keeps the
#    variant visually observable without adding unrelated dimensions.
#
# c) Old implicit assumptions audited:
#    - Divider detection via a full-height constant non-bg column still holds.
#    - Gray start handling still holds because only processing order changes.
#    - Component extraction of 3x3 monochromatic glyphs still holds.
#    - The critical old assumption is `glyphs.sort(key=(c0,r0))`, which encoded the
#      original column-major traversal. In this variant that assumption is false, so
#      the solver must sort by `(r0,c0)` instead.
#    - Generator placement also previously assumed "first N areas fill left column,
#      next N fill right column". That must change to row-major placement or the new
#      solver would decode the wrong path.
#
# d) Placement legality:
#    Glyph footprints remain separated by one blank column and one blank row, so
#    `find_components(..., monochromatic=True)` still isolates each glyph cleanly.
#    The path is sampled in output space first, with bounds checks against the 7-wide
#    left canvas and the current height H=4*N-1, so no drawn segment exits the grid.
#
# e) Visual observability:
#    The variant never relies on hidden properties such as color id sorting. Humans
#    can inspect the 2-column tile layout and infer that examples now pair the two
#    glyphs in each row before moving downward. The rejection test against the old
#    traversal ensures the changed order actually matters in produced samples.
#
# f) Visual cleanliness:
#    No new marks are added. The left glyph area, divider, and right-side start pixel
#    remain spatially separated, and the output remains a single colored polyline plus
#    the gray start pixel.
#
# g) Color inference:
#    Color semantics are unchanged: each fixed glyph shape still determines a fixed
#    segment color. The variant only changes the read order, so no new color mapping
#    ambiguity is introduced.
#
# New solver assumptions checked against generation:
# - Exactly one full-height divider column exists.
# - The gray start shifts into columns 0..6 after removing the divider.
# - Every intended glyph is a 3x3 monochromatic 8-connected component.
# - The sampled instruction sequence is placed in row-major tile order.
# - At least one generated sample property distinguishes this variant from the old
#   one: `main(inp) != _solve_with_order(inp, row_major=False)`.


def find_full_height_constant_col(grid: np.ndarray, *, bg: int | None = None) -> tuple[int | None, int | None]:
    """
    Local helper (only used by this solver): find a column where all values are identical and != bg.
    Returns (col_idx, color) or (None, None).
    """
    _H, W = grid.shape
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    for c in range(int(W)):
        col = grid[:, int(c)]
        if (col == col[0]).all() and int(col[0]) != bg:
            return int(c), int(col[0])
    return None, None


def _collect_glyphs(left: np.ndarray, *, bg: int, div_color: int | None) -> list[dict[str, int]]:
    ignore = {int(bg), 5}
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


def _solve_with_order(grid: np.ndarray, *, row_major: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, _W = g.shape
    bg = int(get_bg(g))

    div_c, div_color = find_full_height_constant_col(g, bg=bg)
    assert div_c is not None

    out = np.zeros((H, int(div_c)), dtype=int)

    gray = np.argwhere(g == 5)
    if gray.size == 0:
        return out
    r_cur, c_cur = map(int, gray[0])
    c_cur = int(c_cur - (int(div_c) + 1))
    if not (0 <= c_cur < int(div_c)):
        return out
    out[int(r_cur), int(c_cur)] = 5

    left = g[:, : int(div_c)]
    glyphs = _collect_glyphs(left, bg=bg, div_color=div_color)
    if row_major:
        glyphs.sort(key=lambda d: (int(d["r0"]), int(d["c0"])))
    else:
        glyphs.sort(key=lambda d: (int(d["c0"]), int(d["r0"])))

    area_to_len = {4: 2, 5: 3, 6: 4, 7: 2}
    area_to_dir = {5: 1, 6: -1, 7: -1}

    for glyph in glyphs:
        color = int(glyph["color"])
        area = int(glyph["area"])
        seg_len = int(area_to_len.get(area, 2))

        r_cur = int(r_cur + 1)
        assert r_cur < H, f"segment exits bottom: r_cur={r_cur}, H={H}"
        out[r_cur, int(c_cur)] = color

        if area == 4:
            end_r = r_cur + (seg_len - 1)
            assert end_r < H, f"vertical segment exits bottom: end_r={end_r}, H={H}"
            for _ in range(seg_len - 1):
                r_cur = int(r_cur + 1)
                out[r_cur, int(c_cur)] = color
        else:
            dir_h = int(area_to_dir.get(area, 1))
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
    Decode tiled 3x3 glyphs into a colored polyline drawn on the left canvas.

    Input structure:
    - Left of a full-height divider: monochromatic 3x3 glyphs placed in two tile columns on a black background.
    - Right of the divider: exactly one gray (5) start pixel indicating where the walk begins after the divider is removed.

    Rule:
    - Drop the divider and shift the gray start into the 7-column left canvas.
    - Read glyphs in row-major tile order (top row first, and within each row left tile before right tile).
    - Use each glyph's pixel area to choose the segment: area 4 -> down 2, area 5 -> right 3, area 6 -> left 4, area 7 -> left 2.
    - For each decoded glyph, step down one row, color that cell, then extend the rest of the segment in the decoded direction.

    Key insight:
    - The 3x3 shape still determines the segment type, but the instruction sequence now comes from row-wise scanning across paired glyphs instead of finishing one glyph column before the other.
    """
    return _solve_with_order(grid, row_major=True)


# --------------- generator (row-major D1 variant) ---------------

N_MIN, N_MAX = 2, 5
GRID_W = 15
DIVIDER_COL = 7
DIVIDER_COLOR = 4
LEFT_W = 7
TILE_COLS = (0, 4)
TILE_ROW_STRIDE = 4
GRAY = 5
GRAY_START_ROW = 0

_TEMPLATES = {
    4: np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
    5: np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int),
    6: np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=int),
    7: np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=int),
}
_AREA_TO_COLOR = {4: 6, 5: 1, 6: 3, 7: 2}
_AREA_DELTA = {4: (2, 0), 5: (1, 2), 6: (1, -3), 7: (1, -1)}
_AREAS = (4, 5, 6, 7)

MIN_DIRECTION_CHANGES = 2
MAX_ATTEMPTS = 400


def _sample_path(N: int, start_col: int) -> list[int] | None:
    """Random-walk a sequence of 2*N area instructions that stays in the left canvas."""
    H = 4 * N - 1
    r, c = GRAY_START_ROW, start_col
    areas: list[int] = []
    for _ in range(2 * N):
        opts = [a for a in _AREAS if r + _AREA_DELTA[a][0] < H and 0 <= c + _AREA_DELTA[a][1] < LEFT_W]
        if not opts:
            return None
        area = random.choice(opts)
        areas.append(area)
        dr, dc = _AREA_DELTA[area]
        r += dr
        c += dc

    types = [a == 4 for a in areas]
    changes = sum(1 for i in range(1, len(types)) if types[i] != types[i - 1])
    if changes < MIN_DIRECTION_CHANGES:
        return None
    return areas


def _render_grid(N: int, start_col: int, areas: list[int]) -> np.ndarray:
    """Paint divider, gray start, and all 2*N tiled glyphs into the input grid."""
    H = 4 * N - 1
    grid = np.zeros((H, GRID_W), dtype=int)
    grid[:, DIVIDER_COL] = DIVIDER_COLOR
    grid[GRAY_START_ROW, DIVIDER_COL + 1 + start_col] = GRAY

    # Row-major tiling: every tile row contributes left glyph first, then right glyph.
    for i, area in enumerate(areas):
        tile_col = TILE_COLS[i % 2]
        r0 = TILE_ROW_STRIDE * (i // 2)
        grid[r0:r0 + 3, tile_col:tile_col + 3] = _TEMPLATES[area] * _AREA_TO_COLOR[area]
    return grid


def _column_major_permutation(areas: list[int]) -> list[int]:
    N = len(areas) // 2
    left_col = [areas[2 * r] for r in range(N)]
    right_col = [areas[2 * r + 1] for r in range(N)]
    return left_col + right_col


def _variant_observable(inp: np.ndarray, areas: list[int]) -> bool:
    # Ensure the changed traversal both permutes the glyph sequence and changes the output.
    if areas == _column_major_permutation(areas):
        return False
    return not np.array_equal(main(inp), _solve_with_order(inp, row_major=False))


def _gen_attempt() -> np.ndarray | None:
    N = random.randint(N_MIN, N_MAX)
    start_col = random.randint(0, LEFT_W - 1)
    areas = _sample_path(N, start_col)
    if areas is None:
        return None

    inp = _render_grid(N, start_col, areas)
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
