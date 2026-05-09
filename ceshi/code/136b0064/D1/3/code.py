# Variant of 136b0064
# Dimension: D1 — 字形处理顺序——排序规则 (match_cond)
# Change: 按列主序（先按 tile 列 c0 从左到右，再按 tile 行 r0 从上到下）依次处理字形... → 列主序但从右到左、从下到上

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg


# Design analysis for the D1 variant
# ----------------------------------
# 1) Concretization and behavior trace
#    - The input still shows the same visual ingredients as the base task:
#      two left-side columns of 3x3 glyphs, a full-height divider at col 7,
#      and one gray start pixel on the right side.
#    - The output is still a 7-column canvas containing the gray start and the
#      colored polyline decoded from glyph areas.
#    - Only the glyph traversal order changes. Base task order:
#        left column top->bottom, then right column top->bottom.
#      Variant order:
#        right column bottom->top, then left column bottom->top.
#    - Small concrete example with N=2 and physical slots:
#        LT, LB, RT, RB  (L/R = tile column, T/B = top/bottom row)
#      Base order reads:    LT -> LB -> RT -> RB
#      Variant order reads: RB -> RT -> LB -> LT
#      Therefore the segment nearest the gray start is now determined by the
#      bottom-right glyph, not the top-left glyph.
#
# 2) Complexity calibration
#    - The glyph vocabulary stays small and unchanged: four templates keyed by
#      area {4,5,6,7}. This keeps the task learnable.
#    - To keep the traversal reversal visible, the generator rejects degenerate
#      cases where the produced output matches the output from the legacy order.
#      Since the same physical grid would be read in the exact reverse sequence
#      by the legacy solver, we compare against reversed(area_sequence).
#
# 3) Implicit-assumption audit for main()
#    - Divider detection still assumes one full-height constant non-bg column.
#      This remains true because generation keeps the same divider.
#    - Gray handling still assumes exactly one gray start on the right side and
#      that removing the divider shifts it into cols 0..6. Generation enforces
#      this exactly as before.
#    - Component parsing still assumes glyphs are monochromatic 3x3 connected
#      objects separated by blank rows/cols. Generation preserves that layout.
#    - The only solver line whose semantics must change for D1 is the sort key.
#      Old key: (c0, r0). New key: (-c0, -r0).
#    - Drawing code assumes the decoded sequence stays in bounds. The generator
#      now samples the walk in variant order, so those bounds remain valid.
#
# 4) Placement legality
#    - Glyph tiles still occupy fixed slots: c0 in {0,4}, r0 in {0,4,8,...}.
#    - The generator samples a valid polyline first, then stamps glyphs into the
#      physical slots that correspond to the variant traversal order:
#        right column bottom->top, then left column bottom->top.
#    - This guarantees no glyph overlap, no divider overlap, and a solver-ready
#      input whose decoded walk stays inside the 7-column output canvas.
#
# 5) Visual observability and cleanliness
#    - Sorting depends only on visible tile positions, never on color ids.
#    - The input remains clean: glyph area, divider, and gray start occupy
#      disjoint regions with the original spacing.
#    - The generator rejects samples where first and last instructions are equal
#      or where reversing the instruction order yields the same final output.
#      This makes the reversed reading order visibly matter in the examples.


def find_full_height_constant_col(grid: np.ndarray, *, bg: int | None = None) -> tuple[int | None, int | None]:
    """
    Local helper: find a column whose values are all identical and not bg.
    Returns (col_idx, color) or (None, None).
    """
    H, W = grid.shape
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
        col = int(comp["color"])
        if col in ignore:
            continue
        b = comp["bbox"]
        if bbox_h(b) == 3 and bbox_w(b) == 3:
            r0, c0, _r1, _c1 = map(int, b)
            glyphs.append({"color": col, "r0": r0, "c0": c0, "area": int(comp["area"])})
    return glyphs


def _trace_polyline(H: int, start_col: int, areas: list[int]) -> np.ndarray:
    out = np.zeros((int(H), LEFT_W), dtype=int)
    r_cur, c_cur = GRAY_START_ROW, int(start_col)
    out[r_cur, c_cur] = GRAY

    for area in areas:
        color = int(_AREA_TO_COLOR[area])
        seg_len = int(_AREA_TO_LEN[area])

        r_cur += 1
        assert r_cur < int(H), f"segment exits bottom: r_cur={r_cur}, H={H}"
        out[r_cur, c_cur] = color

        if int(area) == 4:
            end_r = r_cur + (seg_len - 1)
            assert end_r < int(H), f"vertical segment exits bottom: end_r={end_r}, H={H}"
            for _ in range(seg_len - 1):
                r_cur += 1
                out[r_cur, c_cur] = color
        else:
            dir_h = int(_AREA_TO_DIR[area])
            end_c = c_cur + dir_h * (seg_len - 1)
            assert 0 <= end_c < LEFT_W, (
                f"horizontal segment exits canvas: c_cur={c_cur}, dir={dir_h}, "
                f"seg_len={seg_len}, LEFT_W={LEFT_W}"
            )
            for _ in range(seg_len - 1):
                c_cur += dir_h
                out[r_cur, c_cur] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a polyline by decoding 3x3 glyph instructions from a divided grid.

    Input structure:
    - Left of the full-height divider: colored 3x3 glyphs arranged in two fixed tile columns.
    - Right of the divider: exactly one gray (5) start pixel on an otherwise blank field.

    Rule:
    - Remove the divider and shift the gray start onto the 7-column left canvas.
    - Read glyphs in reversed column-major order: right tile column before left, and within each column bottom→top.
    - Decode each glyph by pixel area: area 4 → down 2, area 5 → right 3, area 6 → left 4, area 7 → left 2.
    - Starting from the gray point, step down one row per glyph and draw the decoded colored segment.

    Key insight: The glyph shape still determines the segment through its area alone; the variant changes only which visible tile position is interpreted first, second, and so on.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    div_c, div_color = find_full_height_constant_col(g, bg=bg)
    assert div_c is not None

    out = np.zeros((H, int(div_c)), dtype=int)

    gray = np.argwhere(g == GRAY)
    if gray.size == 0:
        return out
    r_cur, c_cur = map(int, gray[0])
    c_cur = int(c_cur - (int(div_c) + 1))
    if not (0 <= c_cur < int(div_c)):
        return out
    out[r_cur, c_cur] = GRAY

    left = g[:, : int(div_c)]
    glyphs = _collect_glyphs(left, bg=bg, div_color=div_color)
    glyphs.sort(key=lambda d: (-int(d["c0"]), -int(d["r0"])))

    for glyph in glyphs:
        color = int(glyph["color"])
        area = int(glyph["area"])
        seg_len = int(_AREA_TO_LEN.get(area, 2))

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
            dir_h = int(_AREA_TO_DIR.get(area, 1))
            end_c = int(c_cur) + dir_h * (seg_len - 1)
            assert 0 <= end_c < int(div_c), (
                f"horizontal segment exits canvas: c_cur={c_cur}, dir={dir_h}, "
                f"seg_len={seg_len}, div_c={div_c}"
            )
            for _ in range(seg_len - 1):
                c_cur = int(c_cur + dir_h)
                out[r_cur, c_cur] = color

    return out


# --------------- generator ---------------

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
_AREA_TO_LEN = {4: 2, 5: 3, 6: 4, 7: 2}
_AREA_TO_DIR = {5: 1, 6: -1, 7: -1}
_AREA_DELTA = {4: (2, 0), 5: (1, 2), 6: (1, -3), 7: (1, -1)}
_AREAS = (4, 5, 6, 7)

MIN_DIRECTION_CHANGES = 2
MAX_ATTEMPTS = 1200


def _sample_path(N: int, start_col: int) -> list[int] | None:
    H = 4 * N - 1
    r, c = GRAY_START_ROW, int(start_col)
    areas: list[int] = []

    for _ in range(2 * N):
        opts = [a for a in _AREAS if r + _AREA_DELTA[a][0] < H and 0 <= c + _AREA_DELTA[a][1] < LEFT_W]
        if not opts:
            return None
        area = random.choice(opts)
        areas.append(int(area))
        dr, dc = _AREA_DELTA[area]
        r += dr
        c += dc

    types = [a == 4 for a in areas]
    changes = sum(1 for i in range(1, len(types)) if types[i] != types[i - 1])
    if changes < MIN_DIRECTION_CHANGES:
        return None
    if len(set(areas)) < 2:
        return None
    if areas[0] == areas[-1]:
        return None
    return areas


def _variant_slots(N: int) -> list[tuple[int, int]]:
    right = [(TILE_ROW_STRIDE * row_idx, TILE_COLS[1]) for row_idx in range(N - 1, -1, -1)]
    left = [(TILE_ROW_STRIDE * row_idx, TILE_COLS[0]) for row_idx in range(N - 1, -1, -1)]
    return right + left


def _render_grid(N: int, start_col: int, areas: list[int]) -> np.ndarray:
    H = 4 * N - 1
    grid = np.zeros((H, GRID_W), dtype=int)
    grid[:, DIVIDER_COL] = DIVIDER_COLOR
    grid[GRAY_START_ROW, DIVIDER_COL + 1 + int(start_col)] = GRAY

    for area, (r0, c0) in zip(areas, _variant_slots(N)):
        grid[r0:r0 + 3, c0:c0 + 3] = _TEMPLATES[int(area)] * _AREA_TO_COLOR[int(area)]

    return grid


def _gen_attempt() -> np.ndarray | None:
    N = random.randint(N_MIN, N_MAX)
    H = 4 * N - 1
    start_col = random.randint(0, LEFT_W - 1)
    areas = _sample_path(N, start_col)
    if areas is None:
        return None

    expected = _trace_polyline(H, start_col, areas)
    try:
        legacy_expected = _trace_polyline(H, start_col, list(reversed(areas)))
    except AssertionError:
        legacy_expected = None
    if legacy_expected is not None and np.array_equal(expected, legacy_expected):
        return None

    inp = _render_grid(N, start_col, areas)
    solved = main(inp)
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError(f"generate_input: failed after {MAX_ATTEMPTS} attempts")
