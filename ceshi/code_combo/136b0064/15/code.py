# Combo variant of 136b0064 (index 15)
# combo_spec: D1/3+D8/3
# Name: 逆序扫描与大间隔步进

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg


# Design analysis for the combo variant
# -------------------------------------
# a) Concretization and behavior trace
#    - The input still has the same three visible regions: left-side 3x3 glyphs,
#      a full-height divider, and a single gray start pixel on the right.
#    - Two mechanics change together:
#        1. glyph traversal is reversed column-major:
#           right column before left, and within each column bottom->top;
#        2. before drawing each decoded segment, the cursor drops 2 rows instead
#           of 1.
#    - Small concrete trace with N=2 and physical slots LT, LB, RT, RB:
#        base read order:  LT -> LB -> RT -> RB
#        combo read order: RB -> RT -> LB -> LT
#      If the start is at (0,3) and the first two decoded areas are [5, 4],
#      then the combo path draws:
#        area 5: first painted row is 2, segment runs right on row 2
#        area 4: next segment starts on row 4, then extends to row 5
#      So the first instruction comes from the bottom-right glyph, and the
#      polyline shows one blank row between successive painted rows.
#
# b) Complexity calibration
#    - The glyph vocabulary and strict area->color binding stay unchanged, so
#      local evidence remains repeated and learnable.
#    - The larger step would look weak if the output barely turned, so the
#      generator requires both vertical and horizontal instructions plus at
#      least two vertical/horizontal alternations.
#    - To make the fused behavior non-degenerate, the generator rejects samples
#      whose output matches any partial rule:
#        reversed-order with old step size,
#        old order with step size 2,
#        or the full legacy rule.
#
# c) Implicit-assumption audit for main()
#    - Divider detection still assumes exactly one full-height constant non-bg
#      column; generation preserves that.
#    - Gray handling still assumes one right-side start pixel that shifts into
#      the 7-column left canvas after removing the divider; generation enforces
#      this.
#    - Component parsing still assumes monochromatic 3x3 glyphs with clean
#      spacing. The generator keeps fixed 3x3 templates in two visible columns.
#    - The original sort key (c0, r0) is no longer valid. The combo requires
#      (-c0, -r0) so visible position alone determines the reversed traversal.
#    - The original `r_cur += 1` encoded the old fixed step. It must become
#      `r_cur += 2`, and the generator must sample paths against that exact row
#      consumption.
#
# d) Placement legality
#    - To accommodate the larger vertical spacing cleanly, tile rows are placed
#      at r0 = 0, 5, 10, ... with two blank rows between glyph bboxes.
#    - Grid height is H = 5*N + 1, which fits the sparser output walk while
#      keeping a tidy lower margin and avoiding tile overlap.
#    - Path generation is output-first: it backtracks a valid in-bounds walk,
#      forbids segment self-overlap, then stamps glyphs into the physical slots
#      that correspond to the combo traversal order.
#
# e) Visual observability
#    - The traversal rule is visible from tile positions only: examples show
#      that the segment nearest the gray start is controlled by the bottom-right
#      glyph, not the top-left glyph.
#    - The step-size change is also visible: colored rows advance by 2 before
#      each new segment, so the polyline is noticeably more vertically sparse.
#    - No hidden metadata such as color ordering or object-discovery order is
#      used to decide the traversal.
#
# f) Visual cleanliness
#    - Glyphs, divider, and gray marker live in disjoint regions.
#    - The no-self-overlap constraint keeps the output readable despite the
#      longer walk and stronger vertical spacing.
#    - Using stride-5 tile rows avoids a cluttered top panel and keeps the
#      larger output height visually intentional rather than leaving arbitrary
#      extra blank rows only at the bottom.
#
# g) Change inferability
#    - Colors are still tied to the four visible glyph shapes, so no new color
#      inference burden is introduced.
#    - The fused rule can be inferred from a few examples because both changed
#      controls leave stable, repeated evidence: where decoding starts, and how
#      far the cursor drops before each segment.
#
# Post-change solver assumptions guaranteed by the generator:
#    - one divider column of a single non-bg color;
#    - one gray start on the right side;
#    - exactly 2*N glyphs, each a monochromatic 3x3 connected component;
#    - decoding in reversed column-major order stays in-bounds under step size 2.


GRID_W = 15
DIVIDER_COL = 7
DIVIDER_COLOR = 4
LEFT_W = 7
TILE_COLS = (0, 4)
TILE_ROW_STRIDE = 5
GRAY = 5
GRAY_START_ROW = 0
STEP_ROWS = 2
BOTTOM_MARGIN = 3

N_MIN, N_MAX = 2, 5
MIN_DIRECTION_CHANGES = 2
MAX_ATTEMPTS = 1500

_TEMPLATES = {
    4: np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
    5: np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int),
    6: np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=int),
    7: np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=int),
}
_AREA_TO_COLOR = {4: 6, 5: 1, 6: 3, 7: 2}
_AREA_TO_LEN = {4: 2, 5: 3, 6: 4, 7: 2}
_AREA_TO_DIR = {5: 1, 6: -1, 7: -1}
_AREAS = (4, 5, 6, 7)
_AREA_DELTA = {4: (3, 0), 5: (2, 2), 6: (2, -3), 7: (2, -1)}


def _grid_height(n_rows: int) -> int:
    return TILE_ROW_STRIDE * (int(n_rows) - 1) + 3 + BOTTOM_MARGIN


def find_full_height_constant_col(grid: np.ndarray, *, bg: int | None = None) -> tuple[int | None, int | None]:
    """
    Local helper: find a column whose values are all identical and not bg.
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
        bbox = comp["bbox"]
        if bbox_h(bbox) == 3 and bbox_w(bbox) == 3:
            r0, c0, _r1, _c1 = map(int, bbox)
            glyphs.append({"color": color, "r0": r0, "c0": c0, "area": int(comp["area"])})
    return glyphs


def _segment_cells(area: int, r_cur: int, c_cur: int, *, step_rows: int) -> list[tuple[int, int]]:
    start_r = int(r_cur + step_rows)
    if int(area) == 4:
        return [(start_r + offset, int(c_cur)) for offset in range(_AREA_TO_LEN[4])]

    dir_h = int(_AREA_TO_DIR[int(area)])
    return [(start_r, int(c_cur + dir_h * offset)) for offset in range(_AREA_TO_LEN[int(area)])]


def _trace_polyline(height: int, start_col: int, areas: list[int], *, step_rows: int) -> np.ndarray:
    out = np.zeros((int(height), LEFT_W), dtype=int)
    r_cur, c_cur = GRAY_START_ROW, int(start_col)
    out[int(r_cur), int(c_cur)] = GRAY

    for area in areas:
        color = int(_AREA_TO_COLOR[int(area)])
        cells = _segment_cells(int(area), r_cur, c_cur, step_rows=step_rows)
        for rr, cc in cells:
            assert 0 <= int(rr) < int(height), f"segment exits bottom: rr={rr}, H={height}"
            assert 0 <= int(cc) < LEFT_W, f"segment exits canvas: cc={cc}, LEFT_W={LEFT_W}"
            out[int(rr), int(cc)] = color
        r_cur, c_cur = map(int, cells[-1])

    return out


def _safe_trace(height: int, start_col: int, areas: list[int], *, step_rows: int) -> np.ndarray | None:
    try:
        return _trace_polyline(height, start_col, areas, step_rows=step_rows)
    except AssertionError:
        return None


def _count_direction_changes(areas: list[int]) -> int:
    kinds = [int(area) == 4 for area in areas]
    return sum(1 for i in range(1, len(kinds)) if kinds[i] != kinds[i - 1])


def _variant_slots(n_rows: int) -> list[tuple[int, int]]:
    right = [(TILE_ROW_STRIDE * row_idx, TILE_COLS[1]) for row_idx in range(int(n_rows) - 1, -1, -1)]
    left = [(TILE_ROW_STRIDE * row_idx, TILE_COLS[0]) for row_idx in range(int(n_rows) - 1, -1, -1)]
    return right + left


def _order_signal_is_visible(areas: list[int]) -> bool:
    # In this combo, the bottom-right glyph is decoded first while the top-left
    # glyph is decoded last. If those two instructions match, the reversed
    # traversal cue is weaker in examples, so reject that sample.
    return int(areas[0]) != int(areas[-1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode colored 3x3 glyphs into a sparse polyline on the left canvas.

    Input structure:
    - Left of the full-height divider: fixed-size 3x3 monochromatic glyphs arranged in two visible tile columns.
    - Right of the divider: exactly one gray start pixel on an otherwise blank field.

    Rule:
    - Remove the divider and shift the gray start onto the 7-column left canvas.
    - Read glyphs in reversed column-major order: right tile column first, and within each column bottom→top.
    - Interpret each glyph by pixel area exactly as before: area 4 → down 2, area 5 → right 3, area 6 → left 4, area 7 → left 2.
    - For every glyph, first move the drawing cursor down 2 rows, then draw the decoded segment in that glyph's color.

    Key insight: The shape-to-segment code is unchanged; what changes is the visible traversal order of the glyph panel together with the larger fixed gap before every segment, so the first decoded instruction comes from the bottom-right tile and the resulting path is vertically more spread out.
    """
    g = np.array(grid, dtype=int)
    height, _width = g.shape
    bg = int(get_bg(g))

    div_c, div_color = find_full_height_constant_col(g, bg=bg)
    assert div_c is not None

    out = np.zeros((int(height), int(div_c)), dtype=int)

    gray = np.argwhere(g == GRAY)
    if gray.size == 0:
        return out
    r_cur, c_cur = map(int, gray[0])
    c_cur = int(c_cur - (int(div_c) + 1))
    if not (0 <= c_cur < int(div_c)):
        return out
    out[int(r_cur), int(c_cur)] = GRAY

    left = g[:, : int(div_c)]
    glyphs = _collect_glyphs(left, bg=bg, div_color=div_color)
    glyphs.sort(key=lambda d: (-int(d["c0"]), -int(d["r0"])))

    for glyph in glyphs:
        color = int(glyph["color"])
        area = int(glyph["area"])

        r_cur = int(r_cur + STEP_ROWS)
        assert r_cur < int(height), f"segment exits bottom: r_cur={r_cur}, H={height}"
        out[int(r_cur), int(c_cur)] = color

        if area == 4:
            end_r = int(r_cur + (_AREA_TO_LEN[4] - 1))
            assert end_r < int(height), f"vertical segment exits bottom: end_r={end_r}, H={height}"
            for _ in range(_AREA_TO_LEN[4] - 1):
                r_cur = int(r_cur + 1)
                out[int(r_cur), int(c_cur)] = color
        else:
            dir_h = int(_AREA_TO_DIR[area])
            end_c = int(c_cur + dir_h * (_AREA_TO_LEN[area] - 1))
            assert 0 <= end_c < int(div_c), (
                f"horizontal segment exits canvas: c_cur={c_cur}, dir={dir_h}, "
                f"seg_len={_AREA_TO_LEN[area]}, div_c={div_c}"
            )
            for _ in range(_AREA_TO_LEN[area] - 1):
                c_cur = int(c_cur + dir_h)
                out[int(r_cur), int(c_cur)] = color

    return out


def _sample_path(n_rows: int, start_col: int) -> list[int] | None:
    """Backtrack a simple non-overlapping walk of length 2*N inside the output canvas."""
    height = _grid_height(n_rows)
    target_len = 2 * int(n_rows)
    painted: set[tuple[int, int]] = {(GRAY_START_ROW, int(start_col))}
    areas: list[int] = []

    def rec(r_cur: int, c_cur: int) -> bool:
        remaining = target_len - len(areas)
        if remaining == 0:
            has_vertical = any(int(area) == 4 for area in areas)
            has_horizontal = any(int(area) != 4 for area in areas)
            if not (has_vertical and has_horizontal):
                return False
            if len(set(int(area) for area in areas)) < 2:
                return False
            return _count_direction_changes(areas) >= MIN_DIRECTION_CHANGES

        # Even if all remaining moves are horizontal, each consumes 2 rows.
        if int(r_cur) + 2 * remaining > int(height) - 1:
            return False

        opts = list(_AREAS)
        random.shuffle(opts)
        for area in opts:
            cells = _segment_cells(int(area), r_cur, c_cur, step_rows=STEP_ROWS)
            if any(not (0 <= rr < int(height) and 0 <= cc < LEFT_W) for rr, cc in cells):
                continue
            if any((int(rr), int(cc)) in painted for rr, cc in cells):
                continue

            areas.append(int(area))
            for rr, cc in cells:
                painted.add((int(rr), int(cc)))

            r_next, c_next = map(int, cells[-1])
            if rec(r_next, c_next):
                return True

            for rr, cc in cells:
                painted.remove((int(rr), int(cc)))
            areas.pop()
        return False

    if rec(GRAY_START_ROW, int(start_col)):
        return list(areas)
    return None


def _render_grid(n_rows: int, start_col: int, areas: list[int]) -> np.ndarray:
    height = _grid_height(n_rows)
    grid = np.zeros((int(height), GRID_W), dtype=int)
    grid[:, DIVIDER_COL] = DIVIDER_COLOR
    grid[GRAY_START_ROW, DIVIDER_COL + 1 + int(start_col)] = GRAY

    for area, (r0, c0) in zip(areas, _variant_slots(n_rows)):
        grid[int(r0): int(r0) + 3, int(c0): int(c0) + 3] = _TEMPLATES[int(area)] * _AREA_TO_COLOR[int(area)]

    return grid


def _fusion_is_visible(height: int, start_col: int, areas: list[int]) -> bool:
    combo = _trace_polyline(height, start_col, areas, step_rows=STEP_ROWS)
    reverse_only = _safe_trace(height, start_col, areas, step_rows=1)
    big_step_only = _safe_trace(height, start_col, list(reversed(areas)), step_rows=STEP_ROWS)
    legacy = _safe_trace(height, start_col, list(reversed(areas)), step_rows=1)

    for alt in (reverse_only, big_step_only, legacy):
        if alt is not None and np.array_equal(combo, alt):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_rows = random.randint(N_MIN, N_MAX)
    height = _grid_height(n_rows)

    start_cols = list(range(LEFT_W))
    random.shuffle(start_cols)
    for start_col in start_cols:
        areas = _sample_path(n_rows, start_col)
        if areas is None:
            continue
        if not _order_signal_is_visible(areas):
            continue
        if not _fusion_is_visible(height, start_col, areas):
            continue

        inp = _render_grid(n_rows, start_col, areas)
        expected = _trace_polyline(height, start_col, areas, step_rows=STEP_ROWS)
        solved = main(inp)
        if np.array_equal(solved, expected):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            inp = _gen_attempt()
            if inp is not None:
                return inp
        except Exception:
            continue
    raise RuntimeError(f"generate_input: failed after {MAX_ATTEMPTS} attempts")
