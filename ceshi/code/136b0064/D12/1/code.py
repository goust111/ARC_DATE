# Variant of 136b0064
# Dimension: D12 — 垂直线段的处理——area 4 的特殊行为 (transform)
# Change: area 4 字形（plus形状）生成垂直向下2格的线段，不改变列坐标；其他字形生成水平线段并改变列坐标... → 引入斜线段（45度对角线）

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg

# Design analysis for the D12 variant
# a) Concrete behavior:
#    The input layout stays the same: 3x3 glyphs on the left, divider in col 7, one gray start
#    on the right. The changed part is only the decoding of the plus(area=4) glyph. Instead of
#    drawing a short vertical stroke, area 4 now draws a 45-degree descending diagonal of 3 cells.
#    To make both diagonal directions visible without adding new glyph shapes, the existing tile
#    columns provide the cue: an area-4 glyph from the left tile column (c0=0) draws down-right,
#    while one from the right tile column (c0=4) draws down-left.
#    Example trace from endpoint (0,3):
#      original area 4: anchor at (1,3), then (2,3) -> total delta (+2, 0)
#      variant  area 4 in left tile column: (1,3),(2,4),(3,5) -> total delta (+3, +2)
#      variant  area 4 in right tile column: (1,3),(2,2),(3,1) -> total delta (+3, -2)
#    Areas 5/6/7 keep the original horizontal meanings, so the task still reads as a polyline
#    decoder but now with diagonal turns mixed into the path.
# b) Complexity calibration:
#    Only one glyph family changes, so the task is still small enough to infer. The rule is not
#    reduced to a trivial recoloring: it changes geometry, row budget, and final endpoint. Using
#    tile-column position as the left/right diagonal cue keeps the number of cases low: horizontal,
#    diagonal-right, diagonal-left.
# c) Review of old assumptions:
#    - Old `if area == 4` assumed column never changes. No longer true; solver must update both
#      row and column for area 4.
#    - Old row-budget logic treated area 4 as consuming 2 rows total. Now it consumes 3.
#    - Old generator stored a single `_AREA_DELTA[4]`. That no longer works because the delta now
#      depends on which tile column the glyph belongs to.
#    - Overwriting was mostly harmless before because paths were orthogonal and short. Diagonals can
#      cross earlier strokes more easily, so the generator now rejects self-overlapping traces for
#      cleanliness.
# d) Placement legality:
#    The left canvas is still width 7, so diagonal area-4 steps must stay within 0..6 in every
#    intermediate cell. The generator simulates each segment cell-by-cell and accepts only moves
#    that remain in-bounds and do not revisit an occupied output cell. Glyph placement on the input
#    side is unchanged: fixed 3x3 tiles in the original slots.
# e) Visual observability:
#    Every generated instance contains at least one area-4 glyph, so the new diagonal behavior is
#    actually exercised. The direction cue is visible in the input itself because the same plus
#    glyph appears in one of the two tile columns; the output then shows a matching down-right or
#    down-left diagonal branch.
# f) Visual neatness:
#    The generator forbids output-cell reuse, which avoids diagonal strokes scribbling over older
#    horizontal ones. This keeps the trace readable and makes the diagonal segments stand out.
# g) Color reasoning:
#    Color bindings remain exactly the original ones. No hidden color-id ordering or new color map
#    is introduced; only geometry changes.
# Additional solver assumptions after the change:
#    - The left half still uses the original two tile columns separated by a blank column; the
#      solver distinguishes diagonal direction by comparing the glyph's c0 to the canvas midpoint.
#    - There is still exactly one gray start pixel and exactly one full-height divider column.


def find_full_height_constant_col(grid: np.ndarray, *, bg: int | None = None) -> tuple[int | None, int | None]:
    """
    Local helper (only used by this solver): find a column where all values are identical and != bg.
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


def _diag_dc_from_glyph_col(glyph_c0: int, div_c: int) -> int:
    """Left tile column => down-right diagonal, right tile column => down-left diagonal."""
    return 1 if int(glyph_c0) < int(div_c // 2) else -1


def _trace_segment_cells(
    r_cur: int,
    c_cur: int,
    area: int,
    *,
    diag_dc: int | None = None,
) -> list[tuple[int, int]]:
    """Return the output cells painted by one glyph, including the mandatory step-down anchor."""
    rr = int(r_cur + 1)
    cc = int(c_cur)
    cells = [(int(rr), int(cc))]

    if int(area) == 4:
        assert diag_dc in (-1, 1)
        for _ in range(2):
            rr = int(rr + 1)
            cc = int(cc + diag_dc)
            cells.append((int(rr), int(cc)))
        return cells

    area_to_len = {5: 3, 6: 4, 7: 2}
    area_to_dir = {5: 1, 6: -1, 7: -1}
    seg_len = int(area_to_len.get(int(area), 2))
    dir_h = int(area_to_dir.get(int(area), 1))
    for _ in range(seg_len - 1):
        cc = int(cc + dir_h)
        cells.append((int(rr), int(cc)))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode tiled glyph instructions into a colored polyline with horizontal and diagonal segments.

    Input structure:
    - Left of a full-height divider: monochrome 3x3 glyphs placed in two fixed tile columns.
    - Right of the divider: one gray (5) pixel marking the polyline start after the divider is removed.

    Rule:
    - Drop the divider and shift the gray start into the 7-column left canvas.
    - Read glyphs in column-major tile order (top-to-bottom in the left tile column, then the right one).
    - For each glyph, first step down one cell, then draw a segment in the glyph's color:
      area 5 -> right 3, area 6 -> left 4, area 7 -> left 2, area 4 -> diagonal 3.
    - The area-4 diagonal slopes down-right when the glyph comes from the left tile column and
      down-left when it comes from the right tile column.

    Key insight: The 3x3 glyph still names the segment type, but the plus glyph now uses its tile-column
    position as an extra visible cue to choose between the two 45-degree diagonal directions.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
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
            glyphs.append({"color": int(col), "r0": int(r0), "c0": int(c0), "area": int(comp["area"])})

    glyphs.sort(key=lambda d: (int(d["c0"]), int(d["r0"])))

    for glyph in glyphs:
        color = int(glyph["color"])
        area = int(glyph["area"])
        diag_dc = _diag_dc_from_glyph_col(int(glyph["c0"]), int(div_c)) if area == 4 else None
        cells = _trace_segment_cells(int(r_cur), int(c_cur), int(area), diag_dc=diag_dc)

        for rr, cc in cells:
            assert 0 <= int(rr) < H, f"segment exits bottom: rr={rr}, H={H}"
            assert 0 <= int(cc) < int(div_c), f"segment exits canvas: cc={cc}, div_c={div_c}"
            out[int(rr), int(cc)] = color

        r_cur, c_cur = map(int, cells[-1])

    return out


# --------------- generator ---------------

# --- canvas geometry ---
N_MIN, N_MAX = 2, 5
GRID_W = 15
DIVIDER_COL = 7
DIVIDER_COLOR = 4
LEFT_W = 7
TILE_COLS = (0, 4)
TILE_ROW_STRIDE = 4
GRAY = 5
GRAY_START_ROW = 0

# --- shape grammar ---
_TEMPLATES = {
    4: np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]], dtype=int),  # plus    -> diagonal, color 6
    5: np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int),  # diamond -> right 3, color 1
    6: np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=int),  # cross   -> left 4,  color 3
    7: np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=int),  # U       -> left 2,  color 2
}
_AREA_TO_COLOR = {4: 6, 5: 1, 6: 3, 7: 2}
_AREAS = (4, 5, 6, 7)

# --- generation constraints ---
MIN_DIRECTION_CHANGES = 2
SECOND_DIAG_PROB = 0.65
MAX_ATTEMPTS = 600


def _diag_dc_from_index(idx: int, n_rows: int) -> int:
    """Instruction indices 0..N-1 come from the left tile column; N..2N-1 from the right."""
    return 1 if int(idx) < int(n_rows) else -1


def _segment_cells_from_index(r_cur: int, c_cur: int, area: int, idx: int, n_rows: int) -> list[tuple[int, int]]:
    diag_dc = _diag_dc_from_index(int(idx), int(n_rows)) if int(area) == 4 else None
    return _trace_segment_cells(int(r_cur), int(c_cur), int(area), diag_dc=diag_dc)


def _choose_mandatory_diag_indices(n_rows: int) -> set[int]:
    """Force at least one area-4 glyph, and often one in each tile column."""
    first_half = list(range(0, int(n_rows)))
    second_half = list(range(int(n_rows), 2 * int(n_rows)))

    idxs = {random.choice(first_half if random.random() < 0.5 else second_half)}
    if int(n_rows) >= 3 and random.random() < SECOND_DIAG_PROB:
        if min(idxs) < int(n_rows):
            idxs.add(random.choice(second_half))
        else:
            idxs.add(random.choice(first_half))
    return idxs


def _sample_path(n_rows: int, start_col: int) -> list[int] | None:
    """
    Random-walk a sequence of 2*N glyph instructions that stays inside the output canvas.

    The walk is simulated with the variant rule itself, cell-by-cell, so diagonal area-4 moves
    are checked for intermediate bounds and for self-overlap.
    """
    H = 4 * int(n_rows) - 1
    r, c = GRAY_START_ROW, int(start_col)
    occupied = {(int(r), int(c))}
    areas: list[int] = []
    mandatory_diag = _choose_mandatory_diag_indices(int(n_rows))

    for idx in range(2 * int(n_rows)):
        allowed = [4] if idx in mandatory_diag else list(_AREAS)
        opts: list[tuple[int, list[tuple[int, int]]]] = []

        for area in allowed:
            cells = _segment_cells_from_index(int(r), int(c), int(area), int(idx), int(n_rows))
            if any(not (0 <= int(rr) < H and 0 <= int(cc) < LEFT_W) for rr, cc in cells):
                continue
            if any((int(rr), int(cc)) in occupied for rr, cc in cells):
                continue
            opts.append((int(area), cells))

        if not opts:
            return None

        area, cells = random.choice(opts)
        areas.append(int(area))
        for rr, cc in cells:
            occupied.add((int(rr), int(cc)))
        r, c = map(int, cells[-1])

    types = ["diag" if int(a) == 4 else "h" for a in areas]
    changes = sum(1 for i in range(1, len(types)) if types[i] != types[i - 1])
    if changes < MIN_DIRECTION_CHANGES:
        return None
    if 4 not in areas:
        return None
    return areas


def _render_grid(n_rows: int, start_col: int, areas: list[int]) -> np.ndarray:
    H = 4 * int(n_rows) - 1
    grid = np.zeros((H, GRID_W), dtype=int)
    grid[:, DIVIDER_COL] = DIVIDER_COLOR
    grid[GRAY_START_ROW, DIVIDER_COL + 1 + int(start_col)] = GRAY

    for i, area in enumerate(areas):
        tile_col = TILE_COLS[i // int(n_rows)]
        r0 = TILE_ROW_STRIDE * (i % int(n_rows))
        grid[r0 : r0 + 3, tile_col : tile_col + 3] = _TEMPLATES[int(area)] * _AREA_TO_COLOR[int(area)]
    return grid


def _gen_attempt() -> np.ndarray | None:
    n_rows = random.randint(N_MIN, N_MAX)
    start_col = random.randint(0, LEFT_W - 1)
    areas = _sample_path(int(n_rows), int(start_col))
    if areas is None:
        return None
    return _render_grid(int(n_rows), int(start_col), areas)


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
