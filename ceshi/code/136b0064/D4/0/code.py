# Variant of 136b0064
# Dimension: D4 — 目标结构——折线的构建方式 (dst)
# Change: 从灰色起始点出发，每处理一个字形先向下移一行，再按指令画水平或垂直线段，形成连续折线... → 折线从底部向上生长

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg


# Design analysis for the D4 variant
# - Concrete behavior: the left half still contains the same fixed 3x3 glyph grammar, read in the
#   same column-major order. The visible change is entirely on the traced polyline side: the lone
#   gray start marker now sits on the bottom row of the right half, and after shifting into the
#   output canvas the trace grows upward. For each glyph we first move one row up, color that cell,
#   and then extend the segment encoded by the glyph. Horizontal instructions keep their old
#   left/right meaning; the unique vertical instruction flips from "down 2" to "up 2" so the whole
#   path has a consistent bottom-to-top reference frame.
# - Small execution trace: if the shifted gray start is at (10, 3) and the first two glyph areas are
#   [5, 4], the old task would draw area 5 at row 11 and then continue downward from there. The
#   variant instead goes to row 9, draws a rightward length-3 segment there, then goes to row 8 and
#   extends the area-4 segment upward to row 7. The changed rows are therefore not cosmetic; every
#   row update and the vertical segment polarity are inverted.
# - Complexity calibration: the glyph inventory, colors, divider, tile layout, and column-major
#   decoding all stay fixed. This keeps the innovation isolated to D4 while preserving enough
#   repeated evidence per sample for humans to infer the rule. The generator still enforces at least
#   two vertical/horizontal alternations so the output is not a trivial straight run.
# - Implicit-assumption audit for main():
#   1. The original `r_cur += 1` assumed the trace could only descend; it must become `r_cur -= 1`.
#   2. The original bottom-bound asserts (`r_cur < H`, `end_r < H`) depended on that descent and
#      must be replaced by top-bound asserts (`r_cur >= 0`, `end_r >= 0`).
#   3. The original area-4 loop incremented rows, which would silently reintroduce downward motion in
#      this variant; it must decrement rows instead.
#   4. Horizontal bounds and glyph ordering remain valid because D4 does not alter left/right motion
#      or the instruction readout scheme.
# - Placement legality: the generator now simulates the walk from `(H-1, start_col)` with per-area
#   deltas `{4:(-2,0), 5:(-1,+2), 6:(-1,-3), 7:(-1,-1)}` and only accepts choices that keep every
#   intermediate endpoint inside the 7-column left canvas and on/above row 0.
# - Visual observability and cleanliness: keeping the gray pixel on the bottom row in every sample
#   makes the inversion immediately visible. The glyph bank remains isolated to the left of the
#   divider, while the right side still contains only the start marker, so there is no new clutter.
# - Color reasoning: unchanged from the source task. Colors remain shape-bound and are never used for
#   hidden ordering or arithmetic.


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


AREA_TO_LEN = {4: 2, 5: 3, 6: 4, 7: 2}
AREA_TO_HDIR = {5: 1, 6: -1, 7: -1}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode 3x3 glyph instructions and draw a polyline that rises from a bottom gray start.

    Input structure:
    - Left of a full-height vertical divider: 3x3 colored glyphs on the background.
    - Right of divider: one gray (5) start pixel positioned on the bottom row.

    Rule:
    - Remove the divider and shift the gray start into the left canvas.
    - Read glyphs in column-major order (left column top→bottom, then right column top→bottom).
    - Each glyph's pixel area defines a segment: area 4 → up 2, area 5 → right 3, area 6 → left 4,
      area 7 → left 2.
    - Build one continuous trace by moving up one row before each segment, then drawing the segment.

    Key insight: The glyph shapes themselves are still the instruction code, but the construction
    frame has inverted: identical glyphs now generate the same horizontal moves while the path grows
    upward from the bottom instead of downward from the top.
    """
    g = np.array(grid, dtype=int)
    h, _w = g.shape
    bg = int(get_bg(g))

    div_c, div_color = find_full_height_constant_col(g, bg=bg)
    assert div_c is not None

    out = np.full((h, int(div_c)), bg, dtype=int)

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
            glyphs.append({"color": col, "r0": r0, "c0": c0, "area": int(comp["area"])})

    glyphs.sort(key=lambda d: (int(d["c0"]), int(d["r0"])))

    for glyph in glyphs:
        color = int(glyph["color"])
        area = int(glyph["area"])
        seg_len = int(AREA_TO_LEN.get(area, 2))

        r_cur = int(r_cur - 1)
        assert r_cur >= 0, f"segment exits top: r_cur={r_cur}"
        out[r_cur, int(c_cur)] = color

        if area == 4:
            end_r = int(r_cur - (seg_len - 1))
            assert end_r >= 0, f"vertical segment exits top: end_r={end_r}"
            for _ in range(seg_len - 1):
                r_cur = int(r_cur - 1)
                out[r_cur, int(c_cur)] = color
        else:
            dir_h = int(AREA_TO_HDIR.get(area, 1))
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

# --- canvas geometry ---
N_MIN, N_MAX = 2, 5
GRID_W = 15
DIVIDER_COL = 7
DIVIDER_COLOR = 4
LEFT_W = 7
TILE_COLS = (0, 4)
TILE_ROW_STRIDE = 4
GRAY = 5

# --- shape grammar ---
_TEMPLATES = {
    4: np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
    5: np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int),
    6: np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=int),
    7: np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=int),
}
_AREA_TO_COLOR = {4: 6, 5: 1, 6: 3, 7: 2}
_AREA_DELTA = {4: (-2, 0), 5: (-1, 2), 6: (-1, -3), 7: (-1, -1)}
_AREAS = (4, 5, 6, 7)

# --- generation constraints ---
MIN_DIRECTION_CHANGES = 2
MAX_ATTEMPTS = 300


def _sample_path(n: int, start_col: int) -> list[int] | None:
    """Sample a valid upward-growing instruction list of length 2*n."""
    h = 4 * n - 1
    r, c = h - 1, start_col
    areas: list[int] = []
    for _ in range(2 * n):
        opts = [
            area
            for area in _AREAS
            if 0 <= r + _AREA_DELTA[area][0] < h and 0 <= c + _AREA_DELTA[area][1] < LEFT_W
        ]
        if not opts:
            return None
        area = random.choice(opts)
        areas.append(area)
        dr, dc = _AREA_DELTA[area]
        r += dr
        c += dc

    types = [area == 4 for area in areas]
    changes = sum(1 for i in range(1, len(types)) if types[i] != types[i - 1])
    if changes < MIN_DIRECTION_CHANGES:
        return None
    return areas


def _render_grid(n: int, start_col: int, areas: list[int]) -> np.ndarray:
    """Paint divider, bottom gray start, and the tiled glyph bank into the input grid."""
    h = 4 * n - 1
    grid = np.zeros((h, GRID_W), dtype=int)
    grid[:, DIVIDER_COL] = DIVIDER_COLOR
    grid[h - 1, DIVIDER_COL + 1 + start_col] = GRAY

    for i, area in enumerate(areas):
        tile_col = TILE_COLS[i // n]
        r0 = TILE_ROW_STRIDE * (i % n)
        grid[r0 : r0 + 3, tile_col : tile_col + 3] = _TEMPLATES[area] * _AREA_TO_COLOR[area]
    return grid


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(N_MIN, N_MAX)
    start_col = random.randint(0, LEFT_W - 1)
    areas = _sample_path(n, start_col)
    if areas is None:
        return None
    return _render_grid(n, start_col, areas)


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            inp = _gen_attempt()
            if inp is None:
                continue
            out = main(inp)
            if out.shape != (inp.shape[0], LEFT_W):
                continue
            gray = np.argwhere(inp == GRAY)
            if gray.shape != (1, 2) or int(gray[0, 0]) != inp.shape[0] - 1:
                continue
            return inp
        except Exception:
            continue
    raise RuntimeError(f"generate_input: failed after {MAX_ATTEMPTS} attempts")
