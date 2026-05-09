# Combo variant of 136b0064 (index 12)
# combo_spec: D1/2+D9/0
# Name: 单列字形按面积排序

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg


# Design analysis for the fused variant:
# - Concrete behavior shift:
#   Example top-to-bottom glyph areas [7, 4, 6, 5] used to be read by position in the
#   original task, but this combo must ignore the displayed order and decode [4, 5, 6, 7]
#   instead. The generator therefore scrambles the vertical arrangement on purpose.
# - Complexity calibration:
#   Sorting by area collapses the original free-form walk order, so the old generator rule
#   "at least two vertical/horizontal direction flips" is no longer compatible: after sorting,
#   all area-4 vertical steps come before the horizontal groups. Instead we require visible
#   area diversity, a non-sorted displayed order, and a measurable mismatch between sorted
#   decoding and naive top-down decoding.
# - Implicit assumptions that changed:
#   The original solver assumed a 2-column tile grid and column-major order. Those assumptions
#   are removed. The new solver only assumes that every instruction is still a monochrome,
#   8-connected 3x3 component and that the divider/start-pixel conventions remain intact.
# - Placement legality:
#   The generator uses one shared glyph column `c0` per sample. Consecutive glyph top rows
#   differ by at least 4, so the 3x3 bboxes never overlap and never 8-touch. Height is chosen
#   large enough to support irregular gaps and to keep the decoded path inside the output area.
# - Visual observability:
#   The glyph list is a visibly irregular single column, so reading "from top to bottom" is a
#   plausible but wrong hypothesis. Repeated shapes and fixed area->segment bindings make the
#   ascending-area sort inferable from a few examples.


def find_full_height_constant_col(grid: np.ndarray, *, bg: int | None = None) -> tuple[int | None, int | None]:
    """Find a full-height constant non-background column."""
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
AREA_TO_DIR = {4: 0, 5: 1, 6: -1, 7: -1}
AREA_TO_COLOR = {4: 6, 5: 1, 6: 3, 7: 2}
AREA_TO_ROW_ADVANCE = {4: 2, 5: 1, 6: 1, 7: 1}
AREAS = (4, 5, 6, 7)

TEMPLATES = {
    4: np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
    5: np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int),
    6: np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=int),
    7: np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=int),
}

# --- canvas geometry for the fused generator ---
GRID_W = 15
DIVIDER_COL = 7
DIVIDER_COLOR = 4
LEFT_W = 7
GRAY = 5
GRAY_START_ROW = 0
GLYPH_COUNT_OPTIONS = (4, 6)
GLYPH_C0_OPTIONS = (0, 2, 4)
MAX_GRID_H = 30
MAX_ATTEMPTS = 500
MIN_INVERSION_COUNT = 2


def _extract_glyphs(left: np.ndarray, *, bg: int, divider_color: int | None) -> list[dict[str, int]]:
    """Collect all 3x3 monochrome glyphs from the left canvas."""
    ignore = {int(bg), GRAY}
    if divider_color is not None:
        ignore.add(int(divider_color))

    glyphs: list[dict[str, int]] = []
    for comp in find_components(left, bg=bg, connectivity=8, monochromatic=True, return_format="dict"):
        color = int(comp["color"])
        if color in ignore:
            continue
        b = comp["bbox"]
        area = int(comp["area"])
        if bbox_h(b) != 3 or bbox_w(b) != 3 or area not in AREA_TO_LEN:
            continue
        r0, c0, _r1, _c1 = map(int, b)
        glyphs.append({"color": color, "r0": r0, "c0": c0, "area": area})
    return glyphs


def _trace_sequence(
    h: int,
    w: int,
    start_row: int,
    start_col: int,
    glyphs: list[dict[str, int]],
    *,
    bg: int = 0,
) -> np.ndarray:
    """Decode a glyph sequence into the colored output polyline."""
    out = np.full((h, w), int(bg), dtype=int)
    r_cur = int(start_row)
    c_cur = int(start_col)
    out[r_cur, c_cur] = GRAY

    for glyph in glyphs:
        area = int(glyph["area"])
        color = int(glyph["color"])
        seg_len = int(AREA_TO_LEN[area])
        dir_h = int(AREA_TO_DIR[area])

        r_cur += 1
        assert r_cur < h, f"segment exits bottom: r_cur={r_cur}, H={h}"
        out[r_cur, c_cur] = color

        if area == 4:
            end_r = r_cur + (seg_len - 1)
            assert end_r < h, f"vertical segment exits bottom: end_r={end_r}, H={h}"
            for _ in range(seg_len - 1):
                r_cur += 1
                out[r_cur, c_cur] = color
        else:
            end_c = c_cur + dir_h * (seg_len - 1)
            assert 0 <= end_c < w, (
                f"horizontal segment exits canvas: c_cur={c_cur}, dir={dir_h}, seg_len={seg_len}, W={w}"
            )
            for _ in range(seg_len - 1):
                c_cur += dir_h
                out[r_cur, c_cur] = color

    return out


def _trace_areas(h: int, start_col: int, areas: list[int]) -> np.ndarray:
    """Helper for generation-time validation using the fixed area->color binding."""
    glyphs = [{"area": int(area), "color": int(AREA_TO_COLOR[int(area)])} for area in areas]
    return _trace_sequence(h, LEFT_W, GRAY_START_ROW, start_col, glyphs, bg=0)


def _count_inversions(values: list[int]) -> int:
    inv = 0
    for i, a in enumerate(values):
        for b in values[i + 1 :]:
            if int(a) > int(b):
                inv += 1
    return inv


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a polyline from a single vertical glyph list whose displayed order is misleading.

    Input structure:
    - Left of the full-height divider: several colored 3x3 glyphs arranged in one irregular column.
    - Right of the divider: a single gray start pixel.

    Rule:
    - Remove the divider and shift the gray start into the left output canvas.
    - Ignore the glyphs' vertical placement as an ordering signal.
    - Instead, sort all 3x3 glyphs by pixel area from small to large (4, then 5, then 6, then 7).
    - Decode each sorted glyph as the same segment vocabulary as the original task:
      area 4 -> down 2, area 5 -> right 3, area 6 -> left 4, area 7 -> left 2.

    Key insight:
    - The instruction stream is determined by visible shape area, not by where the glyph sits in the column.
      Spatial order is now a distractor; the reliable cue is the glyph's pixel count.
    """
    g = np.array(grid, dtype=int)
    h, _w = g.shape
    bg = int(get_bg(g))

    div_c, div_color = find_full_height_constant_col(g, bg=bg)
    assert div_c is not None

    gray = np.argwhere(g == GRAY)
    out = np.full((h, int(div_c)), bg, dtype=int)
    if gray.size == 0:
        return out

    start_r, start_c = map(int, gray[0])
    start_c -= int(div_c) + 1
    if not (0 <= start_c < int(div_c)):
        return out

    left = g[:, : int(div_c)]
    glyphs = _extract_glyphs(left, bg=bg, divider_color=div_color)
    glyphs.sort(key=lambda d: (int(d["area"]), int(d["r0"]), int(d["c0"])))
    return _trace_sequence(h, int(div_c), start_r, start_c, glyphs, bg=bg)


def _sample_area_multiset(glyph_count: int, start_col: int, h: int) -> tuple[list[int], list[int]] | None:
    """Sample a multiset whose sorted order yields a valid path and whose displayed order will be unsorted."""
    if glyph_count < len(AREAS):
        return None

    for _ in range(200):
        # Keep every instruction type observable in every sample so the fused rule is
        # learnable from a few examples, then add a small number of distinct duplicates.
        area_pool = list(AREAS)
        extra_count = int(glyph_count) - len(AREAS)
        if extra_count > 0:
            area_pool.extend(random.sample(list(AREAS), k=extra_count))
        random.shuffle(area_pool)
        areas_by_position = [int(area) for area in area_pool]

        areas_sorted = sorted(areas_by_position)
        try:
            _trace_areas(h, start_col, areas_sorted)
        except AssertionError:
            continue

        inv = _count_inversions(areas_by_position)
        if inv < MIN_INVERSION_COUNT:
            continue

        try:
            top_down_out = _trace_areas(h, start_col, areas_by_position)
        except AssertionError:
            top_down_out = None
        sorted_out = _trace_areas(h, start_col, areas_sorted)
        if top_down_out is not None and np.array_equal(top_down_out, sorted_out):
            continue
        return areas_by_position, areas_sorted
    return None


def _sample_layout(glyph_count: int) -> tuple[list[int], int] | None:
    """
    Sample irregular top rows for one visible glyph column.

    Minimal packing for `glyph_count` many 3x3 glyphs with one blank row between them is `4*count - 1`.
    Extra slack is injected into the inter-glyph gaps so the column is visibly not aligned to a regular grid.
    """
    base_h = 4 * int(glyph_count) - 1
    if base_h >= MAX_GRID_H:
        return None

    for _ in range(100):
        total_extra = random.randint(1, int(MAX_GRID_H - base_h))
        extras = [0] * max(int(glyph_count) - 1, 0)
        bottom_pad = 0

        for _ in range(total_extra):
            pick = random.randrange(len(extras) + 1)
            if pick < len(extras):
                extras[pick] += 1
            else:
                bottom_pad += 1

        if extras and len(set(extras)) < 2:
            continue

        rows = [0]
        for extra in extras:
            rows.append(int(rows[-1] + 4 + extra))
        h = int(rows[-1] + 3 + bottom_pad)
        if h <= MAX_GRID_H:
            return rows, h
    return None


def _render_grid(h: int, start_col: int, glyph_c0: int, rows: list[int], areas_by_position: list[int]) -> np.ndarray:
    """Paint the fused-variant input grid."""
    grid = np.zeros((h, GRID_W), dtype=int)
    grid[:, DIVIDER_COL] = DIVIDER_COLOR
    grid[GRAY_START_ROW, DIVIDER_COL + 1 + start_col] = GRAY

    for r0, area in zip(rows, areas_by_position):
        color = int(AREA_TO_COLOR[int(area)])
        patch = TEMPLATES[int(area)] * color
        grid[int(r0) : int(r0) + 3, int(glyph_c0) : int(glyph_c0) + 3] = patch
    return grid


def _gen_attempt() -> np.ndarray | None:
    glyph_count = int(random.choice(GLYPH_COUNT_OPTIONS))
    layout = _sample_layout(glyph_count)
    if layout is None:
        return None
    rows, h = layout

    start_col = random.randint(0, LEFT_W - 1)
    sampled = _sample_area_multiset(glyph_count, start_col, h)
    if sampled is None:
        return None
    areas_by_position, _areas_sorted = sampled

    glyph_c0 = int(random.choice(GLYPH_C0_OPTIONS))
    grid = _render_grid(h, start_col, glyph_c0, rows, areas_by_position)

    # Oracle and anti-degeneracy checks.
    glyphs_top_down = _extract_glyphs(grid[:, :DIVIDER_COL], bg=0, divider_color=DIVIDER_COLOR)
    glyphs_top_down.sort(key=lambda d: (int(d["r0"]), int(d["c0"])))
    top_down_areas = [int(g["area"]) for g in glyphs_top_down]
    sorted_areas = sorted(top_down_areas)
    if _count_inversions(top_down_areas) < MIN_INVERSION_COUNT:
        return None
    if top_down_areas == sorted_areas:
        return None

    sorted_out = _trace_areas(h, start_col, sorted_areas)
    top_down_out = _trace_areas(h, start_col, top_down_areas)
    if np.array_equal(sorted_out, top_down_out):
        return None

    oracle = main(grid)
    if oracle.shape != (h, LEFT_W):
        return None
    if not np.array_equal(oracle, sorted_out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            grid = _gen_attempt()
            if grid is not None:
                return grid
        except Exception:
            continue
    raise RuntimeError(f"generate_input: failed after {MAX_ATTEMPTS} attempts")
