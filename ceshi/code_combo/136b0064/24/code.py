# Combo variant of 136b0064 (index 24)
# combo_spec: D1/2+D4/4
# Name: 面积排序的分层堆叠输出

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg


# Design analysis for the combo:
# 1. Concrete trace.
#    Example: start_col=3 and tile-order areas are [6, 4, 5, 4].
#    - Original task: reads them column-major as [6,4,5,4], mutates one shared (r_cur,c_cur),
#      and the later segments start from the previous segment endpoint.
#    - This combo: sorts by area first -> [4,4,5,6], keeps one vertical stack cursor, but
#      every segment starts from the same anchor column 3 instead of the previous endpoint.
#      The output becomes layered bands: two vertical 2-cells, then a rightward length-3 row,
#      then a leftward length-4 row.
#    Difference: D1 changes which glyph rises to earlier layers; D4 changes geometry from one
#    chained polyline to separated layers sharing a common anchor.
# 2. Complexity calibration.
#    The generator always includes all four area branches {4,5,6,7}. That keeps the full decoding
#    grammar observable while still producing a tidy stack: one vertical len-2 layer, one short-left
#    len-2 layer, one right len-3 layer, and one left len-4 layer, with extra repeats added on top.
# 3. Implicit assumptions broken by the combo.
#    - Original `glyphs.sort((c0,r0))` is invalid: order now depends on area.
#    - Original horizontal code mutates `c_cur` across glyphs; invalid because segments are
#      independent layers now.
#    - Original row advance meant "next segment attaches below the last endpoint"; now it means
#      "next layer begins below the previous layer's occupied rows".
# 4. Placement legality.
#    - Generator fixes start_col to 3 or 4 so both right-3 and left-4 segments always fit.
#    - Every sample has at least one copy of each area 4/5/6/7, so all decoding branches appear.
#    - Count(area 4) <= 2*N-2, so total stacked height 1 + num_glyphs + count(area 4) stays
#      within H = 4*N-1.
# 5. Visual observability.
#    - Tile order is intentionally not already area-sorted, so D1 is visible from input/output pairs.
#    - Output is rejected if it matches either single-change ablation:
#      column-major stacking (only D4) or area-sorted polyline (only D1).
# 6. Visual cleanliness.
#    Fixed anchor column plus disjoint layers avoids crossings and overdraw; each band is legible.
# 7. New solver assumptions and generator guarantees.
#    - Solver assumes exactly one gray start on the right and 3x3 monochrome glyph components on the left.
#    - Solver assumes the stacked layers fit in height and each horizontal span fits in width.
#      Generator constructs only legal starts and area counts that satisfy these constraints.


AREA_TO_LEN = {4: 2, 5: 3, 6: 4, 7: 2}
AREA_TO_DIR = {5: 1, 6: -1, 7: -1}


def find_full_height_constant_col(grid: np.ndarray, *, bg: int | None = None) -> tuple[int | None, int | None]:
    """
    Local helper: find a column whose entries are all identical and not bg.
    Returns (col_idx, color) or (None, None).
    """
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    _H, W = grid.shape
    for c in range(int(W)):
        col = grid[:, int(c)]
        if (col == col[0]).all() and int(col[0]) != bg:
            return int(c), int(col[0])
    return None, None


def _extract_scene(grid: np.ndarray) -> tuple[int, int, int, list[dict[str, int]]]:
    g = np.array(grid, dtype=int)
    H, _W = g.shape
    bg = int(get_bg(g))

    div_c, div_color = find_full_height_constant_col(g, bg=bg)
    assert div_c is not None

    gray = np.argwhere(g == 5)
    assert gray.size != 0, "missing gray start"
    r_gray, c_gray = map(int, gray[0])
    assert int(r_gray) == 0, "gray start is expected on the top row"
    start_col = int(c_gray - (int(div_c) + 1))
    assert 0 <= start_col < int(div_c), "shifted gray start must land in left canvas"

    left = g[:, : int(div_c)]
    ignore = {int(bg), 5}
    if div_color is not None:
        ignore.add(int(div_color))

    glyphs: list[dict[str, int]] = []
    for comp in find_components(left, bg=bg, connectivity=8, monochromatic=True, return_format="dict"):
        color = int(comp["color"])
        if color in ignore:
            continue
        b = comp["bbox"]
        if bbox_h(b) != 3 or bbox_w(b) != 3:
            continue
        r0, c0, _r1, _c1 = map(int, b)
        glyphs.append(
            {
                "color": color,
                "r0": int(r0),
                "c0": int(c0),
                "area": int(comp["area"]),
            }
        )

    assert glyphs, "no glyphs found"
    return int(H), int(div_c), int(start_col), glyphs


def _draw_segment_row(out: np.ndarray, row: int, start_col: int, color: int, area: int) -> None:
    W = int(out.shape[1])
    seg_len = int(AREA_TO_LEN[area])
    if area == 4:
        assert row + seg_len - 1 < int(out.shape[0]), "vertical layer exits canvas"
        for k in range(seg_len):
            out[int(row + k), int(start_col)] = int(color)
        return

    dir_h = int(AREA_TO_DIR[area])
    end_c = int(start_col + dir_h * (seg_len - 1))
    assert 0 <= end_c < W, "horizontal layer exits canvas"
    for k in range(seg_len):
        out[int(row), int(start_col + dir_h * k)] = int(color)


def _solve_with_order(grid: np.ndarray, *, area_sorted: bool, stacked: bool) -> np.ndarray:
    H, left_w, start_col, glyphs = _extract_scene(grid)
    out = np.zeros((H, left_w), dtype=int)
    out[0, int(start_col)] = 5

    if area_sorted:
        ordered = sorted(glyphs, key=lambda d: (int(d["area"]), int(d["c0"]), int(d["r0"])))
    else:
        ordered = sorted(glyphs, key=lambda d: (int(d["c0"]), int(d["r0"])))

    if stacked:
        stack_row = 0
        for glyph in ordered:
            row = int(stack_row + 1)
            assert row < H, "stack layer exits bottom"
            _draw_segment_row(out, row, start_col, int(glyph["color"]), int(glyph["area"]))
            stack_row = int(row + 1) if int(glyph["area"]) == 4 else int(row)
        return out

    r_cur = 0
    c_cur = int(start_col)
    for glyph in ordered:
        area = int(glyph["area"])
        color = int(glyph["color"])
        row = int(r_cur + 1)
        assert row < H, "polyline step exits bottom"
        if area == 4:
            assert row + 1 < H, "vertical polyline segment exits bottom"
            out[row, c_cur] = color
            out[row + 1, c_cur] = color
            r_cur = int(row + 1)
            continue

        seg_len = int(AREA_TO_LEN[area])
        dir_h = int(AREA_TO_DIR[area])
        end_c = int(c_cur + dir_h * (seg_len - 1))
        assert 0 <= end_c < left_w, "polyline segment exits side"
        for k in range(seg_len):
            out[row, int(c_cur + dir_h * k)] = color
        r_cur = int(row)
        c_cur = int(end_c)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the 3x3 glyphs into colored segments, but stack them as independent layers after area sorting.

    Input structure:
    - Left of a full-height divider: monochrome 3x3 glyphs placed in fixed tile slots on a black background.
    - Right of the divider: exactly one gray start pixel whose shifted column defines the common anchor.

    Rule:
    - Remove the divider conceptually and shift the gray start into the left 7-column canvas.
    - Collect all 3x3 glyphs, compute each glyph's pixel area, and process them in ascending area order.
    - Each glyph still decodes to the same segment type as the source task:
      area 4 -> vertical length 2, area 5 -> right length 3, area 6 -> left length 4, area 7 -> left length 2.
    - Instead of chaining segments endpoint-to-endpoint, start every decoded segment from the same anchor column
      and place the segments in separate layers from top to bottom, using one row for horizontal segments and
      two rows for the vertical segment.

    Key insight:
    A glyph's area still determines the segment itself, but the glyph's tile position no longer decides when
    it is drawn. Area order decides the stack order, and the output geometry is a layered pile rather than
    one continuous walk.
    """
    return _solve_with_order(grid, area_sorted=True, stacked=True)


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
VALID_START_COLS = (3, 4)

# --- glyph templates and fixed color binding ---
TEMPLATES = {
    4: np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
    5: np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int),
    6: np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=int),
    7: np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=int),
}
AREA_TO_COLOR = {4: 6, 5: 1, 6: 3, 7: 2}
MAX_ATTEMPTS = 400


def _sample_area_multiset(N: int) -> list[int]:
    total = 2 * N
    counts = {4: 1, 5: 1, 6: 1, 7: 1}
    extra = total - 4
    while extra > 0:
        area = random.choice((4, 5, 6, 7))
        if area == 4 and counts[4] >= total - 2:
            continue
        counts[area] += 1
        extra -= 1
    assert counts[4] <= total - 2
    return [area for area, count in counts.items() for _ in range(count)]


def _shuffle_not_sorted(areas: list[int]) -> list[int] | None:
    sorted_areas = sorted(areas)
    for _ in range(80):
        perm = list(areas)
        random.shuffle(perm)
        if perm != sorted_areas:
            return perm
    return None


def _render_grid(N: int, start_col: int, tile_order_areas: list[int]) -> np.ndarray:
    H = 4 * N - 1
    grid = np.zeros((H, GRID_W), dtype=int)
    grid[:, DIVIDER_COL] = DIVIDER_COLOR
    grid[GRAY_START_ROW, DIVIDER_COL + 1 + start_col] = GRAY

    for i, area in enumerate(tile_order_areas):
        tile_col = TILE_COLS[i // N]
        tile_row = i % N
        r0 = TILE_ROW_STRIDE * tile_row
        grid[r0 : r0 + 3, tile_col : tile_col + 3] = TEMPLATES[int(area)] * AREA_TO_COLOR[int(area)]
    return grid


def _fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    combo = main(grid)
    only_d1 = _solve_with_order(grid, area_sorted=True, stacked=False)
    only_d4 = _solve_with_order(grid, area_sorted=False, stacked=True)
    original = _solve_with_order(grid, area_sorted=False, stacked=False)
    _H, _left_w, _start_col, glyphs = _extract_scene(grid)
    tile_order = [int(g["area"]) for g in sorted(glyphs, key=lambda d: (int(d["c0"]), int(d["r0"])))]
    area_order = [int(g["area"]) for g in sorted(glyphs, key=lambda d: (int(d["area"]), int(d["c0"]), int(d["r0"])))]
    lengths = [int(AREA_TO_LEN[a]) for a in area_order]
    return {
        "tile_area_inversions": int(sum(a > b for i, a in enumerate(tile_order) for b in tile_order[i + 1 :])),
        "distinct_lengths": int(len(set(lengths))),
        "distinct_areas": int(len(set(area_order))),
        "combo_vs_d1_pixels": int(np.count_nonzero(combo != only_d1)),
        "combo_vs_d4_pixels": int(np.count_nonzero(combo != only_d4)),
        "combo_vs_original_pixels": int(np.count_nonzero(combo != original)),
    }


def _gen_attempt() -> np.ndarray | None:
    N = random.randint(N_MIN, N_MAX)
    start_col = random.choice(VALID_START_COLS)
    areas = _sample_area_multiset(N)
    tile_order_areas = _shuffle_not_sorted(areas)
    if tile_order_areas is None:
        return None

    grid = _render_grid(N, start_col, tile_order_areas)
    metrics = _fusion_metrics(grid)
    if metrics["tile_area_inversions"] <= 0:
        return None
    if metrics["distinct_areas"] < 4:
        return None
    if metrics["distinct_lengths"] < 3:
        return None
    if metrics["combo_vs_d1_pixels"] <= 0 or metrics["combo_vs_d4_pixels"] <= 0:
        return None
    if metrics["combo_vs_original_pixels"] <= 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            grid = _gen_attempt()
            if grid is None:
                continue
            out = main(grid)
            if out.shape != (grid.shape[0], LEFT_W):
                continue
            return grid
        except Exception:
            continue
    raise RuntimeError("generate_input: failed after max attempts for combo variant 136b0064/24")
