# Combo variant of 136b0064 (index 9)
# combo_spec: D1/4+D7/0
# Name: 从底部出发按距离排序处理
#
# Design analysis for the fused variant
# a) Concrete behavior trace:
#    - Original task: the gray start sits on the top row, glyphs are read in fixed
#      column-major order, and each instruction begins by stepping one row downward.
#    - Fused task: the gray start is moved to the bottom row, so the path must grow
#      upward instead. Glyph order is no longer the fixed tile scan; it is sorted by
#      Manhattan distance from each 3x3 glyph center to the mapped start point.
#    - Small example with N=3, H=11, start_col=2 (mapped left-canvas column):
#      centers are (1,1),(1,5),(5,1),(5,5),(9,1),(9,5). Distances to (10,2) are
#      10,12,6,8,2,4. Original order was TL column top->bottom then TR column
#      top->bottom; new order is bottom-left, bottom-right, middle-left,
#      middle-right, top-left, top-right.
# b) Complexity calibration:
#    - The variant keeps the original four shape-to-instruction bindings so the
#      symbolic load stays stable.
#    - Only two control changes are introduced: start row inversion and distance
#      ordering. To avoid ambiguous ties that would weaken the visual cue, the
#      generator uses start columns 2 or 4, which make all glyph-center distances
#      distinct on the fixed 2-column tile lattice.
# c) Implicit assumptions audited from the original main():
#    - `r_cur += 1` assumed the path always grows downward; this must become
#      `r_cur -= 1`.
#    - Vertical area-4 segments assumed "down 2"; they must become "up 2".
#    - `glyphs.sort(key=(c0,r0))` assumed the visible execution order equals tile
#      scan order; it now must depend on geometry relative to the gray start.
#    - The divider-removal and glyph detection logic still holds because the left
#      layout, colors, and 3x3 tile geometry are unchanged.
# d) Placement legality:
#    - The generator is path-first. It samples an execution sequence that always
#      stays inside the 7-column left canvas while moving upward from the bottom.
#    - Glyphs are still stamped into the non-overlapping 3x3 tile slots at
#      c0 in {0,4} and r0 in {0,4,8,...}; this preserves clean component recovery.
# e) Visual observability:
#    - Across examples, the gray start is visibly on the bottom row.
#    - The closest glyphs are visibly the lowest ones, so input/output pairs reveal
#      that execution starts near the bottom and progresses toward farther glyphs.
# f) Visual cleanliness:
#    - No extra markers are added; only the start location and processing order
#      change. The divider and tile spacing remain unchanged, so objects stay
#      separated and easy to parse.
# g) Inferability of the change:
#    - No hidden color-based ordering is used. The only ordering signal is geometric
#      distance from the visible gray start to the visible glyph centers, which is
#      human-observable from a few examples.

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg


def find_full_height_constant_col(grid: np.ndarray, *, bg: int | None = None) -> tuple[int | None, int | None]:
    """
    Local helper: find a column where every cell has the same non-background color.
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


# --- canvas geometry ---
N_MIN, N_MAX = 2, 5
GRID_W = 15
DIVIDER_COL = 7
DIVIDER_COLOR = 4
LEFT_W = 7
TILE_COLS = (0, 4)
TILE_ROW_STRIDE = 4
GRAY = 5
START_COL_CHOICES = (2, 4)  # avoids equal center-distance ties on the fixed tile lattice

# --- shape grammar ---
TEMPLATES = {
    4: np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
    5: np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int),
    6: np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=int),
    7: np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=int),
}
AREA_TO_COLOR = {4: 6, 5: 1, 6: 3, 7: 2}
AREA_TO_LEN = {4: 2, 5: 3, 6: 4, 7: 2}
AREA_TO_DIR = {5: 1, 6: -1, 7: -1}
AREA_DELTAS = {4: (-2, 0), 5: (-1, 2), 6: (-1, -3), 7: (-1, -1)}
AREAS = (4, 5, 6, 7)

# --- generation constraints ---
MIN_DIRECTION_CHANGES = 2
MAX_ATTEMPTS = 400


def _tile_slots(n: int) -> list[dict[str, int]]:
    slots: list[dict[str, int]] = []
    for c0 in TILE_COLS:
        for row_idx in range(int(n)):
            r0 = TILE_ROW_STRIDE * int(row_idx)
            slots.append(
                {
                    "r0": int(r0),
                    "c0": int(c0),
                    "center_r": int(r0 + 1),
                    "center_c": int(c0 + 1),
                }
            )
    return slots


def _column_major_slots(n: int) -> list[dict[str, int]]:
    return sorted(_tile_slots(n), key=lambda slot: (int(slot["c0"]), int(slot["r0"])))


def _distance_order_slots(n: int, start_row: int, start_col: int) -> list[dict[str, int]]:
    slots = _tile_slots(n)
    for slot in slots:
        slot["dist"] = int(abs(int(slot["center_r"]) - int(start_row)) + abs(int(slot["center_c"]) - int(start_col)))
    # Tie-break only for robustness; generated samples use start cols that make distances unique.
    slots.sort(key=lambda slot: (int(slot["dist"]), -int(slot["center_r"]), int(slot["center_c"])))
    return slots


def _draw_from_area_sequence(h: int, start_col: int, areas: list[int]) -> np.ndarray:
    out = np.zeros((int(h), LEFT_W), dtype=int)
    r_cur = int(h - 1)
    c_cur = int(start_col)
    out[r_cur, c_cur] = GRAY

    for area in areas:
        color = int(AREA_TO_COLOR[int(area)])
        seg_len = int(AREA_TO_LEN[int(area)])

        r_cur -= 1
        if r_cur < 0:
            raise AssertionError(f"segment exits top: r_cur={r_cur}")
        out[r_cur, c_cur] = color

        if int(area) == 4:
            end_r = int(r_cur - (seg_len - 1))
            if end_r < 0:
                raise AssertionError(f"vertical segment exits top: end_r={end_r}")
            for _ in range(seg_len - 1):
                r_cur -= 1
                out[r_cur, c_cur] = color
        else:
            dir_h = int(AREA_TO_DIR[int(area)])
            end_c = int(c_cur + dir_h * (seg_len - 1))
            if not (0 <= end_c < LEFT_W):
                raise AssertionError(
                    f"horizontal segment exits canvas: c_cur={c_cur}, dir={dir_h}, "
                    f"seg_len={seg_len}, left_w={LEFT_W}"
                )
            for _ in range(seg_len - 1):
                c_cur += dir_h
                out[r_cur, c_cur] = color

    return out


def _sample_path(n: int, start_col: int) -> list[int] | None:
    h = 4 * int(n) - 1
    r, c = int(h - 1), int(start_col)
    areas: list[int] = []

    for _ in range(2 * int(n)):
        opts = [
            int(area)
            for area in AREAS
            if 0 <= r + AREA_DELTAS[int(area)][0] < h and 0 <= c + AREA_DELTAS[int(area)][1] < LEFT_W
        ]
        if not opts:
            return None
        area = int(random.choice(opts))
        areas.append(area)
        dr, dc = AREA_DELTAS[area]
        r += int(dr)
        c += int(dc)

    kinds = [int(area == 4) for area in areas]
    changes = sum(1 for i in range(1, len(kinds)) if int(kinds[i]) != int(kinds[i - 1]))
    if changes < MIN_DIRECTION_CHANGES:
        return None
    return areas


def _render_grid(n: int, start_col: int, ordered_areas: list[int]) -> np.ndarray:
    h = 4 * int(n) - 1
    grid = np.zeros((int(h), GRID_W), dtype=int)
    grid[:, DIVIDER_COL] = DIVIDER_COLOR
    grid[int(h - 1), DIVIDER_COL + 1 + int(start_col)] = GRAY

    ordered_slots = _distance_order_slots(int(n), int(h - 1), int(start_col))
    for slot, area in zip(ordered_slots, ordered_areas):
        r0 = int(slot["r0"])
        c0 = int(slot["c0"])
        color = int(AREA_TO_COLOR[int(area)])
        grid[r0 : r0 + 3, c0 : c0 + 3] = TEMPLATES[int(area)] * color

    return grid


def _extract_area_orders(grid: np.ndarray) -> tuple[list[int], list[int], list[int]]:
    g = np.array(grid, dtype=int)
    h, _w = g.shape
    n = (int(h) + 1) // 4
    start_row = int(h - 1)
    gray = np.argwhere(g == GRAY)
    assert gray.size != 0
    start_col = int(gray[0, 1] - (DIVIDER_COL + 1))

    left = g[:, :DIVIDER_COL]
    glyphs: dict[tuple[int, int], int] = {}
    for comp in find_components(left, bg=0, connectivity=8, monochromatic=True, return_format="dict"):
        b = comp["bbox"]
        if bbox_h(b) != 3 or bbox_w(b) != 3:
            continue
        r0, c0, _r1, _c1 = map(int, b)
        glyphs[(int(r0), int(c0))] = int(comp["area"])

    ordered_slots = _distance_order_slots(int(n), int(start_row), int(start_col))
    column_slots = _column_major_slots(int(n))
    distance_order = [int(glyphs[(int(slot["r0"]), int(slot["c0"]))]) for slot in ordered_slots]
    column_order = [int(glyphs[(int(slot["r0"]), int(slot["c0"]))]) for slot in column_slots]
    dists = [int(slot["dist"]) for slot in ordered_slots]
    return distance_order, column_order, dists


def combo_behavior_metrics(grid: np.ndarray) -> dict[str, object]:
    distance_order, column_order, dists = _extract_area_orders(grid)
    h = int(grid.shape[0])
    start_col = int(np.argwhere(grid == GRAY)[0, 1] - (DIVIDER_COL + 1))
    out_distance = _draw_from_area_sequence(h, start_col, distance_order)
    out_column = _draw_from_area_sequence(h, start_col, column_order)
    return {
        "distance_order": distance_order,
        "column_order": column_order,
        "distances": dists,
        "unique_distances": len(set(dists)) == len(dists),
        "order_differs": distance_order != column_order,
        "output_differs_from_column_major": not np.array_equal(out_distance, out_column),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw an upward-growing polyline by decoding 3x3 glyph instructions around a divider.

    Input structure:
    - Left of a full-height vertical divider: 3x3 colored glyphs on the black background.
    - Right of the divider: exactly one gray (5) start pixel, placed on the bottom row.

    Rule:
    - Remove the divider and shift the gray start into the left canvas.
    - Sort glyphs by Manhattan distance from each glyph center to that mapped start point, from near to far.
    - For each glyph in that order, step one row upward, then draw the segment encoded by its area:
      area 4 -> up 2, area 5 -> right 3, area 6 -> left 4, area 7 -> left 2.

    Key insight: The glyph shape still determines the segment, but the execution order is now geometric:
    begin with the bottom start point and follow the nearest glyph centers first.
    """
    g = np.array(grid, dtype=int)
    h, _w = g.shape
    bg = int(get_bg(g))

    div_c, div_color = find_full_height_constant_col(g, bg=bg)
    assert div_c is not None

    out = np.zeros((int(h), int(div_c)), dtype=int)

    gray = np.argwhere(g == GRAY)
    if gray.size == 0:
        return out
    r_cur, c_cur = map(int, gray[0])
    c_cur = int(c_cur - (int(div_c) + 1))
    if not (0 <= c_cur < int(div_c)):
        return out
    out[r_cur, c_cur] = GRAY

    left = g[:, : int(div_c)]
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
            center_r = int(r0 + 1)
            center_c = int(c0 + 1)
            dist = int(abs(center_r - int(r_cur)) + abs(center_c - int(c_cur)))
            glyphs.append(
                {
                    "color": color,
                    "r0": int(r0),
                    "c0": int(c0),
                    "center_r": center_r,
                    "center_c": center_c,
                    "dist": dist,
                    "area": int(comp["area"]),
                }
            )

    glyphs.sort(key=lambda glyph: (int(glyph["dist"]), -int(glyph["center_r"]), int(glyph["center_c"])))

    for glyph in glyphs:
        color = int(glyph["color"])
        area = int(glyph["area"])
        seg_len = int(AREA_TO_LEN.get(area, 2))

        r_cur -= 1
        assert r_cur >= 0, f"segment exits top: r_cur={r_cur}"
        out[r_cur, int(c_cur)] = color

        if area == 4:
            end_r = int(r_cur - (seg_len - 1))
            assert end_r >= 0, f"vertical segment exits top: end_r={end_r}"
            for _ in range(seg_len - 1):
                r_cur -= 1
                out[r_cur, int(c_cur)] = color
        else:
            dir_h = int(AREA_TO_DIR.get(area, 1))
            end_c = int(c_cur + dir_h * (seg_len - 1))
            assert 0 <= end_c < int(div_c), (
                f"horizontal segment exits canvas: c_cur={c_cur}, dir={dir_h}, "
                f"seg_len={seg_len}, div_c={div_c}"
            )
            for _ in range(seg_len - 1):
                c_cur += dir_h
                out[r_cur, c_cur] = color

    return out


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(N_MIN, N_MAX)
    start_col = int(random.choice(START_COL_CHOICES))
    ordered_areas = _sample_path(int(n), int(start_col))
    if ordered_areas is None:
        return None

    grid = _render_grid(int(n), int(start_col), ordered_areas)
    metrics = combo_behavior_metrics(grid)
    if not bool(metrics["unique_distances"]):
        return None
    if not bool(metrics["order_differs"]):
        return None
    if not bool(metrics["output_differs_from_column_major"]):
        return None
    return grid


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
