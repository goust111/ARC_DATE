# Combo variant of 136b0064 (index 27)
# combo_spec: D1/4+D4/0+D7/0
# Name: 底部起始按距离向上绘制

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg


# Design analysis for the combo variant
# - Concrete behavior trace:
#   In the source task, the gray start is on the top row, glyphs are executed in column-major tile
#   order, and each instruction begins by moving the pen downward. Here the gray marker is shifted to
#   the bottom row, so the pen first moves upward before every segment. The execution order is also
#   no longer tied to the printed tile scan: for each 3x3 glyph we compute the Manhattan distance
#   from its center `(r0+1, c0+1)` to the shifted start point, then execute from nearest to farthest.
#   Example with `N=3`, `H=11`, `start_col=2`: the six tile-center distances are
#   bottom-left=2, bottom-right=4, middle-left=6, middle-right=8, top-left=10, top-right=12. So the
#   old order `(top-left, middle-left, bottom-left, top-right, middle-right, bottom-right)` becomes
#   `(bottom-left, bottom-right, middle-left, middle-right, top-left, top-right)`, and every step
#   grows upward from row 10 instead of downward from row 0.
# - Complexity calibration:
#   The glyph vocabulary, divider, colors, and 2-column 3x3 tile bank remain unchanged, so the task
#   still has repeated local evidence. To keep the new distance-order rule visually inferable rather
#   than tie-broken by hidden conventions, the generator only uses start columns 2 and 4, the two
#   placements for which all tile-center distances are distinct for every allowed `N`.
# - Implicit-assumption audit for `main()`:
#   1. The original `glyphs.sort(key=(c0, r0))` assumed scan order was the instruction order; this
#      must be replaced with a distance-based key derived from the shifted gray start.
#   2. The original `r_cur += 1` and downward area-4 loop assumed growth toward larger rows; both
#      must invert to upward motion with top-bound checks.
#   3. The original start row being at the top is no longer safe; the solver must accept the gray
#      cell where it actually appears and use that same row in the distance computation.
#   4. Because D1 and D7 interact, distance must be measured after removing the divider so the glyph
#      centers and start point live in the same 7-column coordinate system.
# - Placement legality in the generator:
#   We still sample a path first, but now in true execution order under upward deltas
#   `{4:(-2,0), 5:(-1,+2), 6:(-1,-3), 7:(-1,-1)}` from `(H-1, start_col)`. After sampling a valid
#   walk, we assign instruction `i` to the `i`-th nearest tile slot. This guarantees that solver
#   decoding reproduces the sampled walk exactly and that every intermediate point stays on-canvas.
# - Visual observability:
#   The bottom gray anchor makes the inversion explicit, and distance order is legible because the
#   earliest executed glyphs are physically nearest to that anchor. Using only unique-distance start
#   columns avoids ambiguous "same distance" examples that would hide the ordering rule from humans.
# - Visual cleanliness:
#   The left tile bank stays in its original separated 3x3 lattice, the right half still contains
#   only the lone gray marker, and the output remains a single colored trace plus that marker. No new
#   markers or overlays are introduced.
# - Change inferability / anti-degeneracy:
#   A valid generated sample must do more than merely survive the solver. The generator rejects any
#   case where the upward distance-order trace matches the upward column-major trace on the same
#   input, ensuring D1 has an observable behavioral effect instead of silently collapsing away.


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


def _glyph_sort_key(glyph: dict[str, int], start_r: int, start_c: int) -> tuple[int, int, int]:
    center_r = int(glyph["r0"] + 1)
    center_c = int(glyph["c0"] + 1)
    dist = abs(center_r - int(start_r)) + abs(center_c - int(start_c))
    return int(dist), int(-center_r), int(center_c)


def _draw_trace_from_areas(h: int, start_row: int, start_col: int, areas: list[int], *, start_color: int) -> np.ndarray:
    out = np.zeros((int(h), LEFT_W), dtype=int)
    r_cur = int(start_row)
    c_cur = int(start_col)
    out[r_cur, c_cur] = int(start_color)

    for area in areas:
        color = int(_AREA_TO_COLOR[int(area)])
        seg_len = int(AREA_TO_LEN[int(area)])

        r_cur -= 1
        assert r_cur >= 0, f"segment exits top: r_cur={r_cur}"
        out[r_cur, c_cur] = color

        if int(area) == 4:
            end_r = int(r_cur - (seg_len - 1))
            assert end_r >= 0, f"vertical segment exits top: end_r={end_r}"
            for _ in range(seg_len - 1):
                r_cur -= 1
                out[r_cur, c_cur] = color
        else:
            dir_h = int(AREA_TO_HDIR[int(area)])
            end_c = int(c_cur + dir_h * (seg_len - 1))
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
    Decode 3x3 glyph instructions into a polyline that rises from a bottom gray anchor.

    Input structure:
    - Left of a full-height divider: the same 3x3 monochrome glyph bank on the background.
    - Right of the divider: one gray (5) start pixel, now placed on the bottom row.

    Rule:
    - Remove the divider and map the gray start into the 7-column left canvas.
    - For each 3x3 glyph, use its pixel area to decode a segment: area 4 → up 2, area 5 → right 3,
      area 6 → left 4, area 7 → left 2.
    - Order the glyphs by Manhattan distance from each glyph center to the shifted gray start,
      nearest first.
    - Build one continuous trace by moving up one row before every decoded segment, then drawing it.

    Key insight: The glyph shapes still encode the movement instructions, but execution is now
    anchored by spatial proximity to a bottom start point rather than by a fixed scan order.
    """
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
        box = comp["bbox"]
        if bbox_h(box) == 3 and bbox_w(box) == 3:
            r0, c0, _r1, _c1 = map(int, box)
            glyphs.append({"color": color, "r0": r0, "c0": c0, "area": int(comp["area"])})

    glyphs.sort(key=lambda glyph: _glyph_sort_key(glyph, r_cur, c_cur))

    for glyph in glyphs:
        color = int(glyph["color"])
        area = int(glyph["area"])
        seg_len = int(AREA_TO_LEN.get(area, 2))

        r_cur -= 1
        assert r_cur >= 0, f"segment exits top: r_cur={r_cur}"
        out[r_cur, c_cur] = color

        if area == 4:
            end_r = int(r_cur - (seg_len - 1))
            assert end_r >= 0, f"vertical segment exits top: end_r={end_r}"
            for _ in range(seg_len - 1):
                r_cur -= 1
                out[r_cur, c_cur] = color
        else:
            dir_h = int(AREA_TO_HDIR.get(area, 1))
            end_c = int(c_cur + dir_h * (seg_len - 1))
            assert 0 <= end_c < int(div_c), (
                f"horizontal segment exits canvas: c_cur={c_cur}, dir={dir_h}, "
                f"seg_len={seg_len}, div_c={div_c}"
            )
            for _ in range(seg_len - 1):
                c_cur += dir_h
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
START_COL_CHOICES = (2, 4)

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
MAX_ATTEMPTS = 500


def _slot_records(n: int, start_col: int) -> list[dict[str, int]]:
    h = 4 * int(n) - 1
    start_row = h - 1
    slots: list[dict[str, int]] = []
    for tile_row in range(int(n)):
        r0 = TILE_ROW_STRIDE * int(tile_row)
        for c0 in TILE_COLS:
            center_r = int(r0 + 1)
            center_c = int(c0 + 1)
            dist = abs(center_r - start_row) + abs(center_c - int(start_col))
            slots.append(
                {
                    "tile_row": int(tile_row),
                    "r0": int(r0),
                    "c0": int(c0),
                    "center_r": center_r,
                    "center_c": center_c,
                    "dist": int(dist),
                }
            )
    slots.sort(key=lambda slot: (int(slot["dist"]), int(-slot["center_r"]), int(slot["center_c"])))
    return slots


def _column_major_slots(n: int) -> list[tuple[int, int]]:
    return [(TILE_ROW_STRIDE * int(tile_row), int(tile_col)) for tile_col in TILE_COLS for tile_row in range(int(n))]


def _sample_path(n: int, start_col: int) -> list[int] | None:
    """Sample a valid upward-growing instruction list in execution order."""
    h = 4 * int(n) - 1
    r, c = h - 1, int(start_col)
    areas: list[int] = []
    for _ in range(2 * int(n)):
        opts = [
            area
            for area in _AREAS
            if 0 <= r + _AREA_DELTA[area][0] < h and 0 <= c + _AREA_DELTA[area][1] < LEFT_W
        ]
        if not opts:
            return None
        area = int(random.choice(opts))
        areas.append(area)
        dr, dc = _AREA_DELTA[area]
        r += int(dr)
        c += int(dc)

    types = [area == 4 for area in areas]
    changes = sum(1 for idx in range(1, len(types)) if types[idx] != types[idx - 1])
    if changes < MIN_DIRECTION_CHANGES:
        return None
    return areas


def _assignment_from_distance_order(n: int, start_col: int, areas: list[int]) -> dict[tuple[int, int], int]:
    slots = _slot_records(n, start_col)
    assert len(slots) == len(areas)
    return {(int(slot["r0"]), int(slot["c0"])): int(area) for slot, area in zip(slots, areas)}


def _areas_in_column_major(n: int, assignment: dict[tuple[int, int], int]) -> list[int]:
    return [int(assignment[(int(r0), int(c0))]) for r0, c0 in _column_major_slots(n)]


def _render_grid(n: int, start_col: int, assignment: dict[tuple[int, int], int]) -> np.ndarray:
    """Paint divider, bottom gray start, and the shuffled glyph bank into the input grid."""
    h = 4 * int(n) - 1
    grid = np.zeros((h, GRID_W), dtype=int)
    grid[:, DIVIDER_COL] = DIVIDER_COLOR
    grid[h - 1, DIVIDER_COL + 1 + int(start_col)] = GRAY

    for (r0, c0), area in assignment.items():
        color = int(_AREA_TO_COLOR[int(area)])
        grid[int(r0) : int(r0) + 3, int(c0) : int(c0) + 3] = _TEMPLATES[int(area)] * color
    return grid


def _gen_attempt() -> np.ndarray | None:
    n = int(random.randint(N_MIN, N_MAX))
    start_col = int(random.choice(START_COL_CHOICES))
    slots = _slot_records(n, start_col)
    if len({int(slot["dist"]) for slot in slots}) != len(slots):
        return None

    areas = _sample_path(n, start_col)
    if areas is None:
        return None

    assignment = _assignment_from_distance_order(n, start_col, areas)
    column_major_areas = _areas_in_column_major(n, assignment)
    h = 4 * int(n) - 1

    intended = _draw_trace_from_areas(h, h - 1, start_col, areas, start_color=GRAY)
    baseline = _draw_trace_from_areas(h, h - 1, start_col, column_major_areas, start_color=GRAY)
    if np.array_equal(intended, baseline):
        return None

    grid = _render_grid(n, start_col, assignment)
    if not np.array_equal(main(grid), intended):
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
            gray = np.argwhere(inp == GRAY)
            if gray.shape != (1, 2) or int(gray[0, 0]) != inp.shape[0] - 1:
                continue
            return inp
        except Exception:
            continue
    raise RuntimeError(f"generate_input: failed after {MAX_ATTEMPTS} attempts")
