# Combo variant of 136b0064 (index 4)
# combo_spec: D1/1+D5/1
# Name: 色序排序与循环涂色

from __future__ import annotations

import os
import random
import sys
from functools import lru_cache

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
_SOLVER_DIR = os.path.join(_ROOT, "solvers")
if _SOLVER_DIR not in sys.path:
    sys.path.append(_SOLVER_DIR)

from arc_utils import bbox_h, bbox_w, find_components, get_bg


# Design analysis for the fused variant:
# a) Concrete behavior trace:
#    Original task executes glyphs in tile column-major order and copies each glyph color onto
#    its output segment. This fusion breaks both assumptions. Example: if the four tile colors
#    encountered in space are [6, 1, 3, 2], the original solver would execute them in that
#    spatial order. The fused solver instead reorders them by the visible fixed color ladder
#    [1, 2, 3, 6], so the execution order becomes [1, 2, 3, 6] regardless of tile position.
#    The segment colors no longer come from glyph colors at all: segment 1 uses palette color 7,
#    segment 2 uses 8, segment 3 uses 9, segment 4 wraps to 7 again. Input colors therefore
#    act only as ordering labels, while output colors act only as cyclic paint.
# b) Complexity calibration:
#    To keep the fusion visible, every generated grid contains all four glyph colors {1,2,3,6},
#    so the color-sorted order is observable and cannot collapse to a single-color trivial case.
#    The output palette uses three colors {7,8,9} that do not appear on the input glyphs, making
#    the color decoupling immediately visible across only a few examples.
# c) Hidden assumptions broken by the fusion:
#    - `glyphs.sort(key=(c0,r0))` is no longer valid; execution order depends on color rank first.
#    - `segment_color = glyph_color` is invalid; output colors come from a fixed cycle.
#    - The generator can no longer equate tile placement order with execution order; it must build
#      a path in true color-sorted order and then scramble the glyph placement so D1/1 matters.
# d) Placement legality:
#    Tiles still occupy the original disjoint 3x3 slots, the divider and gray start keep their
#    original geometry, and the sampled walk is validated so every segment stays inside the 7-wide
#    left canvas and below the bottom edge.
# e/f/g) Observability and visual cleanliness:
#    The grid remains clean because only one glyph occupies each tile slot and only one polyline
#    occupies the output canvas. The shuffle of input glyph positions creates visible disagreement
#    between spatial reading order and true execution order, while the disjoint output palette
#    prevents the fused task from looking like a minor recolor of the original.


GRID_W = 15
DIVIDER_COL = 7
DIVIDER_COLOR = 4
LEFT_W = 7
TILE_COLS = (0, 4)
TILE_ROW_STRIDE = 4
GRAY = 5
GRAY_START_ROW = 0
N_MIN, N_MAX = 2, 5
MAX_ATTEMPTS = 400

AREA_TO_TEMPLATE = {
    4: np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
    5: np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int),
    6: np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=int),
    7: np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=int),
}
AREA_TO_INPUT_COLOR = {4: 6, 5: 1, 6: 3, 7: 2}
AREA_TO_LEN = {4: 2, 5: 3, 6: 4, 7: 2}
AREA_TO_DIR = {5: 1, 6: -1, 7: -1}
AREA_TO_DELTA = {4: (2, 0), 5: (1, 2), 6: (1, -3), 7: (1, -1)}

COLOR_ORDER = (1, 2, 3, 6)
COLOR_RANK = {color: idx for idx, color in enumerate(COLOR_ORDER)}
COLOR_TO_AREA = {1: 5, 2: 7, 3: 6, 6: 4}
OUTPUT_PALETTE = (7, 8, 9)
_N_QUEUE: list[int] = []


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


def _collect_glyphs(left: np.ndarray, *, bg: int, ignored_colors: set[int]) -> list[dict[str, int]]:
    glyphs: list[dict[str, int]] = []
    for comp in find_components(left, bg=bg, connectivity=8, monochromatic=True, return_format="dict"):
        color = int(comp["color"])
        if color in ignored_colors:
            continue
        bbox = comp["bbox"]
        if bbox_h(bbox) != 3 or bbox_w(bbox) != 3:
            continue
        r0, c0, _r1, _c1 = map(int, bbox)
        glyphs.append({"color": color, "r0": r0, "c0": c0, "area": int(comp["area"])})
    return glyphs


def _draw_segment(out: np.ndarray, r_cur: int, c_cur: int, area: int, color: int) -> tuple[int, int]:
    h, w = out.shape
    seg_len = int(AREA_TO_LEN[area])

    r_cur = int(r_cur + 1)
    assert r_cur < h, f"segment exits bottom: r_cur={r_cur}, H={h}"
    out[r_cur, c_cur] = color

    if area == 4:
        end_r = int(r_cur + seg_len - 1)
        assert end_r < h, f"vertical segment exits bottom: end_r={end_r}, H={h}"
        for _ in range(seg_len - 1):
            r_cur = int(r_cur + 1)
            out[r_cur, c_cur] = color
    else:
        dir_h = int(AREA_TO_DIR[area])
        end_c = int(c_cur + dir_h * (seg_len - 1))
        assert 0 <= end_c < w, (
            f"horizontal segment exits canvas: c_cur={c_cur}, dir={dir_h}, seg_len={seg_len}, W={w}"
        )
        for _ in range(seg_len - 1):
            c_cur = int(c_cur + dir_h)
            out[r_cur, c_cur] = color

    return int(r_cur), int(c_cur)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the 3x3 glyphs into a polyline, but execute them by input-color rank rather than by location.

    Input structure:
    - Left of the full-height divider: one monochrome 3x3 glyph in every tile slot, using the original
      glyph colors {1,2,3,6}.
    - Right of the divider: a single gray start cell whose shifted position seeds the drawing canvas.

    Rule:
    - Remove the divider and move the gray start onto the 7-column left canvas.
    - Gather all 3x3 glyphs, sort them by the fixed visible color order 1 -> 2 -> 3 -> 6, and use
      column-major position only as a tie-break among equal-colored glyphs.
    - Each glyph's area still defines the geometric instruction, but the drawn segment color now comes
      from the repeating palette 7, 8, 9, 7, 8, 9, ... according to processing index.

    Key insight:
    - Shape still tells you how to move, input color tells you when to execute that move, and output
      color is a separate cyclic paint schedule unrelated to the glyph's own color.
    """
    g = np.array(grid, dtype=int)
    h, _w = g.shape
    bg = int(get_bg(g))

    div_c, div_color = find_full_height_constant_col(g, bg=bg)
    assert div_c is not None

    out = np.zeros((h, int(div_c)), dtype=int)

    gray = np.argwhere(g == GRAY)
    if gray.size == 0:
        return out
    r_cur, c_cur = map(int, gray[0])
    c_cur = int(c_cur - (int(div_c) + 1))
    if not (0 <= c_cur < int(div_c)):
        return out
    out[r_cur, c_cur] = GRAY

    left = g[:, : int(div_c)]
    ignored = {bg, GRAY}
    if div_color is not None:
        ignored.add(int(div_color))
    glyphs = _collect_glyphs(left, bg=bg, ignored_colors=ignored)

    glyphs.sort(key=lambda glyph: (COLOR_RANK[int(glyph["color"])], int(glyph["c0"]), int(glyph["r0"])))

    for idx, glyph in enumerate(glyphs):
        color = int(glyph["color"])
        assert color in COLOR_RANK, f"unexpected glyph color {color}"
        seg_color = int(OUTPUT_PALETTE[idx % len(OUTPUT_PALETTE)])
        r_cur, c_cur = _draw_segment(out, r_cur, c_cur, int(glyph["area"]), seg_color)

    return out


def _execution_areas_from_counts(counts: dict[int, int]) -> list[int]:
    areas: list[int] = []
    for color in COLOR_ORDER:
        area = COLOR_TO_AREA[color]
        areas.extend([area] * int(counts[area]))
    return areas


def _column_major_slots(n: int) -> list[tuple[int, int]]:
    slots: list[tuple[int, int]] = []
    for idx in range(2 * n):
        c0 = TILE_COLS[idx // n]
        r0 = TILE_ROW_STRIDE * (idx % n)
        slots.append((r0, c0))
    return slots


def _start_col_bounds(counts: dict[int, int]) -> tuple[int, int]:
    n5, n7, n6 = counts[5], counts[7], counts[6]
    min_start = max(0, 3 * n6 + n7 - 2 * n5)
    max_start = min(LEFT_W - 1, 6 - 2 * n5)
    return int(min_start), int(max_start)


@lru_cache(maxsize=None)
def _feasible_count_profiles(n: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    extra = 2 * n - 4
    profiles: list[tuple[tuple[int, int], ...]] = []

    for add5 in range(extra + 1):
        for add7 in range(extra - add5 + 1):
            for add6 in range(extra - add5 - add7 + 1):
                add4 = extra - add5 - add7 - add6
                counts = {5: 1 + add5, 7: 1 + add7, 6: 1 + add6, 4: 1 + add4}
                min_start, max_start = _start_col_bounds(counts)
                if min_start <= max_start:
                    profiles.append(tuple((area, counts[area]) for area in (5, 7, 6, 4)))

    assert profiles, f"no feasible count profile for n={n}"
    return tuple(profiles)


def _next_n() -> int:
    global _N_QUEUE
    if not _N_QUEUE:
        _N_QUEUE = list(range(N_MIN, N_MAX + 1))
        random.shuffle(_N_QUEUE)
    return int(_N_QUEUE.pop())


def _sample_execution_plan(n: int) -> tuple[int, list[int]] | None:
    profile = random.choice(_feasible_count_profiles(n))
    counts = {int(area): int(count) for area, count in profile}
    exec_areas = _execution_areas_from_counts(counts)
    min_start, max_start = _start_col_bounds(counts)
    if min_start > max_start:
        return None

    start_col = random.randint(min_start, max_start)
    return start_col, exec_areas


def _shuffle_spatial_areas(exec_areas: list[int]) -> list[int] | None:
    spatial = list(exec_areas)
    for _ in range(80):
        random.shuffle(spatial)
        if spatial == exec_areas:
            continue
        spatial_ranks = [COLOR_RANK[AREA_TO_INPUT_COLOR[area]] for area in spatial]
        inversion_count = sum(
            1
            for i in range(len(spatial_ranks))
            for j in range(i + 1, len(spatial_ranks))
            if spatial_ranks[i] > spatial_ranks[j]
        )
        if inversion_count >= 2:
            return spatial
    return None


def _render_grid(n: int, start_col: int, spatial_areas: list[int]) -> np.ndarray:
    h = 4 * n - 1
    grid = np.zeros((h, GRID_W), dtype=int)
    grid[:, DIVIDER_COL] = DIVIDER_COLOR
    grid[GRAY_START_ROW, DIVIDER_COL + 1 + start_col] = GRAY

    for (r0, c0), area in zip(_column_major_slots(n), spatial_areas):
        color = int(AREA_TO_INPUT_COLOR[area])
        grid[r0 : r0 + 3, c0 : c0 + 3] = AREA_TO_TEMPLATE[area] * color
    return grid


def _old_behavior_output(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    h, _w = g.shape
    bg = int(get_bg(g))
    div_c, div_color = find_full_height_constant_col(g, bg=bg)
    assert div_c is not None

    out = np.zeros((h, int(div_c)), dtype=int)
    gray = np.argwhere(g == GRAY)
    r_cur, c_cur = map(int, gray[0])
    c_cur = int(c_cur - (int(div_c) + 1))
    out[r_cur, c_cur] = GRAY

    left = g[:, : int(div_c)]
    ignored = {bg, GRAY}
    if div_color is not None:
        ignored.add(int(div_color))
    glyphs = _collect_glyphs(left, bg=bg, ignored_colors=ignored)
    glyphs.sort(key=lambda glyph: (int(glyph["c0"]), int(glyph["r0"])))

    for glyph in glyphs:
        r_cur, c_cur = _draw_segment(out, r_cur, c_cur, int(glyph["area"]), int(glyph["color"]))
    return out


def _gen_attempt(n: int) -> np.ndarray | None:
    plan = _sample_execution_plan(n)
    if plan is None:
        return None
    start_col, exec_areas = plan
    spatial_areas = _shuffle_spatial_areas(exec_areas)
    if spatial_areas is None:
        return None
    return _render_grid(n, start_col, spatial_areas)


def generate_input() -> np.ndarray:
    target_n = _next_n()
    for _ in range(MAX_ATTEMPTS):
        try:
            inp = _gen_attempt(target_n)
            if inp is None:
                continue
            out = main(inp)
            if out.shape != (inp.shape[0], LEFT_W):
                continue
            if np.array_equal(out, _old_behavior_output(inp)):
                continue
            return inp
        except Exception:
            continue
    raise RuntimeError(f"generate_input: failed after {MAX_ATTEMPTS} attempts")
