# Combo variant of 4c3d4a41 (index 30)
# combo_spec: D1/1+D4/3+D10/0
# Name: 数值闭环三角

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import bbox_coords, colors_present, components_mask, get_bg, get_bg as _bg_color, paste


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
# ---------------------------------------------------------------------------
#
# a) Concrete behavior trace:
#    - Geometry stays the same as the original task: one right-side grey frame,
#      one detached 7-wide grey bar with 4 odd-column projections, and 4 colored
#      slot columns inside the frame.
#    - The fused rule closes three quantities into one loop:
#        input colored-count n == matched projection height h == output colored-count.
#    - Matching is no longer positional. The bar always moves to its canonical
#      centered location in the frame, then each projection looks for a slot
#      column whose colored pixel count equals its height.
#    - Example:
#        heights left->right      = [2, 1, 2, 1]
#        input counts left->right = [1, 2, 1, 2]
#        input colors             = [A, B, C, D]
#      Count-1 columns are [A, C], count-2 columns are [B, D].
#      Output projections of height 2 take [B, D] in left-to-right order, and
#      height 1 projections take [A, C]. Output colors become [B, A, D, C].
#      The total colored count per moved column is preserved because the matched
#      count already equals the target height.
#
# b) Complexity calibration:
#    - Exact output count = height is only physically possible in this fixed
#      frame if height <= available cells above the projection. With base row 5,
#      a projection of height h leaves exactly (5 - h) cells above it, so h=3/4
#      would make the "keep h colored cells above the tip" rule impossible.
#    - Therefore heights/counts are intentionally restricted to {1, 2}. This is
#      the smallest nontrivial regime that supports the closed loop and remains
#      visually countable by eye.
#    - Each sample uses two 1s and two 2s, giving repeated evidence for the
#      equality rule while still allowing horizontal permutation.
#
# c) Hidden-assumption audit on the original solver:
#    - Original horizontal shift logic assumed colored columns reveal the bar's
#      x-placement. That is invalid here; x-placement is fixed and matching is
#      numeric, not spatial.
#    - Original recoloring assumed output count comes from the source column.
#      Here that count source changes to the matched projection height. Because
#      the generator enforces count==height, the same value is seen from both
#      sides of the loop.
#    - Original output could keep source columns in place. That would collapse
#      the variant, so the generator forbids identity order and requires actual
#      color movement after size matching.
#
# d) Placement legality:
#    - Frame and bar geometries are unchanged from the original task, so the bar
#      still lands fully inside the frame at cols 11..17 and base row 5.
#    - Input counts are only 1 or 2, which fit all slot lengths [3, 4, 4, 3].
#    - Output stacks are top-aligned within the shaft and use exactly h cells;
#      with h in {1, 2}, they always fit above the bar.
#
# e) Visual observability:
#    - A human can count both signatures directly: projection heights on the bar
#      and colored-run lengths in the slot.
#    - Because the input order of counts differs from the bar order, colors move
#      sideways in the output. Because the counts are preserved exactly, the same
#      numbers reappear after the move.
#    - The generator uses the full set of non-degenerate two-value pattern pairs
#      that move all four colors, so the fused behavior stays visible while still
#      covering multiple left-to-right orders within each size group.
#
# f) Visual cleanliness:
#    - The solver clears the slot interior and rebuilds clean shafts above the
#      placed bar. That avoids leftover grey caps from the source layout.
#    - Only one colored run appears in each destination column, separated by one
#      empty column exactly as in the original geometry.
#
# g) Color-rule audit:
#    - Colors are labels only. No sorting or matching uses numeric color ids.
#      Distinct colors make the sideways permutation easy to see.
#
# 9) New solver assumptions and generator guarantees:
#    - The largest grey component is the frame and the second largest is the bar.
#    - There are exactly four colored slot columns, each monochrome and contiguous.
#    - Projection heights and input colored counts both come from {1, 2}.
#    - For each height value, the number of projections of that height equals the
#      number of slot columns with that colored count, so left-to-right stable
#      matching within each size group is total and unambiguous.


def _bbox(pts):
    return bbox_coords(pts if isinstance(pts, np.ndarray) else np.asarray(list(pts), dtype=int))


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _most_common(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _projection_data(bar_pts: np.ndarray, bar_bbox: tuple[int, int, int, int]) -> tuple[int, list[tuple[int, int]]]:
    """Return bar base row and ordered pairs (projection_col, projection_height)."""
    base_row = max(int(r) for r, _ in bar_pts)
    col_min_row: dict[int, int] = {}
    for r, c in bar_pts:
        r = int(r)
        c = int(c)
        col_min_row[c] = min(col_min_row.get(c, r), r)

    _br0, bc0, _br1, bc1 = bar_bbox
    data = []
    for c in range(int(bc0), int(bc1) + 1, 2):
        top = col_min_row.get(int(c), base_row)
        data.append((int(c), int(base_row - top + 1)))
    return int(base_row), data


def _slot_columns(g: np.ndarray, bg: int, grey: int, slot_bbox: tuple[int, int, int, int]) -> list[dict]:
    """Collect colored slot columns left-to-right with their count and color label."""
    sr0, sc0, sr1, sc1 = slot_bbox
    cols = []
    for c in range(sc0 + 1, sc1):
        rows = [r for r in range(sr0 + 1, sr1) if int(g[r, c]) != bg and int(g[r, c]) != grey]
        if not rows:
            continue
        cols.append({"col": int(c), "count": int(len(rows)), "color": int(g[rows[0], c])})
    return cols


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the grey bar into the frame, then close a count-height-count loop to place the colors.

    Input structure:
    - Background plus one structural grey color form two disconnected objects: a large right-side frame and a smaller left-side bar with four upward projections.
    - Inside the frame, four separated colored columns appear at fixed slot positions; each column's visible colored run is monochrome and countable.

    Rule:
    - Translate the bar into its canonical centered position inside the frame.
    - Measure every projection height, count the colored cells in every input slot column, and pair projections to columns by exact equality of those counts.
    - For each matched pair, repaint that column above the placed projection using the matched input color and exactly that many cells, top-aligned inside the shaft.

    Key insight:
    - The same number plays three roles at once: it is the input column length, the matched projection height, and the output colored count. Colors therefore move horizontally by numeric matching rather than staying in their original columns.
    """
    g = np.asarray(grid, dtype=int)

    bg = _bg_color(g)
    other = g[g != bg]
    if other.size == 0:
        return g.copy()
    grey = _most_common(other)

    def comps_of_color(color: int):
        comps = [(len(pts), _bbox(pts), pts) for pts in _cc4(g == color)]
        comps.sort(key=lambda t: (-t[0], t[1]))
        return comps

    grey_comps = comps_of_color(grey)
    if len(grey_comps) < 2:
        return g.copy()

    (_, slot_bbox, _), (_, bar_bbox, bar_pts) = grey_comps[0], grey_comps[1]
    sr0, sc0, sr1, sc1 = slot_bbox
    _br0, bc0, _br1, bc1 = bar_bbox

    out = np.full_like(g, bg)
    out[sr0 : sr1 + 1, sc0 : sc1 + 1] = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
    if sr1 - sr0 >= 2 and sc1 - sc0 >= 2:
        out[sr0 + 1 : sr1, sc0 + 1 : sc1] = bg

    bar_base_row, proj_data = _projection_data(bar_pts, bar_bbox)
    target_base_row = (sr1 - 1) - 1
    dy = target_base_row - bar_base_row

    bar_w = bc1 - bc0 + 1
    slot_w = sc1 - sc0 + 1
    target_bar_left = sc0 + (slot_w - bar_w) // 2
    dx = target_bar_left - bc0

    placed: list[tuple[int, int]] = []
    for r, c in bar_pts:
        rr = int(r + dy)
        cc = int(c + dx)
        if sr0 <= rr <= sr1 and sc0 <= cc <= sc1:
            out[rr, cc] = grey
            placed.append((rr, cc))

    placed_by_col: dict[int, list[int]] = {}
    for rr, cc in placed:
        placed_by_col.setdefault(int(cc), []).append(int(rr))

    slot_cols = _slot_columns(g, bg, grey, slot_bbox)
    by_count: dict[int, list[dict]] = {}
    for info in slot_cols:
        by_count.setdefault(int(info["count"]), []).append(info)

    interior_top = sr0 + 1
    for src_col, height in proj_data:
        out_col = int(src_col + dx)
        if out_col not in placed_by_col:
            continue

        matches = by_count.get(int(height), [])
        if not matches:
            continue
        info = matches.pop(0)

        bar_top = min(placed_by_col[out_col])
        desired_bottom = bar_top - 1
        if desired_bottom < interior_top:
            continue

        paint_count = int(height)
        if interior_top + paint_count - 1 > desired_bottom:
            continue

        out[interior_top : desired_bottom + 1, out_col] = grey
        out[interior_top : interior_top + paint_count, out_col] = int(info["color"])

    return out


BG = 0
GREY = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)

H, W = 8, 20

FRAME_R0, FRAME_C0 = 0, 9
FRAME_H, FRAME_W = 8, 11
LEFT_WALL_OPEN_ROWS = (4, 5)

SLOT_COLS = [11, 13, 15, 17]
SLOT_LENS = [3, 4, 4, 3]
SLOT_TOP_ROW = 1

BAR_INPUT_C0 = 1
BAR_OUTPUT_C0 = 11
BAR_R0 = 2
BAR_H, BAR_W = 4, 7
BAR_LOCAL_PROJ_COLS = [0, 2, 4, 6]
BAR_BASE_ABS_ROW = BAR_R0 + BAR_H - 1

PATTERN_PAIRS = [
    ([1, 1, 2, 2], [2, 1, 2, 1]),
    ([1, 1, 2, 2], [2, 2, 1, 1]),
    ([1, 2, 1, 2], [2, 1, 2, 1]),
    ([1, 2, 1, 2], [2, 2, 1, 1]),
    ([1, 2, 2, 1], [2, 1, 1, 2]),
    ([2, 1, 1, 2], [1, 2, 2, 1]),
    ([2, 1, 2, 1], [1, 1, 2, 2]),
    ([2, 1, 2, 1], [1, 2, 1, 2]),
    ([2, 2, 1, 1], [1, 1, 2, 2]),
    ([2, 2, 1, 1], [1, 2, 1, 2]),
]
MAX_ATTEMPTS = 300


def _frame_patch() -> np.ndarray:
    patch = np.zeros((FRAME_H, FRAME_W), dtype=int)
    patch[[0, -1], :] = GREY
    patch[:, -1] = GREY
    patch[:, 0] = GREY
    patch[list(LEFT_WALL_OPEN_ROWS), 0] = BG
    return patch


def _bar_patch(heights: list[int]) -> np.ndarray:
    patch = np.zeros((BAR_H, BAR_W), dtype=int)
    patch[-1, :] = GREY
    for local_c, height in zip(BAR_LOCAL_PROJ_COLS, heights):
        patch[BAR_H - height : BAR_H, local_c] = GREY
    return patch


def _paint_input_slot_column(grid: np.ndarray, col: int, length: int, color: int, count: int) -> None:
    cap = length - count
    grid[SLOT_TOP_ROW : SLOT_TOP_ROW + cap, col] = GREY
    grid[SLOT_TOP_ROW + cap : SLOT_TOP_ROW + length, col] = color


def _paint_output_slot_column(grid: np.ndarray, col: int, color: int, height: int) -> None:
    desired_bottom = BAR_BASE_ABS_ROW - height
    grid[SLOT_TOP_ROW : desired_bottom + 1, col] = GREY
    grid[SLOT_TOP_ROW : SLOT_TOP_ROW + height, col] = color


def _matched_output_colors(heights: list[int], colors: list[int], counts: list[int]) -> list[int]:
    by_count: dict[int, list[int]] = {}
    for color, count in zip(colors, counts):
        by_count.setdefault(int(count), []).append(int(color))

    out_colors = []
    for height in heights:
        out_colors.append(int(by_count[int(height)].pop(0)))
    return out_colors


def _build_input(heights: list[int], colors: list[int], counts: list[int]) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    paste(grid, _bar_patch(heights), BAR_R0, BAR_INPUT_C0)
    for col, length, color, count in zip(SLOT_COLS, SLOT_LENS, colors, counts):
        _paint_input_slot_column(grid, col, length, color, count)
    return grid


def _build_output(heights: list[int], colors: list[int], counts: list[int]) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    paste(grid, _bar_patch(heights), BAR_R0, BAR_OUTPUT_C0)
    out_colors = _matched_output_colors(heights, colors, counts)
    for col, color, height in zip(SLOT_COLS, out_colors, heights):
        _paint_output_slot_column(grid, col, color, height)
    return grid


def _combo_metrics(heights: list[int], colors: list[int], counts: list[int]) -> dict[str, int]:
    out_colors = _matched_output_colors(heights, colors, counts)
    moved = sum(int(src != dst) for src, dst in zip(colors, out_colors))
    return {
        "moved_colors": int(moved),
        "count_mismatches": int(sum(int(h != n) for h, n in zip(heights, counts))),
        "ones": int(sum(int(h == 1) for h in heights)),
        "twos": int(sum(int(h == 2) for h in heights)),
    }


def _sample_params(rng: np.random.Generator) -> tuple[list[int], list[int], list[int]]:
    colors = [int(v) for v in rng.choice(PALETTE, size=len(SLOT_COLS), replace=False)]
    heights, counts = PATTERN_PAIRS[int(rng.integers(0, len(PATTERN_PAIRS)))]
    return list(heights), colors, list(counts)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    heights, colors, counts = _sample_params(rng)
    metrics = _combo_metrics(heights, colors, counts)
    if metrics["count_mismatches"] == 0:
        return None
    if metrics["moved_colors"] < 4:
        return None

    grid = _build_input(heights, colors, counts)
    expected = _build_output(heights, colors, counts)

    if get_bg(grid) != BG:
        return None
    if sorted(colors_present(grid, exclude=[BG, GREY])) != sorted(colors):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_ATTEMPTS} attempts")
