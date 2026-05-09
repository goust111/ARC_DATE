# Combo variant of 4c3d4a41 (index 4)
# combo_spec: D4/3+D10/0
# Name: 数量由高度决定+高度由列属性决定

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import bbox_coords, colors_present, components_mask, get_bg, get_bg as _bg_color, paste


# Design analysis for the combo variant D4/3 + D10/0
# a) Concrete behavior:
#    - The input still shows a right-hand grey frame plus a left-hand grey bar with four projections.
#    - Each frame column remains monochrome, but now its colored-cell count is no longer free:
#      it must exactly match the aligned projection height.
#    - Example with shared counts/heights [2, 1, 2, 1]:
#      input columns contain 2/1/2/1 colored cells at cols 11/13/15/17, and the bar projections
#      at the matching bar columns have heights 2/1/2/1. After the bar slides into the frame,
#      output columns keep exactly 2/1/2/1 colored cells and place them from the interior top
#      downward, with grey filling the rest of each shaft down to the bar.
#    - Compared with the seed task, two things change together: the input stack length predicts the
#      bar height, and the output count is rebuilt from that shared value rather than from the old
#      bottom-aligned visible stack.
# b) Complexity calibration:
#    - Exact "output count = height" only fits above the bar when the shared count is at most 2 in
#      this fixed 8x20 geometry, because a height-3 projection leaves only 2 cells above it.
#    - Therefore the generator restricts the shared counts/heights to {1, 2}. This is not a
#      simplification accident; it is the smallest legal domain that preserves the fusion rule.
#    - Every sample uses the multiset [1, 1, 2, 2] in shuffled order, so both values repeat and the
#      correspondence can be seen multiple times within a single example.
# c) Hidden-assumption audit on the original solver:
#    - The original solver treated the number of output colored cells as "whatever colored cells were
#      already present in that frame column, clipped by available space" and painted them flush to the bar.
#      That assumption fails here because the count must equal the projection height and the rebuilt stack
#      starts at the top of the frame interior.
#    - The original projection-column detection using only cells above the base would miss height-1
#      projections. The combo solver instead derives projection columns from alternating cells on the full base row.
#    - The combo rule also introduces a new dependency between two formerly independent signals:
#      input colored counts and bar heights. The solver explicitly reads both and only uses columns
#      where the two measurements agree, while the generator guarantees that agreement.
# d) Placement legality:
#    - Frame placement, slot columns, and bar translation remain the same as in the seed task, so the
#      moved bar stays fully inside the frame.
#    - Shared counts/heights in {1, 2} guarantee exact top-aligned repainting above every projection
#      without overlap with the bar or the frame walls.
#    - Slot columns remain separated by one empty column, preserving unambiguous one-to-one matching.
# e) Visual observability:
#    - In the input, taller projections always pair with columns containing two colored cells, while
#      shorter projections pair with columns containing one. Because both values occur twice, the rule
#      has repeated local support instead of a single fragile clue.
#    - In the output, the same per-column quantity reappears as the top-aligned colored segment length,
#      making the closed loop "input count = height = output count" visible across the whole example.
# f) Visual cleanliness:
#    - Output shafts are rebuilt cleanly as "top color segment + grey remainder + bar", so there is no
#      leftover mixture from the input caps.
#    - No new objects are added; only the count dependency and the vertical placement of the colored
#      segment change, keeping the puzzle legible.
# g) Color-rule audit:
#    - Colors are not sorted or remapped by id. Each destination column keeps its own input color;
#      only the shared count controls how many cells survive.
# 9) New solver assumptions checked against the generator:
#    - Largest grey component is the frame and second largest is the bar: guaranteed by fixed geometry.
#    - The bar has a solid 7-cell base, so alternating base columns recover all four projection columns.
#    - Exactly four destination columns exist, each monochrome and each aligned with one projection.
#    - Input colored count equals projection height in every destination column, guaranteed by generation.
#    - Shared counts/heights are in {1, 2}, so exact top-aligned repainting is always feasible.


def _bbox(pts):
    return bbox_coords(pts if isinstance(pts, np.ndarray) else np.asarray(list(pts), dtype=int))


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _most_common(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide a grey bar into a grey frame, then rebuild each colored shaft from a shared count-height cue.

    Input structure:
    - The most common color is background, and the main non-background color is the grey structural color.
    - Grey forms two disconnected objects: a right-side frame and a left-side horizontal bar whose four projections rise from alternating base columns.
    - Inside the frame, four monochrome colored columns mark the projection destinations, and each column's colored-cell count matches the height of its aligned projection.

    Rule:
    - Move the bar into the frame so its base sits one row above the frame interior bottom, and shift it horizontally until the four projection columns line up with the four colored frame columns.
    - Clear the original colored markers, then read both signals in each aligned column: the projection height and the input colored-cell count.
    - When those two counts agree, repaint exactly that many cells in the column's color starting from the top of the frame interior, filling the remainder of the shaft above the bar with grey.

    Key insight: The puzzle encodes one number twice per column. The input column length and the projection height are redundant clues for the same quantity, and that shared quantity is exactly what survives in the output.
    """
    g = np.asarray(grid, dtype=int)
    _, w = g.shape

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

    out = np.full_like(g, bg)
    out[sr0 : sr1 + 1, sc0 : sc1 + 1] = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region = out[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region[(slot_region != bg) & (slot_region != grey)] = bg

    bar_base_row = max(int(r) for r, _ in bar_pts)
    slot_interior_bottom = sr1 - 1
    target_base_row = slot_interior_bottom - 1
    dy = target_base_row - bar_base_row

    _br0, bc0, _br1, _bc1 = bar_bbox
    base_cols = sorted(int(c) for r, c in bar_pts if int(r) == bar_base_row)
    src_proj_cols = base_cols[::2] if len(base_cols) >= 4 else list(range(int(bc0), int(bc0) + 7, 2))

    col_min_row: dict[int, int] = {}
    for r, c in bar_pts:
        rr = int(r)
        cc = int(c)
        col_min_row[cc] = min(col_min_row.get(cc, rr), rr)
    src_heights = {
        int(c): int(bar_base_row - col_min_row.get(int(c), bar_base_row) + 1)
        for c in src_proj_cols
    }

    dst_cols = sorted(
        {
            int(c)
            for r, c in np.argwhere((g != bg) & (g != grey))
            if sr0 <= int(r) <= sr1 and sc0 <= int(c) <= sc1
        }
    )

    dx = None
    if src_proj_cols and len(src_proj_cols) == len(dst_cols):
        deltas = [d - s for s, d in zip(src_proj_cols, dst_cols)]
        if len(set(deltas)) == 1:
            dx = deltas[0]
    if dx is None:
        best = (-1, 0)
        dst_set = set(dst_cols)
        for cand_dx in range(-w, w + 1):
            score = sum((c + cand_dx) in dst_set for c in src_proj_cols)
            if score > best[0]:
                best = (score, cand_dx)
        dx = best[1]

    placed: list[tuple[int, int]] = []
    for r, c in bar_pts:
        rr = int(r) + dy
        cc = int(c) + dx
        if sr0 <= rr <= sr1 and sc0 <= cc <= sc1:
            out[rr, cc] = grey
            placed.append((rr, cc))

    placed_by_col: dict[int, list[int]] = {}
    for rr, cc in placed:
        placed_by_col.setdefault(int(cc), []).append(int(rr))

    dst_heights = {int(c + dx): int(h) for c, h in src_heights.items()}
    interior_top = sr0 + 1

    for col in dst_cols:
        if col not in placed_by_col or col not in dst_heights:
            continue

        rows = [r for r in range(sr0, sr1 + 1) if g[r, col] != bg and g[r, col] != grey]
        if not rows:
            continue

        color = int(g[rows[0], col])
        input_count = len(rows)
        height = dst_heights[col]
        if input_count != height:
            continue

        bar_top = min(placed_by_col[col])
        desired_bottom = bar_top - 1
        if desired_bottom < interior_top:
            continue
        if interior_top + height - 1 > desired_bottom:
            continue

        out[interior_top : desired_bottom + 1, col] = grey
        out[interior_top : interior_top + height, col] = color

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

HEIGHT_PATTERN = (1, 1, 2, 2)
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


def _build_input(heights: list[int], colors: list[int]) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    paste(grid, _bar_patch(heights), BAR_R0, BAR_INPUT_C0)
    for col, length, color, count in zip(SLOT_COLS, SLOT_LENS, colors, heights):
        _paint_input_slot_column(grid, col, length, color, count)
    return grid


def _build_output(heights: list[int], colors: list[int]) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    paste(grid, _bar_patch(heights), BAR_R0, BAR_OUTPUT_C0)
    for col, color, height in zip(SLOT_COLS, colors, heights):
        _paint_output_slot_column(grid, col, color, height)
    return grid


def _sample_params(rng: np.random.Generator) -> tuple[list[int], list[int]]:
    colors = [int(v) for v in rng.choice(PALETTE, size=len(SLOT_COLS), replace=False)]
    heights = [int(v) for v in rng.permutation(np.array(HEIGHT_PATTERN, dtype=int))]
    return heights, colors


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    heights, colors = _sample_params(rng)
    grid = _build_input(heights, colors)
    expected = _build_output(heights, colors)

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
