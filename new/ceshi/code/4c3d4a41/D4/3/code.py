# Variant of 4c3d4a41
# Dimension: D4 — 变换：彩色像素的重排规则 (transform)
# Change: 彩色像素在杆进入后堆叠在对应凸起顶部上方，从框架内顶部开始向下排列，数量受可用空间限制... → 彩色像素数量由凸起高度决定

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import bbox_coords, colors_present, components_mask, get_bg, get_bg as _bg_color, paste


# Design analysis for the D4 variant
# a) Concrete behavior:
#    - Input still contains two grey components: a right-side frame and a left-side bar with four projections.
#    - Colored slot markers remain one monochrome column per destination column inside the frame.
#    - Output still moves the bar into the frame and aligns projection columns with colored columns.
#    - The changed step is the recoloring above each aligned projection:
#      old local logic depended on the input colored count in that frame column;
#      new local logic depends only on the projection height.
#      Example: if a column shows one colored pixel in the input but the aligned projection has height 2,
#      the output must show exactly two pixels of that color at the top of the shaft.
# b) Complexity calibration:
#    - Keeping four distinct destination columns preserves the original one-to-one alignment problem.
#    - Heights are restricted to {1, 2}; this is the minimal compatible change that makes
#      "output count = height" physically possible above the bar without overlap in the fixed 8x20 geometry.
#    - Each generated example contains both heights, plus at least one expansion case (input count 1 -> output count 2)
#      and at least three columns where input count != output count, so the new rule is visible.
# c) Hidden-assumption audit on the original solver:
#    - The original `rows` extraction assumed the number of output colored cells should come from the input stack length.
#      That assumption is invalid here; we now use only the column color identity from the input.
#    - The original clipping by `max_len` was harmless when output count was input-driven, but would destroy the new
#      "strictly equals height" rule. The generator now guarantees heights that fit, so the solver can paint exact counts.
#    - Preserving only pre-existing grey caps made output appearance depend on input cap length. That becomes noisy once
#      count is decoupled from input quantity, so the solver explicitly rebuilds the grey shaft from the top down to the bar.
# d) Placement legality:
#    - Frame geometry and translation remain unchanged, so bar placement is still in-bounds.
#    - Heights {1, 2} imply available cells above a projection are 4 or 3, enough for exact output counts.
#    - Slot columns stay at cols 11, 13, 15, 17 and remain pairwise separated by one empty column.
# e) Visual observability:
#    - A human sees repeated local correspondences: short projection -> 1 color cell, taller projection -> 2 color cells.
#    - Because input colored counts vary independently and often disagree with the output counts, the counting rule is exposed.
# f) Visual cleanliness:
#    - Output columns are rebuilt as a clean top color segment followed by grey cells down to the bar, avoiding broken stacks.
#    - No extra objects or overlays are introduced; only the per-column count rule changes.
# g) Color-rule audit:
#    - No sorting or mapping uses color ids. Each slot column keeps its own input color; only the quantity changes.
# 9) New solver assumptions checked against the generator:
#    - Largest grey component is the frame and second largest is the bar: guaranteed by fixed geometry.
#    - Exactly four colored destination columns exist and correspond to the four projection columns: guaranteed by generation.
#    - Each destination column is monochrome in the input: guaranteed by contiguous single-color stacks.
#    - Projection heights fit the exact-count rule above the bar: guaranteed by heights in {1, 2}.


def _bbox(pts):
    return bbox_coords(pts if isinstance(pts, np.ndarray) else np.asarray(list(pts), dtype=int))


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _most_common(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide a grey bar into a grey frame, then rebuild each colored stack using projection height.

    Input structure:
    - The most common color is background; the main non-background color is the grey structure.
    - Grey forms two disconnected parts: a rectangular frame on the right and a horizontal bar with four upward projections on the left.
    - Inside the frame, four monochrome colored columns indicate which frame columns the bar projections must align to.

    Rule:
    - Move the bar into the frame so its base sits one row above the frame interior bottom, and shift it horizontally to align the projections with the colored columns.
    - Clear the original colored markers and rebuild each aligned shaft above the bar.
    - In each aligned column, keep the input column's color but paint exactly as many colored cells as the aligned projection's height, starting from the top of the frame interior.

    Key insight: The colored columns do not preserve their original counts; they serve as color labels, while the projection heights dictate the final tally.
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
        r = int(r)
        c = int(c)
        col_min_row[c] = min(col_min_row.get(c, r), r)
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

        bar_top = min(placed_by_col[col])
        desired_bottom = bar_top - 1
        if desired_bottom < interior_top:
            continue

        height = dst_heights[col]
        # The generator keeps heights in {1, 2}, so exact painting is always feasible.
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

HEIGHT_VALUES = (1, 2)
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


def _build_input(heights: list[int], colors: list[int], colored_counts: list[int]) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    paste(grid, _bar_patch(heights), BAR_R0, BAR_INPUT_C0)
    for col, length, color, count in zip(SLOT_COLS, SLOT_LENS, colors, colored_counts):
        _paint_input_slot_column(grid, col, length, color, count)
    return grid


def _build_output(heights: list[int], colors: list[int]) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    paste(grid, _bar_patch(heights), BAR_R0, BAR_OUTPUT_C0)
    for col, color, height in zip(SLOT_COLS, colors, heights):
        _paint_output_slot_column(grid, col, color, height)
    return grid


def _variant_metrics(heights: list[int], colored_counts: list[int]) -> dict[str, int]:
    diffs = sum(int(h != n) for h, n in zip(heights, colored_counts))
    expansions = sum(int(h > n) for h, n in zip(heights, colored_counts))
    reductions = sum(int(h < n) for h, n in zip(heights, colored_counts))
    return {"different_cols": diffs, "expansions": expansions, "reductions": reductions}


def _sample_params(rng: np.random.Generator) -> tuple[list[int], list[int], list[int]]:
    colors = [int(v) for v in rng.choice(PALETTE, size=len(SLOT_COLS), replace=False)]

    while True:
        heights = [int(rng.choice(HEIGHT_VALUES)) for _ in SLOT_COLS]
        if 1 not in heights or 2 not in heights:
            continue

        colored_counts = [int(rng.integers(1, length + 1)) for length in SLOT_LENS]
        metrics = _variant_metrics(heights, colored_counts)
        if metrics["different_cols"] >= 3 and metrics["expansions"] >= 1:
            return heights, colors, colored_counts


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    heights, colors, colored_counts = _sample_params(rng)
    grid = _build_input(heights, colors, colored_counts)
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
