# Variant of 4c3d4a41
# Dimension: D2 — 源对象：杆的结构 (src)
# Change: 灰色水平底座（7格宽）带4个等间距向上凸起（位于奇数列），凸起高度1-4各自独立... → 向下凸起的杆（反转方向）

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg as _bg_color
from arc_utils import colors_present, get_bg, paste

# Design analysis for the D2 variant
# a) Concrete behavior:
#    - Keep the fixed 8x20 canvas, the 4 colored destination columns, and the 7-wide grey bar.
#    - Change only the source bar geometry: its full grey base is now the TOP row of the bar,
#      and the 4 odd-column teeth extend downward with independent heights 1..4.
#    - To make this reversal visible, the frame opening is moved from the left wall to the top edge.
#      The bar starts above-left of the frame and ends inside the frame after a downward/rightward move.
#    - In the original task, colors were stacked above each tooth because teeth pointed upward.
#      Here each aligned color stack is placed immediately below the bottom of its downward tooth.
#      Example: if a tooth has height 3 and the inserted bar base lands on row 2, that tooth occupies
#      rows 2..4, so the first colored cell for that column must start at row 5 rather than end at row 1.
# b) Complexity calibration:
#    - The bar still has four independently sampled tooth heights and the frame still has four colored
#      columns at fixed odd positions, so alignment ambiguity stays low.
#    - The opening change plus the below-the-tooth stacking makes the variant visually distinct from
#      the original even though colors, canvas size, and alignment columns are preserved.
# c) Hidden assumptions reviewed in main():
#    - The old code assumed the bar base was the lowest row (`max(r)`); now the base is the highest
#      row (`min(r)`), and projection detection must use rows BELOW the base instead of above it.
#    - The old placement logic assumed colors fit above the bar (`bar_top - 1`); now it must compute
#      free space below the bar (`bar_bottom + 1`).
#    - The frame is still the largest grey component and the bar the second-largest; the generator
#      preserves that by keeping the frame full-height and the bar isolated.
# d) Placement legality:
#    - Input bar bbox: rows 0..3, cols 1..7. Output bar bbox: rows 2..5, cols 11..17.
#    - The frame bbox remains rows 0..7, cols 9..19, with a top opening at absolute cols 11..17.
#    - The inserted bar always stays inside the frame bbox and never leaves the 8x20 canvas.
# e) Visual observability:
#    - A human sees the bar teeth point downward in every sample, sees the only frame gap on the top
#      edge, and sees colors end up below the teeth rather than above them.
#    - No color-id-based ordering or hidden metadata is used; all correspondences are positional.
# f) Visual cleanliness:
#    - The slot columns remain on cols 11,13,15,17 with blank even columns between them, so the bar
#      teeth and color stacks remain easy to read after insertion.
# g) Color inference:
#    - Colors are preserved, not remapped. Only vertical relocation changes, which is directly tied to
#      tooth height and column alignment, so the examples remain learnable from 2-4 pairs.


def _bbox(pts):
    return bbox_coords(pts if isinstance(pts, np.ndarray) else np.asarray(list(pts), dtype=int))


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _most_common(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fit a grey bar with downward teeth into a top-open grey frame, then hang the colors below it.

    Input structure:
    - bg plus a grey structural color forming two components: a large frame on the right with a top opening,
      and a smaller horizontal bar whose full base is on its top row and whose teeth extend downward.
    - Colored pixels inside the frame mark the target columns that the bar teeth must line up with.

    Rule:
    - Move the bar down into the frame so its top base row sits one cell below the frame interior top.
    - Shift it horizontally so the downward teeth align column-wise with the colored columns in the frame.
    - For each aligned column, place that column's colored pixels immediately below the bottom of the tooth.

    Key insight: The same column-matching mechanism as the original task still applies, but the bar is vertically
    inverted, so both the insertion direction and the side on which colors attach must be reversed.
    """
    g = np.asarray(grid, dtype=int)
    _h, w = g.shape

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

    (_, slot_bbox, _), (_, _bar_bbox, bar_pts) = grey_comps[0], grey_comps[1]
    sr0, sc0, sr1, sc1 = slot_bbox

    out = np.full_like(g, bg)
    out[sr0 : sr1 + 1, sc0 : sc1 + 1] = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region = out[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region[(slot_region != bg) & (slot_region != grey)] = bg

    bar_base_row = min(r for r, _ in bar_pts)
    slot_interior_top = sr0 + 1
    target_base_row = slot_interior_top + 1
    dy = target_base_row - bar_base_row

    col_max_row = {}
    for r, c in bar_pts:
        col_max_row[c] = max(col_max_row.get(c, r), r)
    src_proj_cols = sorted([c for c, rmax in col_max_row.items() if rmax > bar_base_row])

    dst_cols_set = set()
    for r, c in np.argwhere((g != bg) & (g != grey)):
        r = int(r)
        c = int(c)
        if sr0 <= r <= sr1 and sc0 <= c <= sc1:
            dst_cols_set.add(c)
    dst_cols = sorted(dst_cols_set)

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

    placed = []
    for r, c in bar_pts:
        rr = r + dy
        cc = c + dx
        if sr0 <= rr <= sr1 and sc0 <= cc <= sc1:
            out[rr, cc] = grey
            placed.append((rr, cc))

    placed_by_col = {}
    for rr, cc in placed:
        placed_by_col.setdefault(cc, []).append(rr)

    interior_bottom = sr1 - 1

    for col in dst_cols:
        if col not in placed_by_col:
            continue
        bar_bottom = max(placed_by_col[col])
        desired_top = bar_bottom + 1
        if desired_top > interior_bottom:
            continue

        rows = [r for r in range(sr0, sr1 + 1) if g[r, col] != bg and g[r, col] != grey]
        if not rows:
            continue
        colors = [int(g[r, col]) for r in rows]

        max_len = interior_bottom - desired_top + 1
        if max_len <= 0:
            continue
        if len(colors) > max_len:
            colors = colors[:max_len]

        for i, colv in enumerate(colors):
            rr = desired_top + i
            if slot_interior_top <= rr <= interior_bottom:
                out[rr, col] = colv

    return out


# --------------- generator ---------------

BG = 0
GREY = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)

H, W = 8, 20

FRAME_R0, FRAME_C0 = 0, 9
FRAME_H, FRAME_W = 8, 11
TOP_OPEN_LOCAL_COLS = tuple(range(2, 9))  # absolute cols 11..17

SLOT_COLS = [11, 13, 15, 17]
SLOT_LENS = [3, 4, 4, 3]
SLOT_TOP_ROW = 1

BAR_INPUT_R0, BAR_INPUT_C0 = 0, 1
BAR_OUTPUT_R0, BAR_OUTPUT_C0 = 2, 11
BAR_H, BAR_W = 4, 7
BAR_LOCAL_PROJ_COLS = [0, 2, 4, 6]
BAR_BASE_OUTPUT_ROW = BAR_OUTPUT_R0

HEIGHT0_RANGE = (1, 4)   # {1,2,3}; guarantees at least one visible below-bar region
HEIGHT_REST_RANGE = (1, 5)
MAX_ATTEMPTS = 400


def _frame_patch() -> np.ndarray:
    """Grey frame with bottom/side walls closed and a 7-cell opening in the top wall."""
    patch = np.zeros((FRAME_H, FRAME_W), dtype=int)
    patch[-1, :] = GREY
    patch[:, 0] = GREY
    patch[:, -1] = GREY
    patch[0, :] = GREY
    patch[0, list(TOP_OPEN_LOCAL_COLS)] = BG
    return patch


def _bar_patch(heights: list[int]) -> np.ndarray:
    """Horizontal top base with 4 downward grey teeth at local cols {0,2,4,6}."""
    patch = np.zeros((BAR_H, BAR_W), dtype=int)
    patch[0, :] = GREY
    for local_c, height in zip(BAR_LOCAL_PROJ_COLS, heights):
        patch[:height, local_c] = GREY
    return patch


def _paint_slot_column(
    grid: np.ndarray, col: int, length: int, color: int, count: int, height: int | None
) -> None:
    """Paint one slot column.

    Input mode (height=None): top (length-count) cells grey caps, bottom `count` cells the color.
    Output mode (height given): preserve the same top grey caps, then place up to (5-height) colored
    cells immediately below the inserted tooth.
    """
    cap = length - count
    grid[SLOT_TOP_ROW : SLOT_TOP_ROW + cap, col] = GREY
    if height is None:
        grid[SLOT_TOP_ROW + cap : SLOT_TOP_ROW + length, col] = color
        return

    start_row = BAR_BASE_OUTPUT_ROW + height
    visible = min(count, 7 - start_row)  # interior bottom is row 6
    if visible > 0:
        grid[start_row : start_row + visible, col] = color


def _build_grid(
    heights: list[int],
    colors: list[int],
    colored_counts: list[int],
    bar_r0: int,
    bar_c0: int,
    output_mode: bool,
) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    paste(grid, _bar_patch(heights), bar_r0, bar_c0)
    for col, length, color, count, height in zip(SLOT_COLS, SLOT_LENS, colors, colored_counts, heights):
        _paint_slot_column(grid, col, length, color, count, height if output_mode else None)
    return grid


def _sample_params(rng: np.random.Generator) -> tuple[list[int], list[int], list[int]]:
    colors = [int(v) for v in rng.choice(PALETTE, size=len(SLOT_COLS), replace=False)]
    colored_counts = [int(rng.integers(1, length + 1)) for length in SLOT_LENS]
    heights = [int(rng.integers(*HEIGHT0_RANGE))]
    heights.extend(int(rng.integers(*HEIGHT_REST_RANGE)) for _ in range(len(SLOT_COLS) - 1))
    return heights, colors, colored_counts


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    heights, colors, colored_counts = _sample_params(rng)
    grid = _build_grid(heights, colors, colored_counts, BAR_INPUT_R0, BAR_INPUT_C0, output_mode=False)
    expected = _build_grid(heights, colors, colored_counts, BAR_OUTPUT_R0, BAR_OUTPUT_C0, output_mode=True)

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
