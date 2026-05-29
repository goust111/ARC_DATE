# Variant of 4c3d4a41
# Dimension: D1 — 匹配条件：凸起-列对齐方式 (match_cond)
# Change: 杆的凸起列与框架内彩色列按绝对列位置一一对齐(projection_slot_alignment + column_alignment) → 颜色匹配对齐

# Design analysis
# a) Concrete behavior and trace:
#    - Keep the original 8x20 canvas, the same right-side grey frame, the same 7-cell bar base,
#      and the same "colors stack above the inserted bar" end effect.
#    - To make color matching observable without changing the action from translation to some
#      unrelated transform, the slot now contains 5 evenly spaced colored columns at cols
#      [10, 12, 14, 16, 18]. The bar still has 4 projections.
#    - Exactly 4 consecutive slot columns match the bar colors in left->right order; this block is
#      either the left block [10,12,14,16] or the right block [12,14,16,18]. The remaining end
#      column is a color distractor that has no partner on the bar and disappears in the output.
#    - Small trace:
#         old logic: collect every colored slot column -> brute-force best overlap among 5 cols ->
#                    tie between dx=9 and dx=11 -> first best dx=9 is chosen, so the bar always
#                    enters the left block.
#         new logic: read the 4 projection colors on the bar -> find the consecutive 4-column slot
#                    block with the same color order -> choose dx from that matched block.
#      The difference is therefore visible exactly when the matching block is the right one.
# b) Complexity calibration:
#    - One extra slot column is enough to force a real color-based choice while keeping the visual
#      grammar simple: 5 candidate slot columns, 4 colored bar projections, 1 unmatched color.
#    - The extra freedom is only binary (left block or right block), so examples stay learnable.
# c) Old-assumption audit:
#    - The original solver assumes "all colored slot columns are the targets" and sorts them by
#      absolute column index. That no longer holds because one colored slot column is intentionally
#      unmatched.
#    - The original solver also treats the bar as a grey-only component. Here the projections are
#      recolored, so the bar must be found as a non-background connected component, not as a grey
#      connected component.
#    - Heights can no longer be 1: with colored projections, each projection must have at least one
#      above-base colored cell, so sampling uses heights in [2,4].
# d) Placement legality:
#    - The two legal output bar anchors are cols 10 and 12, both keeping the 7-wide bar entirely
#      inside the frame interior cols 10..18.
#    - The 5 slot columns fit without overlap because they are one cell wide and spaced by 2.
# e) Visual observability:
#    - The bar now visibly carries 4 colors, and the frame visibly carries 5 colored columns; one
#      frame color is absent on the bar, so the solver must notice the subset/order relation.
#    - Using only a left/right block choice avoids hidden numeric rules or color-id ordering.
# f) Visual cleanliness:
#    - All objects remain separated into the original two large assemblies (frame assembly and bar
#      assembly). The distractor column is an end column, so it does not create clutter in the
#      middle of the slot.
# g) Color-inference constraints:
#    - Slot colors are all distinct. Bar colors are the 4-color consecutive subsequence of those
#      slot colors. This gives enough structure for humans to infer which block is active.

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg as _bg_color
from arc_utils import colors_present, get_bg, paste


def _bbox(pts):
    return bbox_coords(pts if isinstance(pts, np.ndarray) else np.asarray(list(pts), dtype=int))


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _most_common(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Insert a colored-projection bar into a grey frame by matching projection colors to slot-column colors.

    Input structure:
    - bg plus two disconnected non-bg assemblies: a right-side grey frame containing colored slot columns, and a left-side bar with a grey base and four upward colored projections.
    - The slot contains five colored columns, but only four of them correspond to the bar; the unmatched end column is a distractor.

    Rule:
    - Move the bar upward into the frame so its base sits one cell above the slot's interior bottom, as before.
    - Determine which consecutive four slot columns match the bar by comparing projection colors from left to right; shift the bar horizontally so each colored projection lands under the same-colored slot column.
    - Keep the frame, remove the original bar, and restack each matched slot column's colored cells immediately above the top of its aligned projection.

    Key insight: Absolute slot-column position is no longer sufficient because the slot contains one extra colored column; the correct horizontal offset is the one whose color sequence matches the bar's projection colors.
    """
    g = np.asarray(grid, dtype=int)
    _h, w = g.shape

    bg = _bg_color(g)
    other = g[g != bg]
    if other.size == 0:
        return g.copy()
    grey = _most_common(other)

    comps = [(len(pts), _bbox(pts), pts) for pts in _cc4(g != bg)]
    comps.sort(key=lambda t: (-t[0], t[1]))
    if len(comps) < 2:
        return g.copy()

    (_, slot_bbox, _slot_pts), (_, _bar_bbox, bar_pts) = comps[0], comps[1]
    sr0, sc0, sr1, sc1 = slot_bbox

    out = np.full_like(g, bg)
    out[sr0 : sr1 + 1, sc0 : sc1 + 1] = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region = out[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region[(slot_region != bg) & (slot_region != grey)] = bg

    bar_base_row = max(int(r) for r, _ in bar_pts)
    target_base_row = (sr1 - 1) - 1
    dy = target_base_row - bar_base_row

    bar_cols = {}
    for r, c in bar_pts:
        bar_cols.setdefault(int(c), []).append(int(r))

    proj_info = []
    for c in sorted(bar_cols):
        rows = sorted(bar_cols[c])
        if rows[0] >= bar_base_row:
            continue
        proj_colors = sorted(
            {
                int(g[r, c])
                for r in rows
                if r < bar_base_row and int(g[r, c]) != bg and int(g[r, c]) != grey
            }
        )
        if not proj_colors:
            continue
        proj_info.append((int(c), int(proj_colors[0])))

    slot_cols = []
    for c in range(sc0, sc1 + 1):
        rows = [r for r in range(sr0, sr1 + 1) if int(g[r, c]) != bg and int(g[r, c]) != grey]
        if not rows:
            continue
        colors = {int(g[r, c]) for r in rows}
        if len(colors) != 1:
            continue
        slot_cols.append((int(c), int(next(iter(colors))), rows))
    slot_cols.sort(key=lambda t: t[0])

    proj_colors = [color for _, color in proj_info]
    matched_slot_cols = []
    dx = None

    if proj_info and len(slot_cols) >= len(proj_info):
        need = len(proj_info)
        for start in range(len(slot_cols) - need + 1):
            window = slot_cols[start : start + need]
            if [color for _, color, _ in window] != proj_colors:
                continue
            deltas = [dst_col - src_col for (src_col, _), (dst_col, _, _) in zip(proj_info, window)]
            if len(set(deltas)) == 1:
                dx = int(deltas[0])
                matched_slot_cols = window
                break

    if dx is None:
        by_color = {color: (col, rows) for col, color, rows in slot_cols}
        if proj_info and all(color in by_color for _, color in proj_info):
            deltas = [by_color[color][0] - src_col for src_col, color in proj_info]
            if len(set(deltas)) == 1:
                dx = int(deltas[0])
                matched_slot_cols = [(by_color[color][0], color, by_color[color][1]) for _, color in proj_info]

    if dx is None:
        dst_cols = [col for col, _color, _rows in slot_cols]
        src_proj_cols = [col for col, _color in proj_info]
        best = (-1, 0)
        dst_set = set(dst_cols)
        for cand_dx in range(-w, w + 1):
            score = sum((c + cand_dx) in dst_set for c in src_proj_cols)
            if score > best[0]:
                best = (score, cand_dx)
        dx = int(best[1])
        matched_slot_cols = [info for info in slot_cols if info[0] in {c + dx for c in src_proj_cols}]

    placed = []
    for r, c in bar_pts:
        rr = int(r + dy)
        cc = int(c + dx)
        if sr0 <= rr <= sr1 and sc0 <= cc <= sc1:
            out[rr, cc] = int(g[r, c])
            placed.append((rr, cc))

    placed_by_col = {}
    for rr, cc in placed:
        placed_by_col.setdefault(int(cc), []).append(int(rr))

    interior_top = sr0 + 1
    interior_bottom = sr1 - 1

    for col, _color, rows in matched_slot_cols:
        if col not in placed_by_col:
            continue
        bar_top = min(placed_by_col[col])
        desired_bottom = bar_top - 1
        if desired_bottom < interior_top:
            continue

        colors = [int(g[r, col]) for r in rows]
        max_len = desired_bottom - interior_top + 1
        if max_len <= 0:
            continue
        if len(colors) > max_len:
            colors = colors[-max_len:]

        start_row = max(desired_bottom - len(colors) + 1, interior_top)
        for i, colv in enumerate(colors):
            rr = start_row + i
            if interior_top <= rr <= interior_bottom:
                out[rr, col] = int(colv)

    return out


# --------------- generator ---------------

BG = 0
GREY = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)

H, W = 8, 20

FRAME_R0, FRAME_C0 = 0, 9
FRAME_H, FRAME_W = 8, 11
LEFT_WALL_OPEN_ROWS = (4, 5)

SLOT_COLS = [10, 12, 14, 16, 18]
SLOT_LENS = [3, 4, 4, 4, 3]
SLOT_TOP_ROW = 1

MATCH_BLOCK_STARTS = [0, 1]
MATCH_BLOCK_LEN = 4

BAR_INPUT_C0 = 1
BAR_R0 = 2
BAR_H, BAR_W = 4, 7
BAR_LOCAL_PROJ_COLS = [0, 2, 4, 6]
BAR_BASE_ABS_ROW = BAR_R0 + BAR_H - 1

HEIGHT_RANGE = (2, 5)
MAX_ATTEMPTS = 300


def _frame_patch() -> np.ndarray:
    patch = np.zeros((FRAME_H, FRAME_W), dtype=int)
    patch[[0, -1], :] = GREY
    patch[:, -1] = GREY
    patch[:, 0] = GREY
    patch[list(LEFT_WALL_OPEN_ROWS), 0] = BG
    return patch


def _bar_patch(heights: list[int], proj_colors: list[int]) -> np.ndarray:
    patch = np.zeros((BAR_H, BAR_W), dtype=int)
    patch[-1, :] = GREY
    for local_c, height, color in zip(BAR_LOCAL_PROJ_COLS, heights, proj_colors):
        patch[BAR_H - height : BAR_H - 1, local_c] = int(color)
    return patch


def _paint_slot_input(grid: np.ndarray, col: int, length: int, color: int, count: int) -> None:
    cap = length - count
    grid[SLOT_TOP_ROW : SLOT_TOP_ROW + cap, col] = GREY
    grid[SLOT_TOP_ROW + cap : SLOT_TOP_ROW + length, col] = int(color)


def _paint_slot_output(
    grid: np.ndarray,
    col: int,
    color: int,
    count: int,
    height: int,
) -> None:
    visible = min(count, BAR_BASE_ABS_ROW - height)
    if visible <= 0:
        return
    row_end = BAR_BASE_ABS_ROW - height
    grid[row_end - visible + 1 : row_end + 1, col] = int(color)


def _build_input(
    heights: list[int],
    slot_colors: list[int],
    colored_counts: list[int],
    block_start: int,
) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    for col, length, color, count in zip(SLOT_COLS, SLOT_LENS, slot_colors, colored_counts):
        _paint_slot_input(grid, col, length, color, count)
    bar_colors = slot_colors[block_start : block_start + MATCH_BLOCK_LEN]
    paste(grid, _bar_patch(heights, bar_colors), BAR_R0, BAR_INPUT_C0)
    return grid


def _build_expected(
    heights: list[int],
    slot_colors: list[int],
    colored_counts: list[int],
    block_start: int,
) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)

    for col, length, count in zip(SLOT_COLS, SLOT_LENS, colored_counts):
        cap = length - count
        grid[SLOT_TOP_ROW : SLOT_TOP_ROW + cap, col] = GREY

    bar_colors = slot_colors[block_start : block_start + MATCH_BLOCK_LEN]
    bar_output_c0 = SLOT_COLS[block_start]
    paste(grid, _bar_patch(heights, bar_colors), BAR_R0, bar_output_c0)

    matched_cols = SLOT_COLS[block_start : block_start + MATCH_BLOCK_LEN]
    matched_counts = colored_counts[block_start : block_start + MATCH_BLOCK_LEN]
    for col, color, count, height in zip(matched_cols, bar_colors, matched_counts, heights):
        _paint_slot_output(grid, col, color, count, height)

    return grid


def _sample_params(rng: np.random.Generator) -> tuple[list[int], list[int], list[int], int]:
    slot_colors = [int(v) for v in rng.choice(PALETTE, size=len(SLOT_COLS), replace=False)]
    colored_counts = [int(rng.integers(1, length + 1)) for length in SLOT_LENS]
    heights = [int(rng.integers(*HEIGHT_RANGE)) for _ in BAR_LOCAL_PROJ_COLS]
    if all(height == 4 for height in heights):
        heights[int(rng.integers(0, len(heights)))] = 3
    block_start = int(rng.choice(MATCH_BLOCK_STARTS))
    return heights, slot_colors, colored_counts, block_start


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    heights, slot_colors, colored_counts, block_start = _sample_params(rng)
    grid = _build_input(heights, slot_colors, colored_counts, block_start)
    expected = _build_expected(heights, slot_colors, colored_counts, block_start)

    if get_bg(grid) != BG:
        return None
    if sorted(colors_present(grid, exclude=[BG, GREY])) != sorted(slot_colors):
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
