# Combo variant of 4c3d4a41 (index 16)
# combo_spec: D1/0+D4/2+D14/2
# Name: 颜色主宰三合一

# Design analysis
# a) Concrete behavior and trace:
#    - Keep the original frame/bar grammar, but adopt the D1 layout with 5 candidate slot columns at
#      cols [10,12,14,16,18] and only 4 colored protrusions on the bar.
#    - Exactly 4 consecutive slot columns match the bar's left->right protrusion colors. The matching
#      block can be either [10,12,14,16] or [12,14,16,18]; the remaining end column is a distractor
#      that vanishes after docking.
#    - The bar keeps a grey base, while every protrusion has its own visible non-grey color above the
#      base. After docking, the matched columns are not merely restacked: every interior cell above the
#      aligned protrusion is repainted to that protrusion's color, so the original grey cap disappears.
#    - Small trace:
#         old task: use all colored slot columns, align by absolute columns, keep only as many colored
#         cells as the remaining headroom allows.
#         fused task: read the bar's 4 protrusion colors, find the unique consecutive 4-column slot block
#         with the same color order, move the bar under that block, then flood each matched column from
#         the slot top down to the cell just above the protrusion with the protrusion color.
#      The visible differences are: a distractor slot column is discarded, the chosen horizontal offset
#      can switch between 2 anchors, and matched columns become solid color bands instead of grey-capped
#      stacks.
# b) Complexity calibration:
#    - A single extra slot column is enough to make color matching decide the horizontal offset without
#      turning the puzzle into a large search problem.
#    - The match choice is binary (left block or right block), which is easy to infer from a few samples
#      but still invalidates absolute-position heuristics.
#    - Heights stay in [2,4] so every protrusion exposes at least 1 colored cell above the grey base,
#      making the color-carrying role of the protrusions directly visible.
# c) Implicit-assumption audit:
#    - "All colored slot columns are targets" no longer holds because one slot column is a distractor.
#    - "The bar is the second-largest grey component" fails because protrusions are colored; the solver
#      must detect the frame from the largest grey component and the bar from the non-background component
#      that lies strictly left of the frame.
#    - "Copy surviving colors from the slot column itself" is replaced by "paint with the matched
#      protrusion color". Under this exact combo spec the matched slot color and protrusion color are equal,
#      so D4/2 becomes a provenance change rather than a hue change: the solver still has to read the bar's
#      colors because they determine both alignment and repaint color.
#    - "Only preserve min(count, headroom) colored cells" fails because D14/2 now floods the whole
#      reachable segment above each matched protrusion, absorbing the grey cap.
# d) Placement legality:
#    - The frame remains 8x11 at cols 9..19; the 7-wide bar can dock legally at col 10 or col 12, keeping
#      its full width inside the frame interior cols 10..18.
#    - Five one-cell slot columns fit at spacing 2 with no overlap.
#    - Input placement keeps the bar at cols 1..7, safely separated from the frame.
# e) Visual observability:
#    - A human sees 5 slot colors but only 4 colored protrusions, so one frame column must be ignored.
#    - Across samples, the ignored column can be the left end or the right end, making the color-based
#      block choice visible.
#    - Generator constraints guarantee at least one matched column where the fused output paints more cells
#      than the original stack rule would, so the D14-style cap absorption is also visible.
# f) Visual cleanliness:
#    - All complexity stays in the original objects: one frame assembly and one bar assembly.
#    - The distractor is confined to an end column, which avoids clutter in the middle of the slot.
#    - Output repainting is restricted to the 4 matched columns, producing clean monochrome vertical bands.
# g) Change inferability:
#    - Slot colors are all distinct, and the bar colors are exactly one consecutive 4-color subsequence of
#      the 5 slot colors, so the matching rule has a visible structural cue.
#    - The repaint rule is also inferable because the same matched color continues from the slot top through
#      the protrusion body, turning each matched column into a single uninterrupted color stripe.

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, colors_present, components_mask, get_bg, paste
from arc_utils import get_bg as _bg_color


def _bbox(pts):
    arr = pts if isinstance(pts, np.ndarray) else np.asarray(list(pts), dtype=int)
    return bbox_coords(arr)


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _most_common(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _sorted_components(mask: np.ndarray):
    comps = [(len(pts), _bbox(pts), pts) for pts in _cc4(mask)]
    comps.sort(key=lambda t: (-t[0], t[1]))
    return comps


def _slot_info(g: np.ndarray, bbox: tuple[int, int, int, int], bg: int, grey: int):
    sr0, sc0, sr1, sc1 = bbox
    info = []
    for c in range(sc0, sc1 + 1):
        rows = [r for r in range(sr0, sr1 + 1) if int(g[r, c]) != bg and int(g[r, c]) != grey]
        if not rows:
            continue
        colors = {int(g[r, c]) for r in rows}
        if len(colors) != 1:
            continue
        info.append((int(c), int(next(iter(colors))), rows))
    info.sort(key=lambda t: t[0])
    return info


def _bar_projection_info(g: np.ndarray, pts, bg: int, grey: int):
    base_row = max(int(r) for r, _ in pts)
    cols: dict[int, list[int]] = {}
    for r, c in pts:
        cols.setdefault(int(c), []).append(int(r))

    info = []
    for c in sorted(cols):
        above_rows = [r for r in cols[c] if r < base_row]
        if not above_rows:
            continue
        colors = [int(g[r, c]) for r in above_rows if int(g[r, c]) != bg and int(g[r, c]) != grey]
        if not colors:
            continue
        info.append((int(c), _most_common(colors)))
    return base_row, info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Insert a colored-projection bar into the frame by color-matching it to one of two candidate slot blocks, then turn the matched columns into solid stripes of the protrusion colors.

    Input structure:
    - The right side contains a large grey half-open frame whose interior shows 5 colored slot columns, each column using exactly 1 non-grey color beneath a possible grey cap.
    - The left side contains 1 separate bar with a grey base and 4 upward protrusions whose visible cells above the base are colored and mutually distinct.

    Rule:
    - Move the bar upward into the frame so its base sits 1 cell above the interior floor.
    - Compare the bar's 4 protrusion colors with the frame's 5 slot-column colors and find the unique consecutive 4-column block with the same left-to-right color order; shift the bar horizontally so each protrusion docks under the same-colored slot column in that block.
    - Clear the original slot colors, keep the inserted bar, and repaint every matched slot column from the slot top down to the cell just above its aligned protrusion using that protrusion's color, eliminating the old grey cap. The unmatched end column disappears.

    Key insight: Column position alone is insufficient because the frame offers 5 candidates for only 4 protrusions; the solver must first identify the correct color-matched 4-column block, and that same protrusion-color mapping then determines the final full-column repaint.
    """
    g = np.asarray(grid, dtype=int)

    bg = _bg_color(g)
    other = g[g != bg]
    if other.size == 0:
        return g.copy()
    grey = _most_common(other)

    grey_comps = _sorted_components(g == grey)
    if not grey_comps:
        return g.copy()
    _, slot_bbox, _slot_pts = grey_comps[0]
    sr0, sc0, sr1, sc1 = slot_bbox

    fg_comps = _sorted_components(g != bg)
    bar_entry = None
    for entry in fg_comps:
        _size, bbox, _pts = entry
        _br0, bc0, _br1, bc1 = bbox
        if bc1 < sc0:
            bar_entry = entry
            break
    if bar_entry is None:
        return g.copy()
    _, _bar_bbox, bar_pts = bar_entry

    slot_info = _slot_info(g, slot_bbox, bg, grey)
    if len(slot_info) < 4:
        return g.copy()

    bar_base_row, proj_info = _bar_projection_info(g, bar_pts, bg, grey)
    if len(proj_info) != 4:
        return g.copy()

    proj_colors = [color for _col, color in proj_info]
    matched_slot_cols = None
    dx = None
    need = len(proj_info)
    for start in range(len(slot_info) - need + 1):
        window = slot_info[start : start + need]
        if [color for _col, color, _rows in window] != proj_colors:
            continue
        deltas = [dst_col - src_col for (src_col, _), (dst_col, _color, _rows) in zip(proj_info, window)]
        if len(set(deltas)) == 1:
            dx = int(deltas[0])
            matched_slot_cols = window
            break

    if dx is None or matched_slot_cols is None:
        return g.copy()

    target_base_row = (sr1 - 1) - 1
    dy = target_base_row - bar_base_row

    out = np.full_like(g, bg)
    out[sr0 : sr1 + 1, sc0 : sc1 + 1] = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region = out[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region[(slot_region != bg) & (slot_region != grey)] = bg

    placed_by_col: dict[int, list[int]] = {}
    for r, c in bar_pts:
        rr = int(r) + dy
        cc = int(c) + dx
        if sr0 <= rr <= sr1 and sc0 <= cc <= sc1:
            out[rr, cc] = int(g[int(r), int(c)])
            placed_by_col.setdefault(cc, []).append(rr)

    proj_color_by_dst = {int(src_col + dx): int(color) for src_col, color in proj_info}
    interior_top = sr0 + 1
    interior_bottom = sr1 - 1

    for col, _slot_color, _rows in matched_slot_cols:
        if col not in placed_by_col or col not in proj_color_by_dst:
            continue
        bar_top = min(placed_by_col[col])
        desired_bottom = min(bar_top - 1, interior_bottom)
        if desired_bottom < interior_top:
            continue
        out[interior_top : desired_bottom + 1, col] = proj_color_by_dst[col]

    return out


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


def _available_cells(height: int) -> int:
    return BAR_BASE_ABS_ROW - height


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


def _paint_slot_output(grid: np.ndarray, col: int, color: int, height: int) -> None:
    row_end = BAR_BASE_ABS_ROW - height
    if row_end >= SLOT_TOP_ROW:
        grid[SLOT_TOP_ROW : row_end + 1, col] = int(color)


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
    block_start: int,
) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    bar_colors = slot_colors[block_start : block_start + MATCH_BLOCK_LEN]
    bar_output_c0 = SLOT_COLS[block_start]
    paste(grid, _bar_patch(heights, bar_colors), BAR_R0, bar_output_c0)
    for col, color, height in zip(
        SLOT_COLS[block_start : block_start + MATCH_BLOCK_LEN],
        bar_colors,
        heights,
    ):
        _paint_slot_output(grid, col, color, height)
    return grid


def _sample_params(
    rng: np.random.Generator,
) -> tuple[list[int], list[int], list[int], int]:
    while True:
        slot_colors = [int(v) for v in rng.choice(PALETTE, size=len(SLOT_COLS), replace=False)]
        colored_counts = [int(rng.integers(1, length + 1)) for length in SLOT_LENS]
        heights = [int(rng.integers(*HEIGHT_RANGE)) for _ in range(MATCH_BLOCK_LEN)]
        block_start = int(rng.choice(MATCH_BLOCK_STARTS))

        matched_counts = colored_counts[block_start : block_start + MATCH_BLOCK_LEN]
        if len(set(heights)) < 2:
            continue
        if not any(count < _available_cells(height) for count, height in zip(matched_counts, heights)):
            continue
        return heights, slot_colors, colored_counts, block_start


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    heights, slot_colors, colored_counts, block_start = _sample_params(rng)
    grid = _build_input(heights, slot_colors, colored_counts, block_start)
    expected = _build_expected(heights, slot_colors, block_start)

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
