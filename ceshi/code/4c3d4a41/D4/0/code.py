# Variant of 4c3d4a41
# Dimension: D4 — 变换：彩色像素的重排规则 (transform)
# Change: 彩色像素在杆进入后堆叠在对应凸起顶部上方，从框架内顶部开始向下排列，数量受可用空间限制... → 彩色像素堆叠在凸起下方

"""
design_analysis:
  variant_goal:
    keep_fixed:
      - 8x20 canvas, right-hand grey frame, left grey bar, 4 colored slot columns.
      - Horizontal alignment logic: the bar projections still line up with the colored columns.
      - Input column encoding: each slot column still contains a grey cap above a monochrome colored run.
    change_core:
      - The colored run no longer ends up above the inserted bar.
      - It must appear below the inserted bar base, stacked from the frame bottom upward.

  concrete_behavior:
    objects:
      - The frame stays at rows 0..7, cols 9..19, with interior rows 1..6.
      - The bar still has a full 7-cell base and 4 upward projections at odd columns.
      - The target colored columns remain cols 11, 13, 15, 17.
    output_layout:
      - The bar is moved into the frame and aligned horizontally exactly as before.
      - To make "stack below the bar" observable, the bar is also lifted so its tallest occupied cell
        touches the frame interior ceiling at row 1.
      - The colored cells in each target column are then drawn below the bar base, bottom-aligned at row 6.
      - If a column has more colored cells than the room below the base, only the visible prefix survives.

  behavior_trace:
    example:
      input_column:
        length: 4
        grey_cap: 1
        colored_count: 3
      original_task:
        - Bar base is anchored low (row 5), so the usable region is above each projection.
        - The 3 colored cells are shown immediately above that projection.
      this_variant:
        - Bar is anchored high enough to leave room under the base.
        - If the inserted base lands at row 3, the usable region is rows 4..6.
        - The 3 colored cells now fill rows 4, 5, 6 instead of rows above the projection.
      difference:
        - Same source count and same column matching.
        - Opposite vertical side of the bar, opposite stack anchor, different truncation space.

  complexity_calibration:
    why_not_keep_old_vertical_anchor:
      - If the original low anchor were kept, only row 6 would exist below the base.
      - That would collapse most outputs to at most one visible colored pixel per column.
      - The D4 change would be technically present but visually too weak.
    supporting_adjustment:
      - Raise the bar by top-aligning its tallest occupied cell to row 1.
      - This preserves the task's object vocabulary while making the new transform readable.
    controlled_freedom:
      - Heights remain in a small discrete range 1..4.
      - Colors remain 4 distinct column colors.
      - Target columns and frame geometry stay fixed, so matching remains unambiguous.

  implicit_assumption_audit:
    original_main_lines_affected:
      - Original dy used slot-interior-bottom minus bar-base; that assumes colors live above the bar.
      - Original desired_bottom = bar_top - 1 assumes the free space is above the projection tip.
      - Original max_len measured interior_top..desired_bottom, which is invalid once stacking moves below.
      - Original start_row placed colors top-anchored inside the available region.
    replacement_logic:
      - dy is now interior_top - bar_top, so the tallest occupied bar cell reaches the ceiling.
      - Available space is measured from base+1 down to interior_bottom.
      - Colors are bottom-aligned in that region.
    harmless_old_patterns_checked:
      - Clearing slot colors before repainting is still safe because the new output reuses the frame area only.
      - Horizontal dx inference is still valid because the column-matching mechanism is unchanged.

  placement_legality:
    bar_fit:
      - Top alignment guarantees every occupied bar cell lies inside rows 1..6 because the bar's tallest
        projection defines the shift.
      - The base then lands at row 2..4, so there are 2..4 rows available below it.
    overlap:
      - Grey-over-grey overlaps inside the frame are benign.
      - Colored cells are only painted strictly below the base, so they never overwrite the bar.
    generator_solver_contract:
      - Generator always uses four target columns matching the bar's odd-column spacing.
      - Generator rejects samples where every visible stack height would be 1, improving readability.

  visual_observability:
    cues_humans_can_use:
      - The bar clearly moves into the frame and sits higher than in the original task.
      - Colored cells collect under the bar and hug the frame bottom.
      - Multiple examples vary both the free space below the base and the visible stack heights.
    no_hidden_rule:
      - No sorting by color id.
      - No color remapping; each column keeps its original color.

  visual_cleanliness:
    - The frame, bar, and colored columns remain spatially separated by role.
    - Only one transform happens inside each slot column: the colored run relocates downward.
    - Even interior columns stay empty, so the four active columns remain easy to parse.

  new_main_assumptions_and_generator_guards:
    assumptions:
      - There are at least two grey connected components: the frame and the bar.
      - The bar projections are horizontally compatible with the slot columns.
      - Colored pixels inside the frame occur only in the four target columns.
    guaranteed_by_generator:
      - Yes: the generator constructs exactly that layout and validates main(input) against the expected grid.
"""

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
    Slide a grey toothed bar into a grey frame, hang it near the top, and collect the colors below it.

    Input structure:
    - The background is the dominant color; the main structural color is grey, forming a large right-hand frame and a smaller left-hand bar with upward projections.
    - Inside the frame, four colored columns mark the destination columns that the bar projections must align with.

    Rule:
    - Move the bar into the frame so its projections line up horizontally with the colored columns.
    - Raise the bar so its highest occupied grey cell touches the frame's interior ceiling.
    - For each aligned column, take the original colored pixels from that column and stack them below the bar base, bottom-aligned within the frame interior and clipped by the free space there.

    Key insight:
    - The column matching stays the same as the original task, but the vertical packing is mirrored: the bar now hangs high, and the colored material settles underneath it instead of above it.
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

    (_, slot_bbox, _), (_, _, bar_pts) = grey_comps[0], grey_comps[1]
    sr0, sc0, sr1, sc1 = slot_bbox
    interior_top = sr0 + 1
    interior_bottom = sr1 - 1

    out = np.full_like(g, bg)
    out[sr0 : sr1 + 1, sc0 : sc1 + 1] = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region = out[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region[(slot_region != bg) & (slot_region != grey)] = bg

    bar_base_row = max(r for r, _ in bar_pts)
    bar_top_row = min(r for r, _ in bar_pts)
    dy = interior_top - bar_top_row

    col_min_row = {}
    for r, c in bar_pts:
        col_min_row[c] = min(col_min_row.get(c, r), r)
    src_proj_cols = sorted(c for c, rmin in col_min_row.items() if rmin < bar_base_row)

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

    if not placed:
        return out

    placed_cols = {cc for _, cc in placed}
    placed_base_row = max(rr for rr, _ in placed)
    desired_top = placed_base_row + 1
    if desired_top > interior_bottom:
        return out

    for col in dst_cols:
        if col not in placed_cols:
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

        start_row = interior_bottom - len(colors) + 1
        for i, color in enumerate(colors):
            rr = start_row + i
            if desired_top <= rr <= interior_bottom:
                out[rr, col] = color

    return out


BG = 0
GREY = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)

H, W = 8, 20

FRAME_R0, FRAME_C0 = 0, 9
FRAME_H, FRAME_W = 8, 11
LEFT_WALL_OPEN_ROWS = (4, 5)
INTERIOR_TOP = FRAME_R0 + 1
INTERIOR_BOTTOM = FRAME_R0 + FRAME_H - 2

SLOT_COLS = [11, 13, 15, 17]
SLOT_LENS = [3, 4, 4, 3]
SLOT_TOP_ROW = 1

BAR_INPUT_C0 = 1
BAR_OUTPUT_C0 = 11
BAR_BASE_INPUT_ROW = 5
BAR_LOCAL_PROJ_COLS = [0, 2, 4, 6]
BAR_WIDTH = 7

HEIGHT0_RANGE = (2, 5)
HEIGHT_REST_RANGE = (1, 5)
MAX_ATTEMPTS = 300


def _frame_patch() -> np.ndarray:
    patch = np.zeros((FRAME_H, FRAME_W), dtype=int)
    patch[[0, -1], :] = GREY
    patch[:, -1] = GREY
    patch[:, 0] = GREY
    patch[list(LEFT_WALL_OPEN_ROWS), 0] = BG
    return patch


def _bar_points(heights: list[int], left_c0: int, base_row: int) -> list[tuple[int, int]]:
    pts = {(base_row, left_c0 + dc) for dc in range(BAR_WIDTH)}
    for local_c, height in zip(BAR_LOCAL_PROJ_COLS, heights):
        abs_c = left_c0 + local_c
        for r in range(base_row - height + 1, base_row + 1):
            pts.add((r, abs_c))
    return sorted(pts)


def _paint_input_slot_column(grid: np.ndarray, col: int, length: int, color: int, count: int) -> None:
    cap = length - count
    grid[SLOT_TOP_ROW : SLOT_TOP_ROW + cap, col] = GREY
    grid[SLOT_TOP_ROW + cap : SLOT_TOP_ROW + length, col] = color


def _build_input_grid(heights: list[int], colors: list[int], colored_counts: list[int]) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    for r, c in _bar_points(heights, BAR_INPUT_C0, BAR_BASE_INPUT_ROW):
        grid[r, c] = GREY
    for col, length, color, count in zip(SLOT_COLS, SLOT_LENS, colors, colored_counts):
        _paint_input_slot_column(grid, col, length, color, count)
    return grid


def _build_expected_grid(input_grid: np.ndarray, heights: list[int], colors: list[int],
                         colored_counts: list[int]) -> np.ndarray:
    out = np.full_like(input_grid, BG)
    sr0, sc0 = FRAME_R0, FRAME_C0
    sr1, sc1 = FRAME_R0 + FRAME_H - 1, FRAME_C0 + FRAME_W - 1
    out[sr0 : sr1 + 1, sc0 : sc1 + 1] = input_grid[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region = out[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region[(slot_region != BG) & (slot_region != GREY)] = BG

    target_base_row = INTERIOR_TOP + max(heights) - 1
    for r, c in _bar_points(heights, BAR_OUTPUT_C0, target_base_row):
        out[r, c] = GREY

    max_visible = INTERIOR_BOTTOM - target_base_row
    if max_visible <= 0:
        return out

    for col, color, count in zip(SLOT_COLS, colors, colored_counts):
        visible = min(count, max_visible)
        if visible <= 0:
            continue
        start_row = INTERIOR_BOTTOM - visible + 1
        out[start_row : INTERIOR_BOTTOM + 1, col] = color

    return out


def _sample_params(rng: np.random.Generator) -> tuple[list[int], list[int], list[int]]:
    colors = [int(v) for v in rng.choice(PALETTE, size=len(SLOT_COLS), replace=False)]
    colored_counts = [int(rng.integers(1, length + 1)) for length in SLOT_LENS]
    heights = [int(rng.integers(*HEIGHT0_RANGE))]
    heights.extend(int(rng.integers(*HEIGHT_REST_RANGE)) for _ in range(len(SLOT_COLS) - 1))
    return heights, colors, colored_counts


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    heights, colors, colored_counts = _sample_params(rng)
    input_grid = _build_input_grid(heights, colors, colored_counts)
    expected = _build_expected_grid(input_grid, heights, colors, colored_counts)

    target_base_row = INTERIOR_TOP + max(heights) - 1
    max_visible = INTERIOR_BOTTOM - target_base_row
    visible_counts = [min(count, max_visible) for count in colored_counts]
    if max(visible_counts) < 2:
        return None
    if get_bg(input_grid) != BG:
        return None
    if sorted(colors_present(input_grid, exclude=[BG, GREY])) != sorted(colors):
        return None
    if not np.array_equal(main(input_grid), expected):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_ATTEMPTS} attempts")
