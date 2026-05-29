# Variant of 4c3d4a41
# Dimension: D11 — 框架开口与进入方向 (meta)
# Change: 框架左墙在行4-5有2格开口，杆从左侧水平进入... → 多个开口暗示多步进入

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg as _bg_color
from arc_utils import colors_present, get_bg, paste

# Design analysis for the D11 variant
# ----------------------------------
# a) Concrete behavior and trace
#    - The frame keeps the original right-hand position and the same four internal target columns,
#      but now exposes two simultaneous openings:
#        1. left-wall gap at absolute rows 4-5, col 9
#        2. bottom-wall gap at absolute row 7, cols 11-17
#    - The bar stays a 7-wide horizontal base with 4 keyed columns at spacing 2, but it now has
#      two visible orientations:
#        * "up" branch: projections extend upward from the base, bar starts to the left of the frame.
#        * "down" branch: projections extend downward from the base, bar starts below the frame.
#    - The colored slot markers remain vertical columns at frame cols 11,13,15,17.
#    - Output behavior:
#        * up branch: align keyed columns, place the bar low inside the frame, stack colors above.
#        * down branch: align keyed columns, place the bar high inside the frame, stack colors below.
#    - Small trace:
#        * Original: bar rows 2-5, base row 5, always shifts right, colors always finish above.
#        * Variant/down: bar rows 8-11, base row 8 (top row of the patch), shifts up+right into
#          row 1, and the same column colors finish below the downward projections.
#        * Difference: entry side and color-stacking side are no longer constants; they are chosen
#          from the bar's protrusion direction.
#
# b) Complexity calibration
#    - Only two branches are used ("up" and "down"), keeping the decision simple enough to infer.
#    - The frame always shows both openings, so the branch difference is visually present.
#    - The four slot columns and the four keyed bar columns are unchanged, preserving the original
#      matching problem and avoiding overloading the task with extra freedoms.
#
# c) Implicit-assumption audit against the original solver
#    - The original code assumes the bar base is near the bottom of the frame and colors must stack
#      above it. That fails for downward projections, so target row and stacking direction must now
#      depend on orientation.
#    - The original code treats the max bar row as the base row. That fails for the downward branch,
#      where the full-width base is the top row of the component. This variant explicitly finds the
#      unique widest row as the base.
#    - The original code ignores openings entirely. That is harmless in the seed because only one
#      entrance exists, but wrong here because the correct target vertical placement depends on which
#      opening the bar is using.
#
# d) Placement legality
#    - The canvas is raised from 8x20 to 13x20 so a bottom-entry bar can exist outside the frame
#      with a blank separator row, keeping frame and bar as distinct grey components.
#    - The frame keeps size 8x11 at rows 0-7, cols 9-19; the extra lower rows are empty except for
#      the bottom-entry bar.
#    - Left-entry bars live at rows 2-5, cols 0-8 depending on sampled offset; bottom-entry bars
#      live at rows 9-12, cols 7-15 depending on sampled offset. These placements never overlap the
#      frame, never touch it by 4-connectivity, and always stay in-bounds.
#
# e) Visual observability
#    - Humans can see both openings in every sample.
#    - The bar's protrusions visibly point either up or down, and the output correspondingly places
#      the bar low or high and stacks the colors on the matching side. This is observable in 2-3
#      examples without using hidden ids or ordering tricks.
#
# f) Visual cleanliness
#    - The frame, bar, and color markers occupy separate regions in the input.
#    - Inside the output frame, only one grey object (the placed bar) and one monochrome stack per
#      keyed column remain, so the final state stays clean rather than over-layered.
#
# g) Color predictability
#    - Each target column keeps a single repeated color. No color remapping is introduced.
#    - Only the side of the final stack changes; color identity remains tied to column position.
#
# h) Quantitative anti-degeneration target for later validation
#    - Branch distribution should include both "up" and "down" samples.
#    - On "down" samples, the new solver's output must differ from a counterfactual "always use the
#      original left-entry/upward rule" transform; otherwise the D11 change would be fake.
#
# i) New solver assumptions that the generator must guarantee
#    - The bar has exactly one full-width grey base row, and at least one keyed column extends away
#      from that base so the facing direction is visible.
#    - The frame is still the largest grey component and the bar the second largest one.
#    - The colored targets remain inside the frame bbox and use the same four keyed columns.


def _bbox(pts):
    return bbox_coords(pts if isinstance(pts, np.ndarray) else np.asarray(list(pts), dtype=int))


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _most_common(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _bar_mode_and_base(bar_pts: np.ndarray) -> tuple[str, int]:
    row_to_cols: dict[int, set[int]] = {}
    for r, c in bar_pts:
        row_to_cols.setdefault(int(r), set()).add(int(c))
    base_row = max(row_to_cols, key=lambda r: (len(row_to_cols[r]), -r))
    has_above = any(int(r) < base_row for r, _ in bar_pts)
    has_below = any(int(r) > base_row for r, _ in bar_pts)
    if has_above == has_below:
        raise ValueError("bar orientation is ambiguous")
    return ("up" if has_above else "down"), int(base_row)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fit a grey keyed bar into a grey frame that now exposes two possible entrances.

    Input structure:
    - bg and a grey structural color form a large right-hand frame with both a left-wall opening and
      a bottom-wall opening, plus a separate grey bar whose protrusions point either upward or downward.
    - Colored pixels inside the frame still identify the destination keyed columns.

    Rule:
    - Align the bar's protrusion columns with the colored columns inside the frame.
    - If the protrusions point upward, the bar uses the left opening, settles low in the frame, and
      the column colors stack immediately above the protrusions.
    - If the protrusions point downward, the bar uses the bottom opening, settles high in the frame,
      and the column colors stack immediately below the protrusions.

    Key insight: The extra opening is not decorative; the bar's facing direction decides which entrance
    is valid and which side of the bar will receive the stacked colors after alignment.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

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

    mode, bar_base_row = _bar_mode_and_base(bar_pts)
    interior_top = sr0 + 1
    interior_bottom = sr1 - 1
    target_base_row = interior_bottom - 1 if mode == "up" else interior_top
    dy = target_base_row - bar_base_row

    col_rows: dict[int, list[int]] = {}
    for r, c in bar_pts:
        col_rows.setdefault(int(c), []).append(int(r))
    if mode == "up":
        src_proj_cols = sorted(c for c, rows in col_rows.items() if min(rows) < bar_base_row)
    else:
        src_proj_cols = sorted(c for c, rows in col_rows.items() if max(rows) > bar_base_row)

    dst_cols_set = set()
    for r, c in np.argwhere((g != bg) & (g != grey)):
        rr = int(r)
        cc = int(c)
        if sr0 <= rr <= sr1 and sc0 <= cc <= sc1:
            dst_cols_set.add(cc)
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
        rr = int(r) + dy
        cc = int(c) + dx
        if sr0 <= rr <= sr1 and sc0 <= cc <= sc1:
            out[rr, cc] = grey
            placed.append((rr, cc))

    placed_by_col: dict[int, list[int]] = {}
    for rr, cc in placed:
        placed_by_col.setdefault(int(cc), []).append(int(rr))

    for col in dst_cols:
        if col not in placed_by_col:
            continue
        rows = [r for r in range(sr0, sr1 + 1) if g[r, col] != bg and g[r, col] != grey]
        if not rows:
            continue
        colors = [int(g[r, col]) for r in rows]

        if mode == "up":
            bar_top = min(placed_by_col[col])
            desired_bottom = bar_top - 1
            max_len = desired_bottom - interior_top + 1
            if max_len <= 0:
                continue
            if len(colors) > max_len:
                colors = colors[-max_len:]
            start_row = max(interior_top, desired_bottom - len(colors) + 1)
            for i, colv in enumerate(colors):
                rr = start_row + i
                if interior_top <= rr <= interior_bottom:
                    out[rr, col] = colv
        else:
            bar_bottom = max(placed_by_col[col])
            desired_top = bar_bottom + 1
            max_len = interior_bottom - desired_top + 1
            if max_len <= 0:
                continue
            if len(colors) > max_len:
                colors = colors[:max_len]
            for i, colv in enumerate(colors):
                rr = desired_top + i
                if interior_top <= rr <= interior_bottom:
                    out[rr, col] = colv

    return out


# --------------- generator (D11 variant) ---------------

BG = 0
GREY = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)

H, W = 13, 20

FRAME_R0, FRAME_C0 = 0, 9
FRAME_H, FRAME_W = 8, 11
LEFT_WALL_OPEN_ROWS = (4, 5)
BOTTOM_WALL_OPEN_LOCAL_COLS = tuple(range(2, 9))  # absolute cols 11..17

SLOT_COLS = [11, 13, 15, 17]
SLOT_LENS = [3, 4, 4, 3]
SLOT_TOP_ROW = 1

BAR_H, BAR_W = 4, 7
BAR_LOCAL_PROJ_COLS = [0, 2, 4, 6]

LEFT_INPUT_R0 = 2
LEFT_INPUT_C0_CHOICES = [0, 1, 2]
BOTTOM_INPUT_R0 = 9
BOTTOM_INPUT_C0_CHOICES = [7, 8, 9]

TARGET_C0 = 11
TARGET_UP_R0 = 2
TARGET_DOWN_R0 = 1

HEIGHT_RANGE = (1, 5)
MAX_ATTEMPTS = 400


def _frame_patch() -> np.ndarray:
    patch = np.zeros((FRAME_H, FRAME_W), dtype=int)
    patch[0, :] = GREY
    patch[-1, :] = GREY
    patch[:, 0] = GREY
    patch[:, -1] = GREY
    patch[list(LEFT_WALL_OPEN_ROWS), 0] = BG
    patch[-1, list(BOTTOM_WALL_OPEN_LOCAL_COLS)] = BG
    return patch


def _bar_patch(heights: list[int], mode: str) -> np.ndarray:
    patch = np.zeros((BAR_H, BAR_W), dtype=int)
    if mode == "up":
        patch[-1, :] = GREY
        for local_c, height in zip(BAR_LOCAL_PROJ_COLS, heights):
            patch[BAR_H - height : BAR_H, local_c] = GREY
    else:
        patch[0, :] = GREY
        for local_c, height in zip(BAR_LOCAL_PROJ_COLS, heights):
            patch[:height, local_c] = GREY
    return patch


def _paint_input_slot_column(grid: np.ndarray, col: int, length: int, color: int, count: int) -> None:
    cap = length - count
    if cap > 0:
        grid[SLOT_TOP_ROW : SLOT_TOP_ROW + cap, col] = GREY
    grid[SLOT_TOP_ROW + cap : SLOT_TOP_ROW + length, col] = color


def _paint_output_slot_column(
    grid: np.ndarray, col: int, length: int, color: int, count: int, height: int, mode: str
) -> None:
    cap = length - count
    if cap > 0:
        grid[SLOT_TOP_ROW : SLOT_TOP_ROW + cap, col] = GREY

    interior_top = FRAME_R0 + 1
    interior_bottom = FRAME_R0 + FRAME_H - 2
    if mode == "up":
        bar_top = TARGET_UP_R0 + BAR_H - height
        desired_bottom = bar_top - 1
        visible = min(count, desired_bottom - interior_top + 1)
        if visible > 0:
            start_row = desired_bottom - visible + 1
            grid[start_row : desired_bottom + 1, col] = color
    else:
        proj_bottom = TARGET_DOWN_R0 + height - 1
        desired_top = proj_bottom + 1
        visible = min(count, interior_bottom - desired_top + 1)
        if visible > 0:
            grid[desired_top : desired_top + visible, col] = color


def _build_input_grid(
    heights: list[int], colors: list[int], colored_counts: list[int], mode: str, bar_c0: int
) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    bar_r0 = LEFT_INPUT_R0 if mode == "up" else BOTTOM_INPUT_R0
    paste(grid, _bar_patch(heights, mode), bar_r0, bar_c0)
    for col, length, color, count in zip(SLOT_COLS, SLOT_LENS, colors, colored_counts):
        _paint_input_slot_column(grid, col, length, color, count)
    return grid


def _build_output_grid(
    heights: list[int], colors: list[int], colored_counts: list[int], mode: str
) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    target_r0 = TARGET_UP_R0 if mode == "up" else TARGET_DOWN_R0
    paste(grid, _bar_patch(heights, mode), target_r0, TARGET_C0)
    for col, length, color, count, height in zip(SLOT_COLS, SLOT_LENS, colors, colored_counts, heights):
        _paint_output_slot_column(grid, col, length, color, count, height, mode)
    return grid


def _sample_params(rng: np.random.Generator) -> tuple[str, list[int], list[int], list[int], int]:
    mode = str(rng.choice(["up", "down"]))
    colors = [int(v) for v in rng.choice(PALETTE, size=len(SLOT_COLS), replace=False)]
    colored_counts = [int(rng.integers(1, length + 1)) for length in SLOT_LENS]
    heights = [int(rng.integers(*HEIGHT_RANGE)) for _ in SLOT_COLS]
    if max(heights) == 1:
        idx = int(rng.integers(0, len(heights)))
        heights[idx] = int(rng.integers(2, HEIGHT_RANGE[1]))
    if mode == "up":
        bar_c0 = int(rng.choice(LEFT_INPUT_C0_CHOICES))
    else:
        bar_c0 = int(rng.choice(BOTTOM_INPUT_C0_CHOICES))
    return mode, heights, colors, colored_counts, bar_c0


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    mode, heights, colors, colored_counts, bar_c0 = _sample_params(rng)
    grid = _build_input_grid(heights, colors, colored_counts, mode, bar_c0)
    expected = _build_output_grid(heights, colors, colored_counts, mode)

    if get_bg(grid) != BG:
        return None
    if sorted(colors_present(grid, exclude=[BG, GREY])) != sorted(colors):
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_ATTEMPTS} attempts")
