# Combo variant of 4c3d4a41 (index 14)
# combo_spec: D3/3+D11/2
# Name: 框架隔板+多开口

import numpy as np

from arc_utils import bbox_coords, colors_present, components_mask, get_bg as _bg_color
from arc_utils import get_bg, paste

# Design analysis
# ===============
# a) Concrete behavior and trace
#    - The seed task has one right-hand frame, one left-hand upward bar, and four colored slot columns.
#      Everything is solved by one horizontal insertion and one "stack colors above" rule.
#    - The fused variant keeps the same right-hand frame idea, but the frame now has:
#        1. a horizontal divider at absolute row 3
#        2. a left-wall opening at rows 4-5 for an upward bar
#        3. a bottom-wall opening at cols 15-17 for a downward bar
#    - Instead of forcing one object to fake two entrances, the input now shows two small grey keyed bars:
#        * an upward 2-key bar on the left, meant for the upper shafts at cols 11 and 13
#        * a downward 2-key bar below, meant for the lower shafts at cols 15 and 17
#    - Example trace:
#        * Upper col 11 uses rows 1..3. If its count is 3 and the upward key height is 3, the key reaches row 3,
#          so only rows 1..2 stay available and the column keeps its bottom-most 2 color cells there.
#        * Lower col 17 uses rows 4..6. If its count is 2 and the downward key height is 1, the key only occupies
#          row 4, so rows 5..6 remain and the 2 color cells stack immediately below the key.
#        * Difference from the seed: one part of the puzzle still stacks upward, but another part now stacks downward,
#          and the divider decides which target columns belong to which entrance.
#
# b) Complexity calibration
#    - The frame still occupies one fixed bbox, and there are still exactly four target color columns.
#    - Each bar only controls two columns, which keeps matching observable and avoids a visually tangled output.
#    - Heights are restricted to two small sets:
#        * upward keys in {3, 4} so they always pierce the divider holes
#        * downward keys in {1, 2} so they always stay inside the lower chamber
#    - Each shaft uses one repeated color with 4 distinct colors total, preserving the original task's color simplicity.
#
# c) Hidden-assumption audit against the seed solver
#    - The seed assumes "largest grey component = frame, second grey component = bar". That breaks here because there
#      are two bars, so the new solver must detect one upward bar and one downward bar among the non-frame components.
#    - The seed assumes all target columns share one cavity and one stacking direction. That fails because upper shafts
#      stop at the divider while lower shafts start below it, and the lower bar stacks colors below its projections.
#    - The seed uses max-row as the base row. That is only valid for upward bars. The downward bar instead has its
#      full-width base on its top row, so the solver must infer orientation before computing vertical placement.
#    - The seed ignores frame openings. In the fusion they matter structurally: the left opening pairs with the upward
#      bar and the bottom opening pairs with the downward bar, so the generator must keep both visible in every sample.
#
# d) Placement legality
#    - Grid size is raised to 13x20 to leave a clean gap below the frame for the bottom-entry bar.
#    - The frame stays at rows 0..7, cols 9..19. The upper input bar lives at rows 3..6, cols 1..5 at most; the
#      lower input bar lives at rows 9..11, cols 12..16 at most. These placements are disjoint and in-bounds.
#    - Output placements are fixed and disjoint too:
#        * upward bar -> rows 2..5, cols 11..13
#        * downward bar -> rows 4..6, cols 15..17
#      so no two moving objects overlap after insertion.
#
# e) Visual observability
#    - A human can see the divider, the two different openings, and the two bars waiting at different sides.
#    - The upper target columns visibly pierce the divider; the lower target columns visibly do not.
#    - Across 2-3 examples, the consistent pattern "left/up bar serves top shafts, bottom/down bar serves bottom shafts"
#      is recoverable from pixels alone without using hidden ids or color values.
#
# f) Visual cleanliness
#    - The upper and lower target columns are separated into two 2-column groups with a blank column gap between them.
#    - Each output region contains one short grey bar and at most one monochrome stack per shaft, so the frame interior
#      stays readable rather than becoming one large merged grey blob.
#
# g) Change inferability
#    - Colors never remap: each shaft keeps its own monochrome identity before and after movement.
#    - Only the available rows relative to the inserted key change, so the rule is geometric rather than color-coded.
#
# h) Anti-degeneration target
#    - Every generated sample must contain both an upward bar and a downward bar.
#    - Every lower shaft must still show at least one color cell in the fused output; otherwise the bottom-entry branch
#      would be visually mute. Likewise every upper shaft must show at least one visible color cell above its key.
#    - A counterfactual "treat every shaft like the old upward-only rule" transform should disagree with the fused
#      output on the generated samples; later validation measures that directly.
#
# i) New solver assumptions that the generator must guarantee
#    - Exactly one frame component is larger than both bars, and exactly one non-frame bar points up while one points down.
#    - The divider row is the interior row with the widest grey span, and upper target columns are exactly the colored
#      columns where that divider row is open.
#    - Bars have full-width bases and exactly two keyed columns at the base edges. A downward bar may extend farther below
#      the base on one or both keyed columns, but the left/right keyed locations themselves are always visible.

BG = 0
GREY = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)

H, W = 13, 20

FRAME_R0, FRAME_C0 = 0, 9
FRAME_H, FRAME_W = 8, 11
PARTITION_ROW = 3
LEFT_WALL_OPEN_ROWS = (4, 5)
BOTTOM_WALL_OPEN_LOCAL_COLS = (6, 7, 8)  # abs cols 15..17

UPPER_COLS = [11, 13]
LOWER_COLS = [15, 17]
SHAFT_ROWS = {
    11: [1, 2, 3],
    13: [1, 2, 3],
    15: [4, 5, 6],
    17: [4, 5, 6],
}

UP_BAR_H, UP_BAR_W = 4, 3
DOWN_BAR_H, DOWN_BAR_W = 3, 3
BAR_LOCAL_PROJ_COLS = [0, 2]

UP_INPUT_R0 = 3
UP_INPUT_C0_CHOICES = [1, 2, 3]
DOWN_INPUT_R0 = 9
DOWN_INPUT_C0_CHOICES = [12, 13, 14]

UP_TARGET_R0 = 2
UP_TARGET_C0 = 11
DOWN_TARGET_R0 = 4
DOWN_TARGET_C0 = 15

MAX_ATTEMPTS = 400
UP_HEIGHT_PATTERNS = [(3, 3), (3, 4), (4, 3), (4, 4)]
DOWN_HEIGHT_PATTERNS = [(1, 2), (2, 1), (2, 2)]
COUNT_PATTERNS = [
    (1, 2, 1, 2),
    (2, 3, 2, 3),
    (3, 1, 3, 1),
    (2, 1, 3, 2),
    (3, 2, 1, 3),
    (1, 3, 2, 1),
]
_SAMPLE_CURSOR = 0


def _bbox(pts):
    return bbox_coords(pts if isinstance(pts, np.ndarray) else np.asarray(list(pts), dtype=int))


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _most_common(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _bar_mode_and_base(bar_pts: np.ndarray) -> tuple[str, int, list[int]]:
    row_to_cols: dict[int, set[int]] = {}
    col_to_rows: dict[int, list[int]] = {}
    for r, c in bar_pts:
        rr = int(r)
        cc = int(c)
        row_to_cols.setdefault(rr, set()).add(cc)
        col_to_rows.setdefault(cc, []).append(rr)

    base_row = max(row_to_cols, key=lambda r: (len(row_to_cols[r]), -r))
    has_above = any(int(r) < base_row for r, _ in bar_pts)
    has_below = any(int(r) > base_row for r, _ in bar_pts)
    if has_above == has_below:
        raise ValueError("bar orientation is ambiguous")

    base_cols = sorted(row_to_cols[base_row])
    if len(base_cols) < 2:
        raise ValueError("bar base is too narrow")
    proj_cols = [int(base_cols[0]), int(base_cols[-1])]

    if has_above:
        return "up", int(base_row), proj_cols
    return "down", int(base_row), proj_cols


def _aligned_shift(src_cols: list[int], dst_cols: list[int], width: int) -> int:
    dx = None
    if src_cols and len(src_cols) == len(dst_cols):
        deltas = [d - s for s, d in zip(src_cols, dst_cols)]
        if len(set(deltas)) == 1:
            dx = deltas[0]
    if dx is not None:
        return int(dx)

    best = (-1, 0)
    dst_set = set(dst_cols)
    for cand_dx in range(-width, width + 1):
        score = sum((c + cand_dx) in dst_set for c in src_cols)
        if score > best[0]:
            best = (score, cand_dx)
    return int(best[1])


def _collect_target_cols(grid: np.ndarray, bg: int, grey: int, bbox: tuple[int, int, int, int]) -> list[int]:
    sr0, sc0, sr1, sc1 = bbox
    return sorted(
        {
            int(c)
            for r, c in np.argwhere((grid != bg) & (grid != grey))
            if sr0 <= int(r) <= sr1 and sc0 <= int(c) <= sc1
        }
    )


def _paint_colors_above(
    out: np.ndarray,
    grid: np.ndarray,
    bg: int,
    grey: int,
    cols: list[int],
    shaft_rows_by_col: dict[int, list[int]],
    placed_by_col: dict[int, list[int]],
) -> None:
    for col in cols:
        shaft_rows = shaft_rows_by_col[col]
        bar_rows_here = sorted(r for r in placed_by_col.get(col, []) if r in shaft_rows)
        if not bar_rows_here:
            continue
        allowed_rows = [r for r in shaft_rows if r < bar_rows_here[0]]
        if not allowed_rows:
            continue
        colors = [int(grid[r, col]) for r in shaft_rows if grid[r, col] != bg and grid[r, col] != grey]
        if not colors:
            continue
        if len(colors) > len(allowed_rows):
            colors = colors[-len(allowed_rows) :]
        dest_rows = allowed_rows[-len(colors) :]
        for rr, colv in zip(dest_rows, colors):
            out[rr, col] = colv


def _paint_colors_below(
    out: np.ndarray,
    grid: np.ndarray,
    bg: int,
    grey: int,
    cols: list[int],
    shaft_rows_by_col: dict[int, list[int]],
    placed_by_col: dict[int, list[int]],
) -> None:
    for col in cols:
        shaft_rows = shaft_rows_by_col[col]
        bar_rows_here = sorted(r for r in placed_by_col.get(col, []) if r in shaft_rows)
        if not bar_rows_here:
            continue
        allowed_rows = [r for r in shaft_rows if r > bar_rows_here[-1]]
        if not allowed_rows:
            continue
        colors = [int(grid[r, col]) for r in shaft_rows if grid[r, col] != bg and grid[r, col] != grey]
        if not colors:
            continue
        if len(colors) > len(allowed_rows):
            colors = colors[: len(allowed_rows)]
        dest_rows = allowed_rows[: len(colors)]
        for rr, colv in zip(dest_rows, colors):
            out[rr, col] = colv


def main(grid: np.ndarray) -> np.ndarray:
    """
    Insert one grey bar through the left opening and another through the bottom opening of a partitioned frame.

    Input structure:
    - Grey is the structural color. It forms one large right-hand frame with a horizontal divider plus two smaller bars:
      an upward-pointing bar on the left and a downward-pointing bar below.
    - Colored cells appear only in four thin target shafts inside the frame. Two shafts pierce the divider and belong to
      the upper chamber; two shafts stay entirely below the divider.

    Rule:
    - Detect the upward bar and align its two keyed columns with the divider-piercing shafts, then slide it in from the
      left so its base sits low in the frame. Stack each upper-shaft color run immediately above that bar's projection.
    - Detect the downward bar and align its two keyed columns with the lower shafts, then slide it in from the bottom so
      its base sits at the top of the lower chamber. Stack each lower-shaft color run immediately below that bar's projection.

    Key insight:
    - The divider and the two openings split the original single motion into two coordinated insertions: upper shafts must
      be interpreted with an upward-stacking left-entry bar, while lower shafts must be interpreted with a downward-stacking
      bottom-entry bar.
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
    if len(grey_comps) < 3:
        return g.copy()

    (_, slot_bbox, _), *others = grey_comps
    sr0, sc0, sr1, sc1 = slot_bbox

    bars: dict[str, tuple[int, np.ndarray, list[int]]] = {}
    for _size, _bbox_unused, pts in others:
        try:
            mode, base_row, proj_cols = _bar_mode_and_base(pts)
        except ValueError:
            continue
        if mode not in bars or len(pts) > len(bars[mode][1]):
            bars[mode] = (base_row, pts, proj_cols)
    if "up" not in bars or "down" not in bars:
        return g.copy()

    out = np.full_like(g, bg)
    out[sr0 : sr1 + 1, sc0 : sc1 + 1] = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region = out[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region[(slot_region != bg) & (slot_region != grey)] = bg

    interior_cols = list(range(sc0 + 1, sc1))
    divider_row = max(
        range(sr0 + 1, sr1),
        key=lambda r: sum(g[r, c] == grey for c in interior_cols),
    )

    dst_cols = _collect_target_cols(g, bg, grey, slot_bbox)
    upper_cols = [col for col in dst_cols if g[divider_row, col] != grey]
    lower_cols = [col for col in dst_cols if g[divider_row, col] == grey]

    up_base_row, up_pts, up_src_cols = bars["up"]
    up_dx = _aligned_shift(up_src_cols, upper_cols, w)
    up_target_base_row = sr1 - 2
    up_dy = up_target_base_row - up_base_row

    down_base_row, down_pts, down_src_cols = bars["down"]
    down_dx = _aligned_shift(down_src_cols, lower_cols, w)
    down_target_base_row = divider_row + 1
    down_dy = down_target_base_row - down_base_row

    up_placed_by_col: dict[int, list[int]] = {}
    for r, c in up_pts:
        rr = int(r) + up_dy
        cc = int(c) + up_dx
        if sr0 <= rr <= sr1 and sc0 <= cc <= sc1:
            out[rr, cc] = grey
            up_placed_by_col.setdefault(cc, []).append(rr)

    down_placed_by_col: dict[int, list[int]] = {}
    for r, c in down_pts:
        rr = int(r) + down_dy
        cc = int(c) + down_dx
        if sr0 <= rr <= sr1 and sc0 <= cc <= sc1:
            out[rr, cc] = grey
            down_placed_by_col.setdefault(cc, []).append(rr)

    upper_rows = {col: list(range(sr0 + 1, divider_row + 1)) for col in upper_cols}
    lower_rows = {col: list(range(divider_row + 1, sr1)) for col in lower_cols}

    _paint_colors_above(out, g, bg, grey, upper_cols, upper_rows, up_placed_by_col)
    _paint_colors_below(out, g, bg, grey, lower_cols, lower_rows, down_placed_by_col)
    return out


def _frame_patch() -> np.ndarray:
    patch = np.zeros((FRAME_H, FRAME_W), dtype=int)
    patch[0, :] = GREY
    patch[-1, :] = GREY
    patch[:, 0] = GREY
    patch[:, -1] = GREY
    patch[list(LEFT_WALL_OPEN_ROWS), 0] = BG
    patch[-1, list(BOTTOM_WALL_OPEN_LOCAL_COLS)] = BG
    patch[PARTITION_ROW, :] = GREY
    patch[PARTITION_ROW, 2] = BG  # abs col 11
    patch[PARTITION_ROW, 4] = BG  # abs col 13
    return patch


def _bar_patch(heights: list[int], mode: str) -> np.ndarray:
    if mode == "up":
        patch = np.zeros((UP_BAR_H, UP_BAR_W), dtype=int)
        patch[-1, :] = GREY
        for local_c, height in zip(BAR_LOCAL_PROJ_COLS, heights):
            patch[UP_BAR_H - height : UP_BAR_H, local_c] = GREY
        return patch

    patch = np.zeros((DOWN_BAR_H, DOWN_BAR_W), dtype=int)
    patch[0, :] = GREY
    for local_c, height in zip(BAR_LOCAL_PROJ_COLS, heights):
        patch[:height, local_c] = GREY
    return patch


def _paint_input_column(grid: np.ndarray, col: int, color: int, count: int) -> None:
    rows = SHAFT_ROWS[col]
    cap = len(rows) - count
    for i, r in enumerate(rows):
        grid[r, col] = GREY if i < cap else color


def _paint_output_upper_column(grid: np.ndarray, col: int, color: int, count: int, height: int) -> None:
    rows = SHAFT_ROWS[col]
    cap = len(rows) - count
    for i in range(cap):
        grid[rows[i], col] = GREY

    bar_top = UP_TARGET_R0 + UP_BAR_H - height
    visible_rows = [r for r in rows if r < bar_top]
    visible = min(count, len(visible_rows))
    if visible <= 0:
        return
    for r in visible_rows[-visible:]:
        grid[r, col] = color


def _paint_output_lower_column(grid: np.ndarray, col: int, color: int, count: int, height: int) -> None:
    rows = SHAFT_ROWS[col]
    cap = len(rows) - count
    for i in range(cap):
        grid[rows[i], col] = GREY

    bar_bottom = DOWN_TARGET_R0 + height - 1
    visible_rows = [r for r in rows if r > bar_bottom]
    visible = min(count, len(visible_rows))
    if visible <= 0:
        return
    for r in visible_rows[:visible]:
        grid[r, col] = color


def _build_input_grid(
    up_heights: list[int],
    down_heights: list[int],
    colors: list[int],
    counts: list[int],
    up_c0: int,
    down_c0: int,
) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    paste(grid, _bar_patch(up_heights, "up"), UP_INPUT_R0, up_c0, transparent=BG)
    paste(grid, _bar_patch(down_heights, "down"), DOWN_INPUT_R0, down_c0, transparent=BG)
    for col, color, count in zip(UPPER_COLS + LOWER_COLS, colors, counts):
        _paint_input_column(grid, col, color, count)
    return grid


def _build_output_grid(
    up_heights: list[int],
    down_heights: list[int],
    colors: list[int],
    counts: list[int],
) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    paste(grid, _bar_patch(up_heights, "up"), UP_TARGET_R0, UP_TARGET_C0, transparent=BG)
    paste(grid, _bar_patch(down_heights, "down"), DOWN_TARGET_R0, DOWN_TARGET_C0, transparent=BG)

    for col, color, count, height in zip(UPPER_COLS, colors[:2], counts[:2], up_heights):
        _paint_output_upper_column(grid, col, color, count, height)
    for col, color, count, height in zip(LOWER_COLS, colors[2:], counts[2:], down_heights):
        _paint_output_lower_column(grid, col, color, count, height)
    return grid


def _sample_params(
    rng: np.random.Generator,
) -> tuple[list[int], list[int], list[int], list[int], int, int]:
    global _SAMPLE_CURSOR

    cursor = _SAMPLE_CURSOR
    _SAMPLE_CURSOR += 1

    colors = [int(v) for v in rng.choice(PALETTE, size=4, replace=False)]
    counts = list(COUNT_PATTERNS[cursor % len(COUNT_PATTERNS)])
    up_heights = list(UP_HEIGHT_PATTERNS[cursor % len(UP_HEIGHT_PATTERNS)])
    down_heights = list(DOWN_HEIGHT_PATTERNS[cursor % len(DOWN_HEIGHT_PATTERNS)])
    up_c0 = int(UP_INPUT_C0_CHOICES[cursor % len(UP_INPUT_C0_CHOICES)])
    down_c0 = int(DOWN_INPUT_C0_CHOICES[(cursor * 2 + 1) % len(DOWN_INPUT_C0_CHOICES)])
    return up_heights, down_heights, colors, counts, up_c0, down_c0


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    up_heights, down_heights, colors, counts, up_c0, down_c0 = _sample_params(rng)
    grid = _build_input_grid(up_heights, down_heights, colors, counts, up_c0, down_c0)
    expected = _build_output_grid(up_heights, down_heights, colors, counts)

    if get_bg(grid) != BG:
        return None
    if sorted(colors_present(grid, exclude=[BG, GREY])) != sorted(colors):
        return None
    if sorted(colors_present(expected, exclude=[BG, GREY])) != sorted(colors):
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
