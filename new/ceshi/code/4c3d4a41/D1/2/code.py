# Variant of 4c3d4a41
# Dimension: D1 — 匹配条件：凸起-列对齐方式 (match_cond)
# Change: 杆的凸起列与框架内彩色列按绝对列位置一一对齐(projection_slot_alignment + column_alignment) -> 形状轮廓匹配

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg as _bg_color
from arc_utils import colors_present, get_bg, paste

# Design analysis for the D1 variant
# a) Concrete behavior and trace
#    - Keep the original 8x20 canvas, the same right-hand grey frame, the same left-hand grey bar,
#      and the same "move bar into frame, then place colors above projections" macro-action.
#    - The changed part is the pairing rule between the 4 input color columns and the 4 bar teeth.
#      In the original task, color column i stayed with absolute column i after the bar moved.
#      Here, each input color column is reassigned to the tooth whose remaining headroom matches
#      that column's visible colored height, so the two 1D contours are complementary.
#    - Supportive adjustment: all 4 slot columns are length 4 instead of [3,4,4,3]. This is needed
#      so every column can realize one of the 4 distinct contour levels {1,2,3,4}; otherwise the
#      outer columns would only admit 3 levels and shape matching would become ambiguous.
#    - Small trace: if bar heights left->right are [3,1,4,2], then the free spaces above them are
#      [2,4,1,3]. If the input frame shows colored heights [4,2,3,1] on cols [11,13,15,17], then
#      the colors from those 4 columns must move to output cols [13,11,17,15] respectively, because
#      each color follows its column height, not its original absolute x-position.
# b) Complexity calibration
#    - The bar uses a permutation of heights [1,2,3,4], and the frame uses a derangement of the
#      complementary counts [4,3,2,1]. This gives a visible, non-degenerate 1D profile while keeping
#      the branching small and human-readable.
#    - Distinct counts avoid ambiguous many-to-one matches. Requiring a derangement guarantees that
#      the new matching rule actually changes at least one color's destination column.
# c) Old-solver assumptions audited
#    - Grey-component detection, bar/base-row detection, and bar translation remain valid because the
#      frame opening, bar footprint, and insertion berth are unchanged.
#    - The original same-column loop `for col in dst_cols` implicitly assumed "input colored column ==
#      output projection column". That is no longer true and is replaced by a count/headroom bijection.
#    - The original horizontal dx computation stays valid because the bar still lands in the same
#      7-cell berth under the 4 designated slot columns; only the color-to-tooth correspondence changed.
# d) Placement legality
#    - Frame, bar, and slot columns still fit the original 8x20 geometry.
#    - Slot columns occupy rows 1..4 only; the inserted bar occupies row 5 plus optional rows above,
#      so output colors stacked in rows 1..(5-h) never cross the frame border or the bar.
# e) Visual observability
#    - Humans can read the new rule from 2-3 examples because every sample shows two explicit contours:
#      the bar tooth skyline and the colored-column skyline. Colors move sideways according to the
#      complementary heights, which is visually different from the original same-column behavior.
#    - No color-id ordering is used anywhere. Matching depends only on visible heights.
# f) Visual cleanliness
#    - The frame region still contains only three layers: frame grey, bar grey, and solid monochrome
#      color stacks. There are no overlapping annotations or extra markers.
# g) New assumptions and generator guarantees
#    - Solver assumes exactly 4 colored slot columns and 4 bar teeth, with unique colored counts and
#      unique tooth headrooms. The generator enforces this by sampling a permutation of [1,2,3,4] for
#      bar heights and a derangement of the complementary counts for the input slot columns.


def _bbox(pts):
    return bbox_coords(pts if isinstance(pts, np.ndarray) else np.asarray(list(pts), dtype=int))


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _most_common(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fit a grey bar into a grey frame by matching a complementary contour, then relocate each color stack above its matched tooth.

    Input structure:
    - Background plus grey structure form two components: a large right-side frame and a smaller left-side bar with four upward projections.
    - Inside the frame, four colored vertical stacks sit on fixed slot columns; their visible heights define a lock-like contour, with grey caps possibly above them.

    Rule:
    - Move the bar upward into the frame so its base sits one cell above the interior floor, keeping the same berth as in the original task.
    - Do not preserve input colors by absolute column. Instead, match each input color column to the bar projection whose available headroom above the tooth equals that column's colored height.
    - After the match is chosen, stack that entire color block immediately above the matched projection.

    Key insight: The relevant correspondence is a complementary 1D profile match: tall teeth pair with short color columns, and short teeth pair with tall color columns.
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

    bar_base_row = max(int(r) for r, _ in bar_pts)
    slot_interior_bottom = sr1 - 1
    target_base_row = slot_interior_bottom - 1
    dy = target_base_row - bar_base_row

    bar_cols = sorted({int(c) for _, c in bar_pts})
    src_tooth_cols = bar_cols[::2]

    col_min_row = {}
    for r, c in bar_pts:
        c = int(c)
        r = int(r)
        col_min_row[c] = min(col_min_row.get(c, r), r)
    slot_info = []
    for c in range(sc0, sc1 + 1):
        rows = [r for r in range(sr0, sr1 + 1) if g[r, c] != bg and g[r, c] != grey]
        if not rows:
            continue
        colors = [int(g[r, c]) for r in rows]
        slot_info.append({"col": int(c), "count": len(rows), "colors": colors})
    if not slot_info:
        return out

    dst_cols = sorted(info["col"] for info in slot_info)

    dx = None
    if src_tooth_cols and len(src_tooth_cols) == len(dst_cols):
        deltas = [d - s for s, d in zip(src_tooth_cols, dst_cols)]
        if len(set(deltas)) == 1:
            dx = deltas[0]
    if dx is None:
        best = (-1, 0)
        dst_set = set(dst_cols)
        for cand_dx in range(-w, w + 1):
            score = sum((c + cand_dx) in dst_set for c in src_tooth_cols)
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

    interior_top = sr0 + 1
    interior_bottom = sr1 - 1

    space_to_dst = {}
    dst_bar_top = {}
    for src_col in src_tooth_cols:
        dst_col = int(src_col + dx)
        bar_top = int(col_min_row[src_col] + dy)
        headroom = bar_top - interior_top
        space_to_dst.setdefault(int(headroom), []).append(dst_col)
        dst_bar_top[dst_col] = bar_top

    for info in slot_info:
        count = int(info["count"])
        colors = info["colors"]
        dst_for_count = space_to_dst.get(count, [])
        if len(dst_for_count) != 1:
            continue
        col = dst_for_count[0]
        bar_top = dst_bar_top[col]
        desired_bottom = bar_top - 1
        if desired_bottom < interior_top:
            continue

        max_len = desired_bottom - interior_top + 1
        if max_len <= 0:
            continue
        if len(colors) > max_len:
            colors = colors[-max_len:]

        start_row = desired_bottom - len(colors) + 1
        start_row = max(start_row, interior_top)
        for i, colv in enumerate(colors):
            rr = start_row + i
            if interior_top <= rr <= interior_bottom:
                out[rr, col] = colv

    return out


# --------------- generator (D1 shape-match variant) ---------------

BG = 0
GREY = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)

H, W = 8, 20

FRAME_R0, FRAME_C0 = 0, 9
FRAME_H, FRAME_W = 8, 11
LEFT_WALL_OPEN_ROWS = (4, 5)

SLOT_COLS = [11, 13, 15, 17]
SLOT_LEN = 4
SLOT_TOP_ROW = 1

BAR_INPUT_C0 = 1
BAR_OUTPUT_C0 = 11
BAR_R0 = 2
BAR_H, BAR_W = 4, 7
BAR_LOCAL_PROJ_COLS = [0, 2, 4, 6]
BAR_BASE_ABS_ROW = BAR_R0 + BAR_H - 1

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


def _paint_input_slot_column(grid: np.ndarray, col: int, color: int, count: int) -> None:
    cap = SLOT_LEN - count
    if cap > 0:
        grid[SLOT_TOP_ROW : SLOT_TOP_ROW + cap, col] = GREY
    grid[SLOT_TOP_ROW + cap : SLOT_TOP_ROW + SLOT_LEN, col] = color


def _paint_output_slot_column(grid: np.ndarray, col: int, color: int, count: int) -> None:
    if count > 0:
        grid[SLOT_TOP_ROW : SLOT_TOP_ROW + count, col] = color


def _build_input_grid(heights: list[int], colors: list[int], input_counts: list[int]) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    paste(grid, _bar_patch(heights), BAR_R0, BAR_INPUT_C0)
    for col, color, count in zip(SLOT_COLS, colors, input_counts):
        _paint_input_slot_column(grid, col, color, count)
    return grid


def _build_expected_grid(heights: list[int], colors: list[int], input_counts: list[int]) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    paste(grid, _bar_patch(heights), BAR_R0, BAR_OUTPUT_C0)

    count_to_color = {int(count): int(color) for color, count in zip(colors, input_counts)}
    target_counts = [BAR_BASE_ABS_ROW - int(height) for height in heights]  # 5 - h
    for col, target_count in zip(SLOT_COLS, target_counts):
        _paint_output_slot_column(grid, col, count_to_color[int(target_count)], int(target_count))
    return grid


def _sample_heights(rng: np.random.Generator) -> list[int]:
    while True:
        heights = [int(v) for v in rng.permutation(np.array([1, 2, 3, 4], dtype=int))]
        if heights != sorted(heights) and heights != sorted(heights, reverse=True):
            return heights


def _sample_input_counts(target_counts: list[int], rng: np.random.Generator) -> list[int]:
    target = np.array(target_counts, dtype=int)
    for _ in range(100):
        cand = [int(v) for v in rng.permutation(target)]
        if all(a != b for a, b in zip(cand, target_counts)):
            return cand
    return [int(v) for v in target[::-1]]


def _sample_params(rng: np.random.Generator) -> tuple[list[int], list[int], list[int]]:
    heights = _sample_heights(rng)
    target_counts = [BAR_BASE_ABS_ROW - int(height) for height in heights]
    input_counts = _sample_input_counts(target_counts, rng)
    colors = [int(v) for v in rng.choice(PALETTE, size=len(SLOT_COLS), replace=False)]
    return heights, colors, input_counts


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    heights, colors, input_counts = _sample_params(rng)
    grid = _build_input_grid(heights, colors, input_counts)
    expected = _build_expected_grid(heights, colors, input_counts)

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
