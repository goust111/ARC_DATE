# Variant of 4c3d4a41
# Dimension: D7 — 多步处理顺序 (multi_step)
# Change: 步骤1：识别杆和框架；步骤2：杆水平移入框架对齐凸起与彩色列；步骤3：彩色像素向上堆叠到凸起顶部... → 对接后触发额外填充

import numpy as np

from arc_utils import bbox_coords, colors_present, components_mask, get_bg, get_bg as _bg_color, paste


# ---------------------------------------------------------------------------
# Design analysis for the D7 variant
#
# a) Concrete behavior + trace
#    - Input keeps the original objects: one large grey open frame on the right,
#      one smaller grey comb-bar on the left, and four colored slot-columns at
#      frame columns 11/13/15/17.
#    - The new output still docks the bar into the frame, but docking is no
#      longer the last structural step. After docking, every occupied bar row is
#      horizontally closed from its leftmost grey cell to its rightmost grey
#      cell. This turns the comb silhouette into a solid stepped block.
#    - Only after this solidification do we place the colored cells above the
#      resulting block top in each colored column.
#    - Example: heights [4,1,4,2] dock to rows
#         r2: 11,15
#         r3: 11,15,17
#         r4: 11,13,15,17
#         r5: 11..17
#      Original task keeps those exact cells. Variant fills row spans:
#         r2: 11..15
#         r3: 11..17
#         r4: 11..17
#         r5: 11..17
#      So middle columns 13 and 15 gain higher grey support; their colors must
#      then stack higher than in the original task.
#
# b) Complexity calibration
#    - The frame, bar, and four target columns stay unchanged, so object
#      recognition is still stable.
#    - The new reasoning is visible because the grey mass becomes contiguous on
#      upper rows, not just at the base row.
#    - To avoid a weak variant, the generator rejects cases where solidification
#      adds cells but never changes any colored column's support. At least one
#      middle target column must be lifted by the closure.
#
# c) Hidden assumptions from the original main()
#    - Original color placement used raw projection tops as supports. That is no
#      longer valid after the extra fill step.
#    - Original `placed_by_col` could be built directly from pasted bar pixels.
#      Here it must be built from the solidified cells instead, or colors would
#      ignore the second phase.
#    - Keeping the frame region and clearing only non-grey/non-bg colors is
#      still safe because the variant does not alter frame identity or slot-cap
#      semantics.
#    - Largest grey component = frame, second grey component = bar remains valid
#      because generation preserves the original component sizes and separation.
#
# d) Placement legality
#    - Geometry stays the original 8x20 layout.
#    - The docked bar spans cols 11..17 and rows 2..5, entirely inside the
#      frame interior rows 1..6, cols 10..18.
#    - Row-span closure never escapes the bar width because it only fills
#      between already occupied cells on the same row.
#
# e) Visual observability
#    - The new rule is inferable from 2-4 examples because the added grey cells
#      create conspicuous bridges between projections, and the supported color
#      stacks in inner columns move upward accordingly.
#    - No color-id arithmetic or invisible ordering is used; all changes follow
#      visible spatial support.
#
# f) Visual cleanliness
#    - The fill remains inside the frame and uses the same grey structural
#      color, so the scene still reads as one frame and one inserted block.
#    - Colors are still confined to four odd columns, keeping the output sparse
#      enough to parse.
#
# g) Color-change inferability
#    - Colors are preserved, not remapped. Only their vertical placement changes
#      as a consequence of the new grey support. This keeps the added reasoning
#      structural rather than color-symbolic.
# ---------------------------------------------------------------------------


def _bbox(pts):
    return bbox_coords(pts if isinstance(pts, np.ndarray) else np.asarray(list(pts), dtype=int))


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _most_common(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _solidify_row_spans(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Close each occupied row into one contiguous grey segment."""
    row_to_cols: dict[int, list[int]] = {}
    for rr, cc in cells:
        row_to_cols.setdefault(int(rr), []).append(int(cc))

    solid = set()
    for rr, cols in row_to_cols.items():
        c0, c1 = min(cols), max(cols)
        for cc in range(c0, c1 + 1):
            solid.add((rr, cc))
    return sorted(solid)


def _support_tops(cells: list[tuple[int, int]]) -> dict[int, int]:
    tops: dict[int, int] = {}
    for rr, cc in cells:
        rr = int(rr)
        cc = int(cc)
        tops[cc] = rr if cc not in tops else min(tops[cc], rr)
    return tops


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock a grey bar into a grey frame, solidify the docked comb into a block, then place colors above the fused top.

    Input structure:
    - The background is dominant, while grey forms two disconnected structural pieces: a large right-hand frame and a smaller left-hand bar with upward projections.
    - Inside the frame, four colored vertical markers indicate the destination columns that the bar projections must align to.

    Rule:
    - Translate the bar into the frame so its base sits one row above the frame interior floor and its projection columns line up with the colored columns.
    - After docking, close every occupied bar row into a continuous horizontal grey segment between that row's leftmost and rightmost bar cells.
    - Re-stack each column's original colored pixels directly above the top of this solidified grey support in that column.

    Key insight:
    - The task is no longer "align then stack" only; alignment first creates a comb, and that comb then acts as a scaffold whose row-wise horizontal closure changes where colors can finally rest.
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

    col_min_row: dict[int, int] = {}
    for r, c in bar_pts:
        r = int(r)
        c = int(c)
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
        rr = int(r) + dy
        cc = int(c) + dx
        if sr0 <= rr <= sr1 and sc0 <= cc <= sc1:
            placed.append((rr, cc))

    solid = _solidify_row_spans(placed)
    for rr, cc in solid:
        out[rr, cc] = grey

    support_by_col = _support_tops(solid)
    interior_top = sr0 + 1
    interior_bottom = sr1 - 1

    for col in dst_cols:
        if col not in support_by_col:
            continue
        solid_top = support_by_col[col]
        desired_bottom = solid_top - 1
        if desired_bottom < interior_top:
            continue

        rows = [r for r in range(sr0, sr1 + 1) if g[r, col] != bg and g[r, col] != grey]
        if not rows:
            continue
        colors = [int(g[r, col]) for r in rows]

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

    return out


# --------------- generator ---------------

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

HEIGHT0_RANGE = (2, 5)
HEIGHT_REST_RANGE = (1, 5)
MAX_ATTEMPTS = 1000


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


def _bar_cells(bar_c0: int, heights: list[int]) -> list[tuple[int, int]]:
    pts = np.argwhere(_bar_patch(heights) == GREY)
    return [(BAR_R0 + int(r), bar_c0 + int(c)) for r, c in pts]


def _paint_input_slot_column(grid: np.ndarray, col: int, length: int, color: int, count: int) -> None:
    cap = length - count
    grid[SLOT_TOP_ROW : SLOT_TOP_ROW + cap, col] = GREY
    grid[SLOT_TOP_ROW + cap : SLOT_TOP_ROW + length, col] = color


def _build_input_grid(heights: list[int], colors: list[int], colored_counts: list[int]) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    paste(grid, _bar_patch(heights), BAR_R0, BAR_INPUT_C0)
    for col, length, color, count in zip(SLOT_COLS, SLOT_LENS, colors, colored_counts):
        _paint_input_slot_column(grid, col, length, color, count)
    return grid


def _build_expected_grid(
    heights: list[int],
    colors: list[int],
    colored_counts: list[int],
    *,
    solidify: bool,
) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)

    for col, length, count in zip(SLOT_COLS, SLOT_LENS, colored_counts):
        cap = length - count
        grid[SLOT_TOP_ROW : SLOT_TOP_ROW + cap, col] = GREY

    bar_cells = _bar_cells(BAR_OUTPUT_C0, heights)
    if solidify:
        bar_cells = _solidify_row_spans(bar_cells)
    support_by_col = _support_tops(bar_cells)

    for rr, cc in bar_cells:
        grid[rr, cc] = GREY

    interior_top = FRAME_R0 + 1
    interior_bottom = FRAME_R0 + FRAME_H - 2
    for col, color, count in zip(SLOT_COLS, colors, colored_counts):
        solid_top = support_by_col[col]
        desired_bottom = solid_top - 1
        if desired_bottom < interior_top:
            continue
        visible = min(count, desired_bottom - interior_top + 1)
        if visible <= 0:
            continue
        start_row = desired_bottom - visible + 1
        for rr in range(start_row, desired_bottom + 1):
            if interior_top <= rr <= interior_bottom:
                grid[rr, col] = color
    return grid


def _variant_metrics(
    heights: list[int],
    colors: list[int],
    colored_counts: list[int],
) -> tuple[int, int, int, bool]:
    raw = _bar_cells(BAR_OUTPUT_C0, heights)
    solid = _solidify_row_spans(raw)
    raw_support = _support_tops(raw)
    solid_support = _support_tops(solid)

    fill_gain = len(set(solid)) - len(set(raw))
    raised_support_cols = sum(solid_support[col] < raw_support[col] for col in SLOT_COLS)
    visible_cells = sum(min(count, solid_support[col] - SLOT_TOP_ROW) for col, count in zip(SLOT_COLS, colored_counts))

    raw_expected = _build_expected_grid(heights, colors, colored_counts, solidify=False)
    solid_expected = _build_expected_grid(heights, colors, colored_counts, solidify=True)
    output_differs = not np.array_equal(raw_expected, solid_expected)
    return fill_gain, raised_support_cols, visible_cells, output_differs


def _sample_params(rng: np.random.Generator) -> tuple[list[int], list[int], list[int]]:
    colors = [int(v) for v in rng.choice(PALETTE, size=len(SLOT_COLS), replace=False)]
    colored_counts = [int(rng.integers(1, length + 1)) for length in SLOT_LENS]
    heights = [int(rng.integers(*HEIGHT0_RANGE))]
    heights.extend(int(rng.integers(*HEIGHT_REST_RANGE)) for _ in range(len(SLOT_COLS) - 1))
    return heights, colors, colored_counts


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    heights, colors, colored_counts = _sample_params(rng)
    fill_gain, raised_support_cols, visible_cells, output_differs = _variant_metrics(
        heights, colors, colored_counts
    )
    if fill_gain <= 0 or raised_support_cols <= 0 or visible_cells < 3 or not output_differs:
        return None

    grid = _build_input_grid(heights, colors, colored_counts)
    expected = _build_expected_grid(heights, colors, colored_counts, solidify=True)

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
