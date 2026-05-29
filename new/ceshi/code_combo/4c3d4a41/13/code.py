# Combo variant of 4c3d4a41 (index 13)
# combo_spec: D7/3+D14/0
# Name: 触发填充+顶部灰色填充

import numpy as np

from arc_utils import bbox_coords, colors_present, components_mask, get_bg, get_bg as _bg_color, paste


# ---------------------------------------------------------------------------
# Design analysis for the combo variant D7/3 + D14/0
#
# a) Concrete behavior + trace
#    - Input geometry stays the original 8x20 scene: one grey open frame on the
#      right, one grey toothed bar on the left, and four colored slot columns at
#      frame cols 11/13/15/17.
#    - Fusion order matters:
#        1. Dock the bar into the frame.
#        2. Close every occupied bar row into one solid grey span.
#        3. Re-stack each column's colors above the new support.
#        4. Fill any remaining vertical gap between the preserved top grey cap
#           and the relocated colors with grey.
#    - Trace a concrete case with heights [3,2,3,1] and colored counts [2,2,3,1].
#      Raw docked bar supports:
#        row 3: 11,15
#        row 4: 11,13,15
#        row 5: 11,13,15,17
#      Original task keeps those cells only, so column 13 is supported at row 4
#      and column 17 is supported only at row 5.
#      This combo first solidifies rows:
#        row 3: 11..15
#        row 4: 11..15
#        row 5: 11..17
#      Now column 13 rises to support row 3, so its colors move up by one row.
#      Column 17 is still base-only, so after colors are re-stacked there remains
#      one empty cell between the old top cap and the color; D14 fills that cell
#      grey. The final frame interior reads as one grey mass topped by colors.
#
# b) Complexity calibration
#    - D7 visibility comes from forcing at least one middle column to gain higher
#      support after row-span closure.
#    - D14 visibility comes from forcing the right short slot column to keep only
#      base support after solidification, which creates a visible grey filler cell
#      above its color stack.
#    - Heights are restricted to a small family: h1 in {3,4}, h2 fixed at 2,
#      h3 in {3,4}, h4 fixed at 1. This is enough variation for multiple examples
#      while keeping both effects reliably observable.
#
# c) Hidden assumptions from the original main()
#    - Original code identified tooth columns only by looking for grey pixels
#      above the base row. That breaks here because the D14-triggering right tooth
#      is intentionally base-only. The combo solver recovers the four tooth
#      columns from the contiguous base row instead.
#    - The support map for color placement can no longer come from the raw pasted
#      bar. It must come from the solidified bar, otherwise the D7 phase is lost.
#    - Clearing slot colors while preserving existing grey caps still works, since
#      D14 extends those caps downward rather than changing their meaning.
#    - Largest grey component = frame and second grey component = bar remains
#      valid because generation preserves the original object sizes and separation.
#
# d) Placement legality
#    - The frame bbox is fixed at rows 0..7, cols 9..19; the docked bar remains
#      inside the interior rows 1..6, cols 10..18.
#    - Row-span closure only fills between already occupied cells on the same row,
#      so it never escapes the bar's horizontal extent.
#    - Top-gap fill is applied only inside the four target columns and only above
#      the docked support, so it cannot overwrite the border.
#
# e) Visual observability
#    - Across examples, the middle-left target column visibly rises because the
#      comb becomes a solid stepped block.
#    - Independently, the right short column gains a grey shaft between its old
#      cap and its color. Seeing both changes together makes the two-stage rule
#      inferable from a few input/output pairs.
#
# f) Visual cleanliness
#    - All added cells use the same structural grey and stay inside the frame.
#    - Colors remain confined to the four odd slot columns, so the output is still
#      sparse and easy to parse despite the extra fill stages.
#
# g) Color-change inferability
#    - No color remapping is introduced. Colors only move vertically according to
#      grey support, which keeps the rule structural and human-visible.
# ---------------------------------------------------------------------------


def _bbox(pts):
    return bbox_coords(pts if isinstance(pts, np.ndarray) else np.asarray(list(pts), dtype=int))


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _most_common(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _solidify_row_spans(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    row_to_cols: dict[int, list[int]] = {}
    for rr, cc in cells:
        row_to_cols.setdefault(int(rr), []).append(int(cc))

    solid = set()
    for rr, cols in row_to_cols.items():
        for cc in range(min(cols), max(cols) + 1):
            solid.add((rr, cc))
    return sorted(solid)


def _support_tops(cells: list[tuple[int, int]]) -> dict[int, int]:
    tops: dict[int, int] = {}
    for rr, cc in cells:
        rr = int(rr)
        cc = int(cc)
        tops[cc] = rr if cc not in tops else min(tops[cc], rr)
    return tops


def _top_cap_end(
    grid: np.ndarray,
    col: int,
    interior_top: int,
    interior_bottom: int,
    grey: int,
) -> int:
    cap_end = interior_top - 1
    for rr in range(interior_top, interior_bottom + 1):
        if grid[rr, col] == grey:
            cap_end = rr
            continue
        break
    return cap_end


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock a grey toothed bar into a grey frame, fuse the docked teeth into a solid block, then rebuild each target column with continuous grey support up to the colors.

    Input structure:
    - The dominant background surrounds two disconnected grey structures: a large right-hand frame and a smaller left-hand bar whose bottom row is a full horizontal base with teeth at alternating columns.
    - Inside the frame, four odd columns contain a top grey cap and a same-colored vertical stack that marks where the bar must dock.

    Rule:
    - Move the bar into the frame so its base sits one row above the interior floor and its four tooth columns align with the four colored target columns.
    - After docking, turn every occupied bar row into one continuous grey span between that row's leftmost and rightmost bar cells.
    - Re-stack the original column colors above this solidified support, trimming overflow if needed, and fill any remaining gap between the preserved top grey cap and the relocated colors with grey.

    Key insight:
    - The final support is produced by a chain: infer the tooth lattice from the bar base, dock the bar, solidify the comb into a block, and only then rebuild each colored column as grey shaft plus color top.
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

    base_cols = sorted(int(c) for r, c in bar_pts if int(r) == bar_base_row)
    src_tooth_cols = base_cols[::2]

    dst_cols = sorted(
        {
            int(c)
            for r, c in np.argwhere((g != bg) & (g != grey))
            if sr0 <= int(r) <= sr1 and sc0 <= int(c) <= sc1
        }
    )

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
        desired_bottom = support_by_col[col] - 1
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
        fill_start = _top_cap_end(
            g, col, interior_top, min(interior_bottom, desired_bottom), grey
        ) + 1
        if fill_start < start_row:
            out[fill_start:start_row, col] = grey

        for i, colv in enumerate(colors):
            rr = start_row + i
            if interior_top <= rr <= interior_bottom:
                out[rr, col] = colv

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
SHORT_GAP_SLOT_IDX = 3
RAISED_SLOT_IDX = 1

BAR_INPUT_C0 = 1
BAR_OUTPUT_C0 = 11
BAR_R0 = 2
BAR_H, BAR_W = 4, 7
BAR_LOCAL_PROJ_COLS = [0, 2, 4, 6]
BAR_BASE_ABS_ROW = BAR_R0 + BAR_H - 1

LEFT_HEIGHT_OPTIONS = (3, 4)
MID_LEFT_HEIGHT = 2
MID_RIGHT_HEIGHT_OPTIONS = (3, 4)
RIGHT_HEIGHT = 1
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
    fill_top_gap: bool,
) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)

    for col, length, count in zip(SLOT_COLS, SLOT_LENS, colored_counts):
        cap = length - count
        if cap > 0:
            grid[SLOT_TOP_ROW : SLOT_TOP_ROW + cap, col] = GREY

    bar_cells = _bar_cells(BAR_OUTPUT_C0, heights)
    if solidify:
        bar_cells = _solidify_row_spans(bar_cells)
    support_by_col = _support_tops(bar_cells)

    for rr, cc in bar_cells:
        grid[rr, cc] = GREY

    for col, length, color, count in zip(SLOT_COLS, SLOT_LENS, colors, colored_counts):
        desired_bottom = support_by_col[col] - 1
        visible = min(count, desired_bottom - SLOT_TOP_ROW + 1)
        if visible <= 0:
            continue

        start_row = desired_bottom - visible + 1
        cap = length - count
        fill_start = SLOT_TOP_ROW + cap
        if fill_top_gap and fill_start < start_row:
            grid[fill_start:start_row, col] = GREY
        grid[start_row : start_row + visible, col] = color
    return grid


def _variant_metrics(
    heights: list[int],
    colors: list[int],
    colored_counts: list[int],
) -> tuple[int, int, int, int, int]:
    raw = _bar_cells(BAR_OUTPUT_C0, heights)
    solid = _solidify_row_spans(raw)
    raw_support = _support_tops(raw)
    solid_support = _support_tops(solid)

    fill_gain = len(set(solid)) - len(set(raw))
    raised_support_cols = sum(solid_support[col] < raw_support[col] for col in SLOT_COLS)

    original_expected = _build_expected_grid(
        heights, colors, colored_counts, solidify=False, fill_top_gap=False
    )
    d7_expected = _build_expected_grid(
        heights, colors, colored_counts, solidify=True, fill_top_gap=False
    )
    combo_expected = _build_expected_grid(
        heights, colors, colored_counts, solidify=True, fill_top_gap=True
    )

    top_gap_fill_cells = int(np.count_nonzero((combo_expected == GREY) & (d7_expected == BG)))
    d7_delta = int(np.count_nonzero(d7_expected != original_expected))
    combo_delta = int(np.count_nonzero(combo_expected != original_expected))
    return fill_gain, raised_support_cols, top_gap_fill_cells, d7_delta, combo_delta


def _sample_params(rng: np.random.Generator) -> tuple[list[int], list[int], list[int]]:
    colors = [int(v) for v in rng.choice(PALETTE, size=len(SLOT_COLS), replace=False)]
    colored_counts = [int(rng.integers(1, length + 1)) for length in SLOT_LENS]
    heights = [
        int(rng.choice(LEFT_HEIGHT_OPTIONS)),
        MID_LEFT_HEIGHT,
        int(rng.choice(MID_RIGHT_HEIGHT_OPTIONS)),
        RIGHT_HEIGHT,
    ]
    return heights, colors, colored_counts


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    heights, colors, colored_counts = _sample_params(rng)
    fill_gain, raised_support_cols, top_gap_fill_cells, d7_delta, combo_delta = _variant_metrics(
        heights, colors, colored_counts
    )
    if fill_gain <= 0 or raised_support_cols <= 0:
        return None
    if top_gap_fill_cells <= 0 or d7_delta <= 0 or combo_delta <= d7_delta:
        return None

    grid = _build_input_grid(heights, colors, colored_counts)
    expected = _build_expected_grid(heights, colors, colored_counts, solidify=True, fill_top_gap=True)

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
