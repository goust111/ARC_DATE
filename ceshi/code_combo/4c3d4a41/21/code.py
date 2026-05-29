# Combo variant of 4c3d4a41 (index 21)
# combo_spec: D4/1+D7/3
# Name: 横向填充+触发填充

import numpy as np

from arc_utils import bbox_coords, colors_present, components_mask, get_bg, get_bg as _bg_color, paste


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) Concrete fused behavior + trace
#    - The original task docks the left comb-bar into the right frame and then
#      uses each colored source column only above its matching projection.
#    - D4/1 changes the destination from "above one projection" to the bounded
#      cavity between neighboring projections.
#    - D7/3 inserts an extra structural step after docking: every occupied bar
#      row is horizontally closed into a solid grey span before colors are used.
#    - The fused rule is therefore:
#         1. dock the bar,
#         2. solidify each occupied row into one grey horizontal span,
#         3. use the original colored columns as pairwise sequences and pour
#            them into the local cavities between adjacent projections.
#    - Example with heights [4,4,3,4] and counts [1,1,1,1]:
#         raw docked rows:
#           r2: 11,13,17
#           r3: 11,13,15,17
#           r4: 11,13,15,17
#           r5: 11..17
#         after solidify:
#           r2: 11..17
#           r3: 11..17
#           r4: 11..17
#           r5: 11..17
#         local cavity cells from neighboring projection tops are:
#           gap 11-13 -> col 12 rows 2..4
#           gap 13-15 -> col 14 rows 3..4
#           gap 15-17 -> col 16 rows 3..4
#         The color sequence for gap 11-13 is [A,B]; it paints rows 2,3 of
#         col 12, leaving row 4 grey. So the output shows both the triggered
#         grey base and the horizontal color fill built on top of it.
#
# b) Complexity calibration
#    - Geometry remains the fixed 8x20 frame/bar layout, so object recognition
#      is stable.
#    - To keep the combo visually distinct, the generator enforces:
#         * stage-1 solidification adds visible grey cells,
#         * stage-2 paints at least several cavity cells,
#         * the final output differs from the D4-only output.
#    - Heights are restricted to {3,4}. This keeps cavities large enough to be
#      seen while still leaving a small, learnable state space.
#
# c) Hidden assumptions audited from the original solver
#    - Original same-column stacking assumes support is read from each colored
#      column independently. That is false here; colors must be redistributed by
#      adjacent pairs instead.
#    - Original raw bar cells were sufficient to describe the final scaffold.
#      That also breaks: row-span closure changes the final grey mass.
#    - The frame being the largest grey component and the bar being the second
#      largest still holds, because generation preserves the original layout.
#    - Using the original colored columns only as alignment markers still works,
#      but their colors now become pairwise sequences rather than per-column
#      stacks.
#
# d) Placement legality
#    - The docked bar still lives entirely inside the frame interior.
#    - Row-span solidification only fills between already occupied cells on the
#      same row, so it cannot leave the bar's horizontal extent.
#    - Cavity painting only uses cells between neighboring projection columns
#      and above the base row, all inside the interior.
#
# e) Visual observability
#    - A human sees two changes at once: the comb becomes a stepped grey block,
#      and the colors move into the even-column cavities 12/14/16.
#    - No invisible ordering is used; the sequence is determined by visible
#      left-to-right source columns and their top-to-bottom colored runs.
#
# f) Visual cleanliness
#    - The output contains only three easy-to-separate elements: the grey frame,
#      the grey solidified bar, and cavity colors.
#    - Colors remain monochrome per source column and occupy only narrow cavity
#      columns, so the image stays legible.
#
# g) Change inferability
#    - Distinct source colors let the solver infer which neighboring pair feeds
#      each cavity.
#    - The generator rejects cases where the combo collapses to the D4-only
#      behavior, so the extra solidification step is always behaviorally active.
#
# h) Quantified non-degeneracy target
#    - `fill_gain > 0`: stage 1 must add grey cells beyond the raw comb.
#    - `colored_gap_cells >= 3`: stage 2 must visibly paint cavities.
#    - `combo_vs_d4_diff`: the final combo output must differ from the D4-only
#      gap-fill output on the same input.
#
# i) New solver assumptions and generator guarantees
#    - Solver assumes four colored guide columns sorted left->right correspond
#      to four docked projections sorted left->right.
#    - Solver assumes each source column is monochrome in its colored segment;
#      the generator constructs exactly that.
#    - Solver assumes cavities are defined from raw neighboring projection tops,
#      not from the solidified skyline. The generator uses the same definition
#      when building targets, so this remains consistent.
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
        c0, c1 = min(cols), max(cols)
        for cc in range(c0, c1 + 1):
            solid.add((rr, cc))
    return sorted(solid)


def _projection_tops_from_heights(heights: list[int]) -> list[int]:
    return [BAR_BASE_ABS_ROW - height + 1 for height in heights]


def _gap_fill_cells(
    proj_cols: list[int],
    proj_tops: list[int],
    *,
    base_row: int,
    sr0: int,
    sr1: int,
    sc0: int,
    sc1: int,
) -> list[list[tuple[int, int]]]:
    gaps: list[list[tuple[int, int]]] = []
    for i in range(len(proj_cols) - 1):
        left_proj = int(proj_cols[i])
        right_proj = int(proj_cols[i + 1])
        gap_top = max(int(proj_tops[i]), int(proj_tops[i + 1]))
        cells = [
            (rr, cc)
            for rr in range(gap_top, base_row)
            for cc in range(left_proj + 1, right_proj)
            if sr0 < rr < sr1 and sc0 < cc < sc1
        ]
        gaps.append(cells)
    return gaps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock a grey comb into a grey frame, close it into a grey base, then pour neighboring color pairs into the local cavities.

    Input structure:
    - The scene has a dominant background plus two disconnected grey structures: a large right-hand frame and a smaller left-hand bar with upward projections.
    - Inside the frame, four monochrome colored guide columns identify where the four projections must align after docking.

    Rule:
    - Translate the bar into the frame so its base sits one row above the frame floor and its projection columns align with the colored guide columns.
    - After docking, solidify the comb by filling each occupied row from its leftmost to rightmost bar cell with grey.
    - Clear the original colored guides, then for each neighboring pair of source columns, concatenate their colored cells left column first and pour that sequence top-to-bottom into the cavity between the matching neighboring projections, clipping if necessary.

    Key insight:
    - The transformation is genuinely two-stage: docking creates a scaffold, scaffold closure creates a grey base, and only then do the source colors move horizontally into pairwise cavities instead of staying in their own columns.
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
    target_base_row = (sr1 - 1) - 1
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

    placed_by_col: dict[int, list[int]] = {}
    for rr, cc in placed:
        placed_by_col.setdefault(int(cc), []).append(int(rr))

    projection_cols = []
    projection_tops = []
    for col in sorted(placed_by_col):
        top = min(placed_by_col[col])
        if top < target_base_row:
            projection_cols.append(int(col))
            projection_tops.append(int(top))

    solid = _solidify_row_spans(placed)
    for rr, cc in solid:
        out[rr, cc] = grey

    source_sequences = []
    for col in dst_cols:
        rows = [r for r in range(sr0, sr1 + 1) if g[r, col] != bg and g[r, col] != grey]
        source_sequences.append([int(g[r, col]) for r in rows])

    gap_cells_by_pair = _gap_fill_cells(
        projection_cols,
        projection_tops,
        base_row=target_base_row,
        sr0=sr0,
        sr1=sr1,
        sc0=sc0,
        sc1=sc1,
    )
    pair_count = min(len(gap_cells_by_pair), max(0, len(source_sequences) - 1))
    for i in range(pair_count):
        fill_cells = gap_cells_by_pair[i]
        if not fill_cells:
            continue
        seq = source_sequences[i] + source_sequences[i + 1]
        for (rr, cc), colv in zip(fill_cells, seq[: len(fill_cells)]):
            out[rr, cc] = int(colv)

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

MAX_ATTEMPTS = 1000

# Each listed height branch has at least one count pattern that visibly shows
# both stages of the combo; sampling is balanced across geometry first, then
# across the valid count profiles for that geometry.
HEIGHT_COUNT_OPTIONS: dict[tuple[int, int, int, int], tuple[tuple[int, int, int, int], ...]] = {
    (3, 3, 4, 4): ((1, 1, 1, 1), (1, 2, 1, 1), (2, 1, 1, 1), (2, 2, 1, 1)),
    (3, 4, 3, 4): (
        (1, 1, 1, 1), (1, 1, 1, 2), (1, 1, 2, 1), (1, 1, 2, 2),
        (1, 2, 1, 1), (1, 2, 1, 2), (1, 2, 2, 1), (1, 2, 2, 2),
        (2, 1, 1, 1), (2, 1, 1, 2), (2, 1, 2, 1), (2, 1, 2, 2),
        (2, 2, 1, 1), (2, 2, 1, 2), (2, 2, 2, 1), (2, 2, 2, 2),
    ),
    (3, 4, 4, 3): ((1, 1, 1, 1), (1, 1, 1, 2), (2, 1, 1, 1), (2, 1, 1, 2)),
    (3, 4, 4, 4): ((1, 1, 1, 1), (1, 1, 1, 2), (1, 2, 1, 1), (2, 1, 1, 1), (2, 1, 1, 2), (2, 2, 1, 1)),
    (4, 3, 3, 4): (
        (1, 1, 1, 1), (1, 1, 1, 2), (1, 1, 2, 1), (1, 1, 2, 2),
        (1, 2, 1, 1), (1, 2, 1, 2), (1, 2, 2, 1), (1, 2, 2, 2),
        (2, 1, 1, 1), (2, 1, 1, 2), (2, 1, 2, 1), (2, 1, 2, 2),
        (2, 2, 1, 1), (2, 2, 1, 2), (2, 2, 2, 1), (2, 2, 2, 2),
    ),
    (4, 3, 4, 3): (
        (1, 1, 1, 1), (1, 1, 1, 2), (1, 1, 2, 1), (1, 1, 2, 2),
        (1, 2, 1, 1), (1, 2, 1, 2), (1, 2, 2, 1), (1, 2, 2, 2),
        (2, 1, 1, 1), (2, 1, 1, 2), (2, 1, 2, 1), (2, 1, 2, 2),
        (2, 2, 1, 1), (2, 2, 1, 2), (2, 2, 2, 1), (2, 2, 2, 2),
    ),
    (4, 3, 4, 4): (
        (1, 1, 1, 1), (1, 1, 1, 2), (1, 1, 2, 1), (1, 1, 2, 2),
        (1, 2, 1, 1), (1, 2, 1, 2), (1, 2, 2, 1), (1, 2, 2, 2),
        (2, 1, 1, 1), (2, 1, 1, 2), (2, 1, 2, 1), (2, 1, 2, 2),
        (2, 2, 1, 1), (2, 2, 1, 2), (2, 2, 2, 1), (2, 2, 2, 2),
    ),
    (4, 4, 3, 3): ((1, 1, 1, 1), (1, 1, 1, 2), (1, 1, 2, 1), (1, 1, 2, 2)),
    (4, 4, 3, 4): (
        (1, 1, 1, 1), (1, 1, 1, 2), (1, 1, 2, 1), (1, 1, 2, 2),
        (1, 2, 1, 1), (1, 2, 1, 2), (1, 2, 2, 1), (1, 2, 2, 2),
        (2, 1, 1, 1), (2, 1, 1, 2), (2, 1, 2, 1), (2, 1, 2, 2),
        (2, 2, 1, 1), (2, 2, 1, 2), (2, 2, 2, 1), (2, 2, 2, 2),
    ),
    (4, 4, 4, 3): ((1, 1, 1, 1), (1, 1, 1, 2), (1, 1, 2, 1), (1, 1, 2, 2), (2, 1, 1, 1), (2, 1, 1, 2)),
    (4, 4, 4, 4): ((1, 1, 1, 1), (1, 1, 1, 2), (1, 1, 2, 1), (1, 1, 2, 2), (1, 2, 1, 1), (2, 1, 1, 1), (2, 1, 1, 2), (2, 2, 1, 1)),
}
BALANCED_HEIGHTS = tuple(sorted(HEIGHT_COUNT_OPTIONS))


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


def _paint_slot_caps(grid: np.ndarray, col: int, length: int, count: int) -> None:
    cap = length - count
    grid[SLOT_TOP_ROW : SLOT_TOP_ROW + cap, col] = GREY


def _build_input_grid(heights: list[int], colors: list[int], counts: list[int]) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    paste(grid, _bar_patch(heights), BAR_R0, BAR_INPUT_C0, transparent=BG)
    for col, length, color, count in zip(SLOT_COLS, SLOT_LENS, colors, counts):
        _paint_input_slot_column(grid, col, length, color, count)
    return grid


def _build_gap_fill_expected(heights: list[int], colors: list[int], counts: list[int], *, solidify: bool) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)

    for col, length, count in zip(SLOT_COLS, SLOT_LENS, counts):
        _paint_slot_caps(grid, col, length, count)

    raw_bar_cells = _bar_cells(BAR_OUTPUT_C0, heights)
    draw_cells = _solidify_row_spans(raw_bar_cells) if solidify else raw_bar_cells
    for rr, cc in draw_cells:
        grid[rr, cc] = GREY

    proj_cols = [BAR_OUTPUT_C0 + c for c in BAR_LOCAL_PROJ_COLS]
    proj_tops = _projection_tops_from_heights(heights)
    gap_cells_by_pair = _gap_fill_cells(
        proj_cols,
        proj_tops,
        base_row=BAR_BASE_ABS_ROW,
        sr0=FRAME_R0,
        sr1=FRAME_R0 + FRAME_H - 1,
        sc0=FRAME_C0,
        sc1=FRAME_C0 + FRAME_W - 1,
    )
    for i, fill_cells in enumerate(gap_cells_by_pair):
        seq = [colors[i]] * counts[i] + [colors[i + 1]] * counts[i + 1]
        for (rr, cc), colv in zip(fill_cells, seq[: len(fill_cells)]):
            grid[rr, cc] = int(colv)
    return grid


def _support_tops(cells: list[tuple[int, int]]) -> dict[int, int]:
    tops: dict[int, int] = {}
    for rr, cc in cells:
        rr = int(rr)
        cc = int(cc)
        tops[cc] = rr if cc not in tops else min(tops[cc], rr)
    return tops


def _build_d7_expected(heights: list[int], colors: list[int], counts: list[int]) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)

    for col, length, count in zip(SLOT_COLS, SLOT_LENS, counts):
        _paint_slot_caps(grid, col, length, count)

    solid = _solidify_row_spans(_bar_cells(BAR_OUTPUT_C0, heights))
    support = _support_tops(solid)
    for rr, cc in solid:
        grid[rr, cc] = GREY

    interior_top = FRAME_R0 + 1
    for col, color, count in zip(SLOT_COLS, colors, counts):
        desired_bottom = support[col] - 1
        visible = min(count, desired_bottom - interior_top + 1)
        if visible <= 0:
            continue
        start_row = desired_bottom - visible + 1
        for rr in range(start_row, desired_bottom + 1):
            grid[rr, col] = int(color)
    return grid


def _variant_metrics(heights: list[int], colors: list[int], counts: list[int]) -> dict[str, int | bool]:
    raw = _bar_cells(BAR_OUTPUT_C0, heights)
    solid = _solidify_row_spans(raw)
    combo_expected = _build_gap_fill_expected(heights, colors, counts, solidify=True)
    d4_expected = _build_gap_fill_expected(heights, colors, counts, solidify=False)
    d7_expected = _build_d7_expected(heights, colors, counts)

    proj_cols = [BAR_OUTPUT_C0 + c for c in BAR_LOCAL_PROJ_COLS]
    gap_cells_by_pair = _gap_fill_cells(
        proj_cols,
        _projection_tops_from_heights(heights),
        base_row=BAR_BASE_ABS_ROW,
        sr0=FRAME_R0,
        sr1=FRAME_R0 + FRAME_H - 1,
        sc0=FRAME_C0,
        sc1=FRAME_C0 + FRAME_W - 1,
    )
    colored_gap_cells = sum(min(len(cells), counts[i] + counts[i + 1]) for i, cells in enumerate(gap_cells_by_pair))
    total_gap_capacity = sum(len(cells) for cells in gap_cells_by_pair)

    return {
        "fill_gain": len(set(solid)) - len(set(raw)),
        "colored_gap_cells": colored_gap_cells,
        "total_gap_capacity": total_gap_capacity,
        "residual_gap_cells": total_gap_capacity - colored_gap_cells,
        "combo_vs_d4_diff": not np.array_equal(combo_expected, d4_expected),
        "combo_vs_d7_diff": not np.array_equal(combo_expected, d7_expected),
    }


def _sample_params(rng: np.random.Generator) -> tuple[list[int], list[int], list[int]]:
    height_key = BALANCED_HEIGHTS[int(rng.integers(0, len(BALANCED_HEIGHTS)))]
    count_options = HEIGHT_COUNT_OPTIONS[height_key]
    heights = [int(v) for v in height_key]
    colors = [int(v) for v in rng.choice(PALETTE, size=len(SLOT_COLS), replace=False)]
    counts = [int(v) for v in count_options[int(rng.integers(0, len(count_options)))]]
    return heights, colors, counts


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    heights, colors, counts = _sample_params(rng)
    metrics = _variant_metrics(heights, colors, counts)
    if metrics["fill_gain"] <= 0:
        return None
    if metrics["colored_gap_cells"] < 3:
        return None
    if not metrics["combo_vs_d4_diff"]:
        return None
    if not metrics["combo_vs_d7_diff"]:
        return None

    grid = _build_input_grid(heights, colors, counts)
    expected = _build_gap_fill_expected(heights, colors, counts, solidify=True)

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
