# Variant of 4c3d4a41
# Dimension: D8 — 凸起数量与间距 (meta)
# Change: 固定4个凸起，等间距位于奇数列（间隔2格），对应4条彩色列... → 凸起数量多于彩色列

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


# Design analysis for the D8 variant
# a) Concrete behavior / trace
#    - The frame and the 4 colored slot columns stay unchanged from the source task.
#    - The bar changes from a 7-wide base with 4 evenly spaced protrusions to a 9-wide base.
#    - Four "matching" protrusions are fixed at local cols 1,3,5,7 so that, after the correct shift,
#      they land on slot cols 11,13,15,17 exactly as before.
#    - One or two extra protrusions are added at local col 0 and/or 8, giving 5-6 protrusions total.
#      These extras are visible in the input, but only the central 4-protrusion core is inserted into
#      the frame. The unmatched edge protrusions disappear instead of moving with the useful core.
#    - Behavior trace:
#      old task: 4 source protrusion cols -> translate so all 4 align with the 4 colored columns.
#      new task: 5-6 source protrusion cols -> choose the translation whose shifted protrusion set covers
#      the 4 colored columns best; only that aligned 4-column core enters the frame and receives color
#      stacks. Difference: alignment is now subset selection with cropping, not one-to-one equality of
#      the entire bar.
# b) Complexity calibration
#    - Extras only appear on the two ends. This is enough to make the selective matching visible,
#      but it avoids combinatorial ambiguity from arbitrary extra protrusion placement.
#    - The bar still has a single grey color and one rigid translation, so the solver only needs one
#      extra idea: ignore unmatched protrusions.
# c) Implicit assumptions from the original solver that break here
#    - The old fast path `len(src_proj_cols) == len(dst_cols)` is no longer sufficient because the bar
#      can have more protrusions than destination columns.
#    - Any logic that assumes the whole bar moves as one piece is now wrong; unmatched protrusions must
#      be discarded instead of entering the frame as plain grey decoration.
#    - Component ordering still works only if frame and bar remain disconnected in the input.
# d) Placement legality
#    - Input bar is pasted at cols 0-8. The frame starts at col 9.
#    - The right extra protrusion is constrained to height 2, so it reaches only row 4. This avoids
#      touching the frame's left wall, which is grey at rows 0-3 and 6-7 but open at rows 4-5.
#    - Output bar is pasted at cols 10-18, fully inside the frame interior, so no clipping is needed.
# e) Visual observability
#    - The extra protrusions are visible at the left and/or right edge of the inserted bar.
#    - Human solvers can see that colors climb only above the 4 protrusions that line up with colored
#      columns; the edge extras remain unused.
# f) Visual cleanliness
#    - The output keeps only the compact 7-cell core, so the frame interior is cleaner than a full
#      9-cell bar with dangling unmatched teeth at the edges.
# g) Color inference
#    - No new color rule is introduced. Colors still keep their source-column identity and only move
#      vertically above matched protrusions.
# h) Generator guarantees for solver assumptions
#    - There are always exactly 4 colored columns in the frame.
#    - There are always 5 or 6 visible protrusions in the bar.
#    - The 4 active protrusions all have height >= 2, so they are detectable as protrusions above
#      the base row and uniquely determine the best horizontal shift.
# i) New solver assumptions to audit
#    - The best horizontal shift is the one with maximum overlap between shifted protrusion cols and
#      colored slot cols; the generator guarantees this optimum is unique.
#    - After the shift is chosen, the useful part of the bar is the minimal contiguous span covering the
#      matched protrusions, and the generator guarantees that span is the intended central core.
#    - The bar is still the second-largest grey connected component.
#    - Every colored slot column has a matched protrusion after the chosen translation.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide a wider grey bar into the frame, but only the protrusions aligned with colored columns matter.

    Input structure:
    - Background plus a grey structural color that forms two components: a frame on the right and a separate bar on the left.
    - The frame still contains exactly four colored columns, while the bar now has five or six upward protrusions on a wider base.

    Rule:
    - Move the bar upward into the frame so its base sits one cell above the frame interior floor.
    - Choose the horizontal shift that makes the largest subset of protrusions coincide with the four colored columns.
    - Insert only the minimal contiguous core covering those matched protrusions, then re-stack each column's colors immediately above its matched protrusion.

    Key insight:
    - The extra protrusions are distractors. The correct move is to isolate the four protrusions that actually match the four target columns and discard the unmatched edge teeth.
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

    bar_base_row = max(r for r, _ in bar_pts)
    slot_interior_bottom = sr1 - 1
    target_base_row = slot_interior_bottom - 1
    dy = target_base_row - bar_base_row

    col_min_row = {}
    for r, c in bar_pts:
        col_min_row[c] = min(col_min_row.get(c, r), r)
    src_proj_cols = sorted(c for c, rmin in col_min_row.items() if rmin < bar_base_row)

    dst_cols = sorted(
        {
            int(c)
            for r, c in np.argwhere((g != bg) & (g != grey))
            if sr0 <= int(r) <= sr1 and sc0 <= int(c) <= sc1
        }
    )

    dx = 0
    dst_set = set(dst_cols)
    if src_proj_cols and dst_cols:
        best_key = None
        for cand_dx in range(-w, w + 1):
            shifted = [c + cand_dx for c in src_proj_cols]
            overlap = sum(c in dst_set for c in shifted)
            center_cost = abs(np.median(shifted) - np.median(dst_cols))
            key = (overlap, -center_cost, -abs(cand_dx))
            if best_key is None or key > best_key:
                best_key = key
                dx = cand_dx

    matched_src_cols = [c for c in src_proj_cols if (c + dx) in dst_set]
    if matched_src_cols:
        keep_c0 = min(matched_src_cols)
        keep_c1 = max(matched_src_cols)
    else:
        keep_c0 = min(c for _, c in bar_pts)
        keep_c1 = max(c for _, c in bar_pts)

    placed = []
    for r, c in bar_pts:
        if not (keep_c0 <= c <= keep_c1):
            continue
        rr = r + dy
        cc = c + dx
        if sr0 <= rr <= sr1 and sc0 <= cc <= sc1:
            out[rr, cc] = grey
            placed.append((rr, cc))

    placed_by_col = {}
    for rr, cc in placed:
        placed_by_col.setdefault(cc, []).append(rr)

    interior_top = sr0 + 1
    interior_bottom = sr1 - 1

    for col in dst_cols:
        if col not in placed_by_col:
            continue
        bar_top = min(placed_by_col[col])
        desired_bottom = bar_top - 1
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

        start_row = max(desired_bottom - len(colors) + 1, interior_top)
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

BAR_INPUT_C0 = 0
BAR_OUTPUT_C0 = 11
BAR_R0 = 2
BAR_H, BAR_W = 4, 9
ACTIVE_LOCAL_PROJ_COLS = [1, 3, 5, 7]
LEFT_EXTRA_LOCAL_COL = 0
RIGHT_EXTRA_LOCAL_COL = 8
BAR_BASE_ABS_ROW = BAR_R0 + BAR_H - 1

ACTIVE_HEIGHT_RANGE = (2, 5)
LEFT_EXTRA_HEIGHT_RANGE = (2, 5)
RIGHT_EXTRA_HEIGHT = 2
MAX_ATTEMPTS = 300


def _frame_patch() -> np.ndarray:
    patch = np.zeros((FRAME_H, FRAME_W), dtype=int)
    patch[[0, -1], :] = GREY
    patch[:, -1] = GREY
    patch[:, 0] = GREY
    patch[list(LEFT_WALL_OPEN_ROWS), 0] = BG
    return patch


def _bar_patch(heights_by_local: dict[int, int]) -> np.ndarray:
    patch = np.zeros((BAR_H, BAR_W), dtype=int)
    patch[-1, :] = GREY
    for local_c, height in heights_by_local.items():
        patch[BAR_H - height : BAR_H, local_c] = GREY
    return patch


def _active_core_patch(active_heights: list[int]) -> np.ndarray:
    patch = np.zeros((BAR_H, 7), dtype=int)
    patch[-1, :] = GREY
    for local_c, height in zip([0, 2, 4, 6], active_heights):
        patch[BAR_H - height : BAR_H, local_c] = GREY
    return patch


def _paint_slot_column(
    grid: np.ndarray,
    col: int,
    length: int,
    color: int,
    count: int,
    height: int | None,
) -> None:
    cap = length - count
    grid[SLOT_TOP_ROW : SLOT_TOP_ROW + cap, col] = GREY
    if height is None:
        grid[SLOT_TOP_ROW + cap : SLOT_TOP_ROW + length, col] = color
        return
    visible = min(count, BAR_BASE_ABS_ROW - height)
    if visible > 0:
        row_end = BAR_BASE_ABS_ROW - height
        grid[row_end - visible + 1 : row_end + 1, col] = color


def _build_grid(
    active_heights: list[int],
    extra_cols: list[int],
    extra_heights: dict[int, int],
    colors: list[int],
    colored_counts: list[int],
    bar_c0: int,
    output_mode: bool,
) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)

    if output_mode:
        paste(grid, _active_core_patch(active_heights), BAR_R0, bar_c0)
    else:
        heights_by_local = {
            local_c: height for local_c, height in zip(ACTIVE_LOCAL_PROJ_COLS, active_heights)
        }
        for local_c in extra_cols:
            heights_by_local[local_c] = extra_heights[local_c]
        paste(grid, _bar_patch(heights_by_local), BAR_R0, bar_c0)

    for col, length, color, count, height in zip(SLOT_COLS, SLOT_LENS, colors, colored_counts, active_heights):
        _paint_slot_column(grid, col, length, color, count, height if output_mode else None)
    return grid


def _sample_params(
    rng: np.random.Generator,
) -> tuple[list[int], list[int], dict[int, int], list[int], list[int]]:
    colors = [int(v) for v in rng.choice(PALETTE, size=len(SLOT_COLS), replace=False)]
    colored_counts = [int(rng.integers(1, length + 1)) for length in SLOT_LENS]
    active_heights = [int(rng.integers(*ACTIVE_HEIGHT_RANGE)) for _ in ACTIVE_LOCAL_PROJ_COLS]

    extra_case = int(rng.integers(0, 3))
    if extra_case == 0:
        extra_cols = [LEFT_EXTRA_LOCAL_COL]
    elif extra_case == 1:
        extra_cols = [RIGHT_EXTRA_LOCAL_COL]
    else:
        extra_cols = [LEFT_EXTRA_LOCAL_COL, RIGHT_EXTRA_LOCAL_COL]

    extra_heights = {}
    if LEFT_EXTRA_LOCAL_COL in extra_cols:
        extra_heights[LEFT_EXTRA_LOCAL_COL] = int(rng.integers(*LEFT_EXTRA_HEIGHT_RANGE))
    if RIGHT_EXTRA_LOCAL_COL in extra_cols:
        extra_heights[RIGHT_EXTRA_LOCAL_COL] = RIGHT_EXTRA_HEIGHT

    return active_heights, extra_cols, extra_heights, colors, colored_counts


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    active_heights, extra_cols, extra_heights, colors, colored_counts = _sample_params(rng)
    grid = _build_grid(
        active_heights,
        extra_cols,
        extra_heights,
        colors,
        colored_counts,
        BAR_INPUT_C0,
        output_mode=False,
    )
    expected = _build_grid(
        active_heights,
        extra_cols,
        extra_heights,
        colors,
        colored_counts,
        BAR_OUTPUT_C0,
        output_mode=True,
    )

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
