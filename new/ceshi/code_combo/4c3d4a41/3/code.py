# Combo variant of 4c3d4a41 (index 3)
# combo_spec: D2/0+D4/0
# Name: 向下凸起+下方堆叠

"""
design_analysis:
  a_concrete_behavior_and_trace:
    base_layout_kept:
      - Keep the fixed 8x20 canvas, the right-side frame footprint, the four interior target columns,
        and the per-column encoding "grey cap above a monochrome colored run".
      - Keep the bar width 7 and the four active odd columns, but invert the bar vertically so its full
        base becomes the top row and its teeth extend downward.
    fused_change:
      - Because the teeth now point downward, the bar can no longer plausibly enter through the original
        side opening. The frame opening is moved to the top edge, and the bar starts above-left of it.
      - After insertion, each color run must appear immediately below the bottom of its matched tooth and
        continue downward, clipped by the interior floor.
    example_trace:
      - Suppose one slot column has length 4, grey cap 1, and 3 colored cells in the input.
      - Original task: an upward tooth ends under the free region, so those 3 cells are stacked above the
        tooth tip.
      - Fused variant: if the matched downward tooth has height 3 and the inserted base row is 2, the tooth
        occupies rows 2..4, so the colors start at row 5 and can only use rows 5..6. The visible output
        therefore keeps the first 2 colored cells below the tooth rather than the last cells above it.
      - Difference summary: insertion direction flips, tooth direction flips, and the attachment side of the
        colored material flips with it.

  b_complexity_calibration:
    - The task still uses exactly four target columns and four independent tooth heights in 1..4, so column
      matching remains easy to parse and does not explode combinatorially.
    - The top opening plus the downward teeth makes the fusion visually obvious; it cannot be mistaken for the
      original left-entry upward-comb layout.
    - The generator rejects visually weak samples where every visible below-tooth stack would have length 1,
      because those would underspecify the new clipping direction.

  c_implicit_assumption_audit:
    original_assumptions_broken:
      - `bar_base_row = max(r)` is invalid after the inversion; for the new bar the base row is `min(r)`.
      - Projection detection can no longer look for pixels above the base. It must find columns whose grey
        extent continues below the base row.
      - Anchoring the bar near the frame floor, as in the source task, would leave almost no room below the
        teeth. The fused geometry instead anchors the base one row below the interior ceiling.
      - The original stacking logic used the region above the tooth and kept the suffix of overlong color runs;
        the new logic uses the region below the tooth and keeps the prefix that remains visible first.
    assumptions_kept_and_checked:
      - The largest grey component is still the frame and the second-largest is still the free bar.
      - Colored pixels inside the frame still identify the destination columns, so horizontal matching via
        column overlap remains valid.
      - Clearing the frame interior before repainting colors is still safe because output colors are rebuilt
        solely from the original in-frame evidence.

  d_placement_legality:
    - Frame bbox stays fixed at rows 0..7, cols 9..19, with a 7-cell opening on the top edge at cols 11..17.
    - Input bar bbox is rows 0..3, cols 1..7. Output bar bbox is rows 2..5, cols 11..17.
    - The inserted bar lies fully inside the frame bbox and remains disconnected from the input bar because the
      bar is drawn outside the frame in the input and removed from its source location in the output.
    - Colors are painted strictly below each tooth bottom, so they never overwrite grey bar cells.

  e_visual_observability:
    - A human can see the bar hanging above the frame opening in the input, then seated inside the frame in the
      output with teeth pointing downward in both places.
    - Different tooth heights visibly change how many colored cells fit below each tooth, exposing the new
      clipping rule directly.
    - No hidden sort key is used: colors are preserved columnwise and only move because of visible geometry.

  f_visual_cleanliness:
    - The frame uses only odd interior columns for active evidence, leaving even columns blank as separators.
    - Inside the frame, each active column contains at most one grey tooth and one colored stack below it.
    - The top opening and the bar's isolated starting position keep the two structural components easy to
      distinguish in every sample.

  g_inferability:
    - Colors are not remapped. The only transformation is positional and is tied to tooth alignment and tooth
      height, both directly visible.
    - Using four distinct slot colors is enough to make per-column correspondence obvious without adding noise.

  new_main_assumptions_and_generator_guards:
    assumptions:
      - The bar has one contiguous top base row and at least one tooth extending below it.
      - The target colored evidence appears only inside the frame bbox and only in the four odd slot columns.
      - The bar's tooth columns and the slot columns differ by one consistent horizontal offset.
    guaranteed_by_generator:
      - yes for all three assumptions above;
      - additionally, at least one tooth has height >= 2 and at least one visible below-tooth stack has
        length >= 2, preventing degeneration into a nearly trivial case.
"""

import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import bbox_coords, components_mask, get_bg as _bg_color
    from arc_utils import colors_present, get_bg, paste
except ModuleNotFoundError:  # Allow direct import from the combo folder.
    sys.path.append(str(Path(__file__).resolve().parents[4] / "solvers"))
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
    Insert a grey bar with downward teeth through a top opening, then hang each column's colors below the matched tooth.

    Input structure:
    - The dominant color is background; grey forms two connected components: a large frame on the right with a gap on its top edge, and a separate 7-wide bar on the left.
    - The bar's full base is its top row, and four evenly spaced teeth descend from that base.
    - Inside the frame, four colored columns indicate which columns the teeth must align with.

    Rule:
    - Move the bar downward into the frame so its top base row sits one cell below the frame interior ceiling.
    - Shift it horizontally so the downward teeth line up with the columns that contain colors inside the frame.
    - For each aligned column, copy that column's original colored cells and place them immediately below the bottom of the corresponding tooth, clipping any excess that falls past the interior floor.

    Key insight:
    - The matching cue is still column alignment, but the geometry is fully reversed relative to the source task: the bar enters from above, its teeth point downward, and the colors attach below the teeth instead of above them.
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
    interior_top = sr0 + 1
    interior_bottom = sr1 - 1

    out = np.full_like(g, bg)
    out[sr0 : sr1 + 1, sc0 : sc1 + 1] = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region = out[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region[(slot_region != bg) & (slot_region != grey)] = bg

    bar_base_row = min(int(r) for r, _ in bar_pts)
    target_base_row = interior_top + 1
    dy = target_base_row - bar_base_row

    base_cols = sorted(int(c) for r, c in bar_pts if int(r) == bar_base_row)
    src_proj_cols = []
    if base_cols:
        left = base_cols[0]
        right = base_cols[-1]
        if base_cols == list(range(left, right + 1)):
            src_proj_cols = list(range(left, right + 1, 2))

    if not src_proj_cols:
        col_max_row = {}
        for r, c in bar_pts:
            r = int(r)
            c = int(c)
            col_max_row[c] = max(col_max_row.get(c, r), r)
        src_proj_cols = sorted(c for c, rmax in col_max_row.items() if rmax > bar_base_row)

    dst_cols = sorted(
        int(c)
        for c in {
            c
            for r, c in np.argwhere((g != bg) & (g != grey))
            if sr0 <= int(r) <= sr1 and sc0 <= int(c) <= sc1
        }
    )

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

    placed_by_col = {}
    for rr, cc in placed:
        placed_by_col.setdefault(cc, []).append(rr)

    for col in dst_cols:
        if col not in placed_by_col:
            continue
        tooth_bottom = max(placed_by_col[col])
        start_row = tooth_bottom + 1
        if start_row > interior_bottom:
            continue

        rows = [r for r in range(sr0, sr1 + 1) if g[r, col] != bg and g[r, col] != grey]
        if not rows:
            continue
        colors = [int(g[r, col]) for r in rows]

        max_len = interior_bottom - start_row + 1
        if max_len <= 0:
            continue
        if len(colors) > max_len:
            colors = colors[:max_len]

        for i, color in enumerate(colors):
            rr = start_row + i
            if interior_top <= rr <= interior_bottom:
                out[rr, col] = color

    return out


BG = 0
GREY = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)

H, W = 8, 20

FRAME_R0, FRAME_C0 = 0, 9
FRAME_H, FRAME_W = 8, 11
TOP_OPEN_LOCAL_COLS = tuple(range(2, 9))  # absolute cols 11..17
INTERIOR_TOP = FRAME_R0 + 1
INTERIOR_BOTTOM = FRAME_R0 + FRAME_H - 2

SLOT_COLS = [11, 13, 15, 17]
SLOT_LENS = [3, 4, 4, 3]
SLOT_TOP_ROW = 1

BAR_INPUT_R0, BAR_INPUT_C0 = 0, 1
BAR_OUTPUT_R0, BAR_OUTPUT_C0 = 2, 11
BAR_H, BAR_W = 4, 7
BAR_LOCAL_PROJ_COLS = [0, 2, 4, 6]
BAR_BASE_OUTPUT_ROW = BAR_OUTPUT_R0

HEIGHT0_RANGE = (1, 4)  # {1,2,3}: forces at least one column with room for >= 1 colored cell.
HEIGHT_REST_RANGE = (1, 5)
MAX_ATTEMPTS = 500


def _frame_patch() -> np.ndarray:
    patch = np.zeros((FRAME_H, FRAME_W), dtype=int)
    patch[0, :] = GREY
    patch[-1, :] = GREY
    patch[:, 0] = GREY
    patch[:, -1] = GREY
    patch[0, list(TOP_OPEN_LOCAL_COLS)] = BG
    return patch


def _bar_patch(heights: list[int]) -> np.ndarray:
    patch = np.zeros((BAR_H, BAR_W), dtype=int)
    patch[0, :] = GREY
    for local_c, height in zip(BAR_LOCAL_PROJ_COLS, heights):
        patch[:height, local_c] = GREY
    return patch


def _paint_input_slot_column(grid: np.ndarray, col: int, length: int, color: int, count: int) -> None:
    cap = length - count
    grid[SLOT_TOP_ROW : SLOT_TOP_ROW + cap, col] = GREY
    grid[SLOT_TOP_ROW + cap : SLOT_TOP_ROW + length, col] = color


def _paint_output_slot_column(grid: np.ndarray, col: int, length: int, color: int, count: int, height: int) -> None:
    cap = length - count
    grid[SLOT_TOP_ROW : SLOT_TOP_ROW + cap, col] = GREY
    start_row = BAR_BASE_OUTPUT_ROW + height
    visible = min(count, INTERIOR_BOTTOM - start_row + 1)
    if visible > 0:
        grid[start_row : start_row + visible, col] = color


def _build_input_grid(heights: list[int], colors: list[int], colored_counts: list[int]) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    paste(grid, _bar_patch(heights), BAR_INPUT_R0, BAR_INPUT_C0)
    for col, length, color, count in zip(SLOT_COLS, SLOT_LENS, colors, colored_counts):
        _paint_input_slot_column(grid, col, length, color, count)
    return grid


def _build_expected_grid(heights: list[int], colors: list[int], colored_counts: list[int]) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    paste(grid, _bar_patch(heights), BAR_OUTPUT_R0, BAR_OUTPUT_C0)
    for col, length, color, count, height in zip(SLOT_COLS, SLOT_LENS, colors, colored_counts, heights):
        _paint_output_slot_column(grid, col, length, color, count, height)
    return grid


def _sample_params(rng: np.random.Generator) -> tuple[list[int], list[int], list[int]]:
    colors = [int(v) for v in rng.choice(PALETTE, size=len(SLOT_COLS), replace=False)]
    colored_counts = [int(rng.integers(1, length + 1)) for length in SLOT_LENS]
    heights = [int(rng.integers(*HEIGHT0_RANGE))]
    heights.extend(int(rng.integers(*HEIGHT_REST_RANGE)) for _ in range(len(SLOT_COLS) - 1))
    return heights, colors, colored_counts


def _is_observable_sample(heights: list[int], colored_counts: list[int]) -> bool:
    if max(heights) < 2:
        return False
    visible_counts = [min(count, INTERIOR_BOTTOM - (BAR_BASE_OUTPUT_ROW + height) + 1) for count, height in zip(colored_counts, heights)]
    return max(visible_counts) >= 2


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    heights, colors, colored_counts = _sample_params(rng)
    if not _is_observable_sample(heights, colored_counts):
        return None

    input_grid = _build_input_grid(heights, colors, colored_counts)
    expected = _build_expected_grid(heights, colors, colored_counts)

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
