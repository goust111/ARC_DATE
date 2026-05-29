# Combo variant of 4c3d4a41 (index 17)
# combo_spec: D2/0+D4/0+D11/0
# Name: 全面空间反转

"""
design_analysis:
  fused_goal:
    summary:
      - This combo is the original task mirrored left/right and inverted top/bottom where the bar logic matters.
      - The frame moves to the left half, its opening moves to the right wall, the bar starts on the right,
        its teeth point downward, and the colors relocate below the teeth after insertion.

  concrete_behavior_and_trace:
    objects:
      - Fixed 8x20 canvas, grey frame, grey 7-wide bar, four colored slot columns.
      - Frame bbox is mirrored from cols 9..19 to cols 0..10, so the bar can genuinely sit to its right.
      - Active slot columns are mirrored from 11,13,15,17 to 2,4,6,8, keeping the same odd-spacing pattern.
      - The bar keeps the same 7x4 footprint but its full base is now the TOP row and each odd column tooth
        extends downward with independent heights 1..4.
    trace_example:
      - Suppose a target column has 3 colored cells in the input and the aligned tooth height is 2.
      - Original task: the upward tooth ended at row 4 after insertion, so colors occupied rows 1..3 above it.
      - Fused task: the downward tooth starts at row 2 and ends at row 3 after insertion, so those 3 cells
        occupy rows 4..6 below it instead.
      - The difference is simultaneous: entry side flips rightward, tooth direction flips downward, and color
        attachment flips to the opposite side of the tooth.

  complexity_calibration:
    controlled_freedom:
      - Keep exactly four target columns and four teeth, with the same fixed spacing, so column matching is still
        unambiguous to a human.
      - Keep tooth heights in the small discrete range 1..4 and per-column colored counts in the fixed lengths
        [3,4,4,3], preserving the original task's manageable variability.
    observability_guards:
      - The generator forces at least one tooth height in {1,2,3}, guaranteeing at least one column with two or
        more visible output cells below the tooth. Without that, the inversion could collapse visually.
      - The generator also rejects the all-ones height pattern, so the bar never collapses into a flat strip with
        no visible downward tooth. That keeps D2/0 observable in every training example.
      - Because the frame is relocated to the left half, the right-wall opening and right-side bar placement are
        immediately visible instead of being hidden by a nearly unchanged layout.

  implicit_assumption_audit:
    original_assumptions_broken:
      - The original solver treated the bar base as the LOWEST occupied row; here the base is the highest row,
        so `min(r)` defines the base and projection detection must look for pixels below that row.
      - The original vertical anchor placed the base near the frame floor; that would leave almost no room for
        below-tooth stacking, so the fused variant anchors the top base row one cell below the interior ceiling.
      - The original generator assumed the frame stayed on the right and the bar on the left; D11 invalidates
        that, so both absolute frame and bar coordinates must be mirrored.
    new_assumptions_in_main:
      - The largest grey connected component is still the frame and the second-largest is still the bar.
      - Colored pixels inside the frame occur only in the four target columns and each target column matches one
        tooth after a single horizontal translation.
      - Once the bar is placed, colors for a column are painted strictly below that column's tooth bottom.
    generator_contract:
      - The generator constructs exactly two separated grey components with those relative sizes.
      - It keeps one blank column between frame and bar in the input so the components never merge.
      - It validates `main(input) == expected` on every sampled example before returning it.

  placement_legality:
    frame_and_bar:
      - Frame bbox: rows 0..7, cols 0..10. Interior: rows 1..6, cols 1..9.
      - Input bar bbox: rows 2..5, cols 12..18. Output bar bbox: rows 2..5, cols 2..8.
      - This preserves one blank column between the right wall (col 10) and the input bar (col 12), and one-cell
        interior margins after insertion, so no overlap or out-of-bounds placement occurs.
    color_fit:
      - For an inserted tooth of height h, the first color row is `base_row + h`, leaving `5-h` visible cells
        through interior row 6. Since h is in 1..4, every column has at least one valid landing cell.

  visual_observability_and_cleanliness:
    human_cues:
      - A solver sees the bar on the right of the frame, the only gap on the frame's right wall, and downward
        teeth that visually point toward the eventual colored stacks.
      - Even interior columns remain empty, so the four active columns remain easy to pair with the four teeth.
      - No color id arithmetic or hidden ordering is used: each column keeps its own color and only moves by
        visible geometric alignment.

  color_inference:
    rule:
      - Colors are preserved, not remapped.
      - The only change is geometric relocation to the mirrored/below-tooth position, which is inferable from a
        few examples because column identity is carried by fixed position and monochrome runs.
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
    Slide a grey bar with downward teeth leftward into a left-side frame, then stack each color run below its tooth.

    Input structure:
    - The dominant background surrounds two grey connected components: a large left-side frame whose right wall has
      a two-cell opening, and a smaller bar on the right whose full base is its top row and whose teeth hang down.
    - Four monochrome color runs inside the frame mark the destination columns that the teeth must align with.

    Rule:
    - Move the bar horizontally into the frame and, if needed, vertically so its top base row sits one cell below
      the frame interior ceiling.
    - Align the tooth columns with the colored frame columns using a single horizontal translation.
    - For each aligned column, copy that column's original colored pixels and place them immediately below the
      corresponding tooth, clipped by the remaining space down to the frame floor.

    Key insight:
    - This is the original mechanism under a full spatial reversal: the frame/bar relationship is mirrored left-right,
      the teeth point downward instead of upward, and the colored material attaches on the opposite side of each tooth.
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

    bar_base_row = min(r for r, _ in bar_pts)
    target_base_row = interior_top + 1
    dy = target_base_row - bar_base_row

    col_max_row = {}
    for r, c in bar_pts:
        col_max_row[c] = max(col_max_row.get(c, r), r)
    src_proj_cols = sorted(c for c, rmax in col_max_row.items() if rmax > bar_base_row)

    dst_cols = sorted(
        {
            int(c)
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
        rr = r + dy
        cc = c + dx
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
        desired_top = tooth_bottom + 1
        if desired_top > interior_bottom:
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

        for i, color in enumerate(colors):
            rr = desired_top + i
            if interior_top <= rr <= interior_bottom:
                out[rr, col] = color

    return out


BG = 0
GREY = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)

H, W = 8, 20

FRAME_R0, FRAME_C0 = 0, 0
FRAME_H, FRAME_W = 8, 11
RIGHT_WALL_OPEN_ROWS = (4, 5)
INTERIOR_TOP = FRAME_R0 + 1
INTERIOR_BOTTOM = FRAME_R0 + FRAME_H - 2

SLOT_COLS = [2, 4, 6, 8]
SLOT_LENS = [3, 4, 4, 3]
SLOT_TOP_ROW = 1

BAR_INPUT_R0, BAR_INPUT_C0 = 2, 12
BAR_OUTPUT_R0, BAR_OUTPUT_C0 = 2, 2
BAR_H, BAR_W = 4, 7
BAR_LOCAL_PROJ_COLS = [0, 2, 4, 6]
BAR_BASE_OUTPUT_ROW = BAR_OUTPUT_R0

HEIGHT0_RANGE = (1, 4)  # {1,2,3} so at least one column visibly extends below its tooth.
HEIGHT_REST_RANGE = (1, 5)
MAX_ATTEMPTS = 400


def _frame_patch() -> np.ndarray:
    """Grey frame with a two-cell opening on the right wall."""
    patch = np.zeros((FRAME_H, FRAME_W), dtype=int)
    patch[[0, -1], :] = GREY
    patch[:, 0] = GREY
    patch[:, -1] = GREY
    patch[list(RIGHT_WALL_OPEN_ROWS), -1] = BG
    return patch


def _bar_patch(heights: list[int]) -> np.ndarray:
    """Horizontal top base with four downward teeth at local cols {0,2,4,6}."""
    patch = np.zeros((BAR_H, BAR_W), dtype=int)
    patch[0, :] = GREY
    for local_c, height in zip(BAR_LOCAL_PROJ_COLS, heights):
        patch[:height, local_c] = GREY
    return patch


def _paint_slot_column(
    grid: np.ndarray, col: int, length: int, color: int, count: int, height: int | None
) -> None:
    """Paint one mirrored slot column."""
    cap = length - count
    grid[SLOT_TOP_ROW : SLOT_TOP_ROW + cap, col] = GREY
    if height is None:
        grid[SLOT_TOP_ROW + cap : SLOT_TOP_ROW + length, col] = color
        return

    start_row = BAR_BASE_OUTPUT_ROW + height
    visible = min(count, INTERIOR_BOTTOM - start_row + 1)
    if visible > 0:
        grid[start_row : start_row + visible, col] = color


def _build_grid(
    heights: list[int],
    colors: list[int],
    colored_counts: list[int],
    bar_r0: int,
    bar_c0: int,
    output_mode: bool,
) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    paste(grid, _bar_patch(heights), bar_r0, bar_c0)
    for col, length, color, count, height in zip(SLOT_COLS, SLOT_LENS, colors, colored_counts, heights):
        _paint_slot_column(grid, col, length, color, count, height if output_mode else None)
    return grid


def _sample_params(rng: np.random.Generator) -> tuple[list[int], list[int], list[int]]:
    colors = [int(v) for v in rng.choice(PALETTE, size=len(SLOT_COLS), replace=False)]
    colored_counts = [int(rng.integers(1, length + 1)) for length in SLOT_LENS]
    heights = [int(rng.integers(*HEIGHT0_RANGE))]
    heights.extend(int(rng.integers(*HEIGHT_REST_RANGE)) for _ in range(len(SLOT_COLS) - 1))
    return heights, colors, colored_counts


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    heights, colors, colored_counts = _sample_params(rng)
    grid = _build_grid(heights, colors, colored_counts, BAR_INPUT_R0, BAR_INPUT_C0, output_mode=False)
    expected = _build_grid(heights, colors, colored_counts, BAR_OUTPUT_R0, BAR_OUTPUT_C0, output_mode=True)

    # Reject the visually degenerate case where every tooth height is 1 and the bar becomes flat.
    if max(heights) < 2:
        return None
    visible_counts = [min(count, INTERIOR_BOTTOM - (BAR_BASE_OUTPUT_ROW + height) + 1) for count, height in zip(colored_counts, heights)]
    if max(visible_counts) < 2:
        return None
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
