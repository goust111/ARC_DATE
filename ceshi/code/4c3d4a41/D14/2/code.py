# Variant of 4c3d4a41
# Dimension: D14 — 对接后灰色填充规则 (transform)
# Change: 杆对接后，彩色像素上方到凸起顶部之间无额外灰色填充；灰色帽保持原位... → 灰色帽颜色变为对应彩色列颜色

# Design analysis for the D14 variant
# ----------------------------------
# spec_like:
#   preserved_geometry:
#     grid: fixed 8x20
#     frame: fixed grey semi-open frame at rows 0..7, cols 9..19
#     slot_columns: fixed active interior columns 11,13,15,17 with lengths 3,4,4,3
#     bar: fixed 7-wide grey base with four odd-column projections, translated into the frame
#   changed_transform:
#     before:
#       after docking, each active column kept only its original colored count above the aligned
#       projection; any original grey cap at the top stayed grey.
#     after:
#       once the bar is aligned, each active column transfers its own color upward so every cell
#       from the slot interior top down to the cell immediately above that projection becomes that
#       column color. This visibly absorbs the grey cap into the color column.
#   behavior_trace:
#     example_column:
#       slot_length: 4
#       input_color_count: 2
#       projection_height: 2
#       available_rows_above_projection: rows 1..3
#       original_output: row 1 grey, rows 2..3 colored
#       variant_output: rows 1..3 all colored with the same column color
#     difference:
#       the alignment step is unchanged; only the post-docking fill rule changes from
#       "preserve cap + stack colors low" to "propagate the column's own color through the whole
#       reachable vertical segment".
#   complexity_calibration:
#     - Keep four independently aligned columns and four distinct colors so the task still requires
#       matching the bar to the frame, not just recoloring a static object.
#     - Do not add new objects or color-id-based logic. The only new complexity is spatially
#       constrained color transfer from each column to the cells above its aligned projection.
#   implicit_assumptions_review:
#     - The largest grey component is still the frame and the second-largest grey component is the
#       bar; generation keeps the original fixed layout so this remains valid.
#     - Each active slot column still uses exactly one non-grey color in the input. The solver now
#       reads that column color once and floods the reachable segment with it; generation enforces
#       monochrome colored stacks per column.
#     - Clearing slot colors before repainting is still safe because the variant reconstructs every
#       active output column from the original input color and the placed bar geometry.
#     - Repainting the whole reachable segment would be wrong on wall columns, so the solver only
#       applies it to destination columns discovered from input colored pixels.
#   placement_legality:
#     - The frame, bar, and slot columns retain the original non-overlapping fixed coordinates.
#     - Output painting is restricted to frame-interior rows and the four active odd columns, so
#       no wall overwrite or out-of-bounds write can occur.
#   observability:
#     - Generator sampling is constrained so each puzzle has at least one column where the variant
#       paints more cells than the original solver would, and at least one such column also has an
#       actual grey cap in the input. This prevents silent degeneration back to the base task.
#   visual_cleanliness_and_color_inference:
#     - Each target column stays a single solid color, inherited from its own input colored cells.
#       Distinct colors across the four columns make the recoloring rule easy to compare by eye.

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
    Dock a grey toothed bar into a grey frame, then let each matched slot column absorb its own color upward.

    Input structure:
    - bg plus a grey structural color forming two components: a large semi-open frame on the right and a separate
      horizontal bar with upward projections on the left.
    - Inside the frame, four active columns contain a single column color beneath a possible grey cap; those colors
      identify which frame columns the bar projections must align to.

    Rule:
    - Move the bar into the frame so its base sits one cell above the frame interior floor, and shift it horizontally
      until the projection columns line up with the colored slot columns.
    - For each aligned slot column, read that column's non-grey color from the input and repaint every interior cell
      from the slot top down to the cell just above the aligned projection with that same color.

    Key insight:
    - The bar still acts as the alignment key, but after docking the frame no longer keeps a grey cap above each
      colored stack; instead, each column's own color propagates upward to create a solid monochrome stripe.
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

    bar_base_row = max(r for r, _ in bar_pts)
    slot_interior_bottom = sr1 - 1
    target_base_row = slot_interior_bottom - 1
    dy = target_base_row - bar_base_row

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

    placed_by_col = {}
    for rr, cc in placed:
        placed_by_col.setdefault(cc, []).append(rr)

    interior_top = sr0 + 1
    interior_bottom = sr1 - 1

    for col in dst_cols:
        rows = [r for r in range(sr0, sr1 + 1) if g[r, col] != bg and g[r, col] != grey]
        if not rows or col not in placed_by_col:
            continue
        col_color = _most_common([int(g[r, col]) for r in rows])
        bar_top = min(placed_by_col[col])
        desired_bottom = min(bar_top - 1, interior_bottom)
        if desired_bottom < interior_top:
            continue
        out[interior_top : desired_bottom + 1, col] = col_color

    return out


# --------------- generator (D14 color-transfer output-first) ---------------

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
MAX_ATTEMPTS = 300


def _available_cells(height: int) -> int:
    return BAR_BASE_ABS_ROW - height


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


def _paint_slot_column(
    grid: np.ndarray,
    col: int,
    length: int,
    color: int,
    count: int,
    height: int | None,
) -> None:
    """Input mode keeps a grey cap; output mode transfers the column color through the reachable height."""
    if height is None:
        cap = length - count
        grid[SLOT_TOP_ROW : SLOT_TOP_ROW + cap, col] = GREY
        grid[SLOT_TOP_ROW + cap : SLOT_TOP_ROW + length, col] = color
        return

    row_end = BAR_BASE_ABS_ROW - height
    if row_end >= SLOT_TOP_ROW:
        grid[SLOT_TOP_ROW : row_end + 1, col] = color


def _build_grid(
    heights: list[int],
    colors: list[int],
    colored_counts: list[int],
    bar_c0: int,
    output_mode: bool,
) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    paste(grid, _bar_patch(heights), BAR_R0, bar_c0)
    for col, length, color, count, height in zip(SLOT_COLS, SLOT_LENS, colors, colored_counts, heights):
        _paint_slot_column(grid, col, length, color, count, height if output_mode else None)
    return grid


def _sample_params(rng: np.random.Generator) -> tuple[list[int], list[int], list[int]]:
    while True:
        colors = [int(v) for v in rng.choice(PALETTE, size=len(SLOT_COLS), replace=False)]
        colored_counts = [int(rng.integers(1, L + 1)) for L in SLOT_LENS]
        heights = [int(rng.integers(*HEIGHT0_RANGE))]
        heights.extend(int(rng.integers(*HEIGHT_REST_RANGE)) for _ in range(len(SLOT_COLS) - 1))

        diff_cols = [
            i
            for i, (length, count, height) in enumerate(zip(SLOT_LENS, colored_counts, heights))
            if count < _available_cells(height)
        ]
        cap_transfer_cols = [
            i
            for i, (length, count, height) in enumerate(zip(SLOT_LENS, colored_counts, heights))
            if count < length and _available_cells(height) >= length
        ]
        if len(diff_cols) >= 2 and cap_transfer_cols:
            return heights, colors, colored_counts


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    heights, colors, colored_counts = _sample_params(rng)
    grid = _build_grid(heights, colors, colored_counts, BAR_INPUT_C0, output_mode=False)
    expected = _build_grid(heights, colors, colored_counts, BAR_OUTPUT_C0, output_mode=True)

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
