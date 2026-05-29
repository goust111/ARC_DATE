# Variant of 4c3d4a41
# Dimension: D1 — 匹配条件：凸起-列对齐方式 (match_cond)
# Change: 杆的凸起列与框架内彩色列按绝对列位置一一对齐(projection_slot_alignment + column_alignment) → 镜像列对齐

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg as _bg_color
from arc_utils import colors_present, get_bg, paste

# Design analysis
# variant:
#   source_task: 4c416de3
#   borrowed_mechanism: reference solver normalizes one structure, applies a symmetry
#     transform in a canonical frame, then paints the transformed pattern back. This
#     variant uses the same complexity level: detect the movable grey object, mirror it
#     in its own bbox, then place it into the frame.
#   concretization:
#     - The frame, slot columns, colors, canvas size, and bar footprint stay unchanged.
#     - Only the matching condition changes: the bar is no longer inserted as-is.
#     - Before entering the frame, the whole bar is flipped left-right inside its bbox.
#       Because the base row is symmetric, the visible effect comes from the projection
#       heights: input heights [h1,h2,h3,h4] become output heights [h4,h3,h2,h1].
#     - Slot colors keep their original columns; each colored stack is still restacked
#       above the bar column directly underneath it. What changes is which input
#       projection ends up underneath each slot column.
#   behavior_trace:
#     - Original: projections at source cols [1,3,5,7] are translated to slot cols
#       [11,13,15,17], so slot col 11 uses h1, col 13 uses h2, col 15 uses h3, col 17 uses h4.
#     - Variant: reflect the bar first, so those same source projections become
#       [7,5,3,1] before translation. After translation into the slot, col 11 uses h4,
#       col 13 uses h3, col 15 uses h2, col 17 uses h1.
#     - Example: if heights are [4,1,3,2], the original output would place top rows
#       [2,5,3,4] under slot cols [11,13,15,17], while the mirrored variant places
#       [4,3,5,2]. This reverses the left-right height profile without changing colors.
#   complexity_calibration:
#     - The task remains a single global transformation with one movable object and one
#       receiver frame, so it is still ARC-readable.
#     - To keep the mirror rule visually observable, the generator rejects samples whose
#       mirrored output is identical to the old direct-alignment output.
#     - Freedom is controlled: four slot columns, four projection heights in 1..4, and
#       four distinct colors. This is enough repetition for humans to see the reversal
#       but not so many degrees of freedom that correspondence becomes noisy.
#   implicit_assumptions_review:
#     - Grey detection via "most common non-bg" still holds because frame + bar remain grey.
#     - Largest grey component = frame and second-largest = bar still holds because no
#       extra grey objects are introduced.
#     - The old solver assumed horizontal placement was a pure translation; that premise
#       breaks under the variant, so the paste step must consume mirrored bar points.
#     - The old generator assumed output column i depended on height hi. That premise
#       also breaks; output column i must now depend on reversed height order.
#     - The color-restacking code is still valid once the mirrored bar has been pasted,
#       because it already reads bar tops from placed cells column by column.
#   placement_legality:
#     - The bar bbox is always 4x7 with a full base row, so mirroring inside the bbox
#       cannot change its extent or produce overlaps beyond the slot footprint.
#     - The translated mirrored bar still lands at cols 11..17 and base row 5, which
#       fits strictly inside the fixed frame interior.
#     - Slot colored columns remain at 11,13,15,17 with even-column gaps, so all stacks
#       stay separated and solver column grouping is unambiguous.
#   visual_observability:
#     - Humans can see the difference by comparing which tall/short protrusion ends up
#       beneath each colored column. The output's grey skyline is the input skyline read
#       in reverse.
#     - The generator filters out direct-vs-mirror collisions, so the new rule is
#       present in every accepted sample rather than only statistically.
#   cleanliness:
#     - No new marks or overlays are added. The image remains the same two-object layout,
#       so the grid stays tidy and close to the base task.


def _bbox(pts):
    return bbox_coords(pts if isinstance(pts, np.ndarray) else np.asarray(list(pts), dtype=int))


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _most_common(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _mirror_bar_points(bar_pts, bar_bbox):
    """Reflect bar cells left-right inside the bar bbox."""
    _br0, bc0, _br1, bc1 = bar_bbox
    return [(int(r), int(bc0 + bc1 - c)) for r, c in bar_pts]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Insert a grey bar into a grey frame after mirroring it left-right, then restack the slot colors.

    Input structure:
    - The background is dominant; the main structural color is grey and forms two components: a large right-hand frame and a smaller left-hand bar with upward projections.
    - Inside the frame, four colored columns mark the target slots where colors must remain, but their supporting bar projections come from the mirrored bar.

    Rule:
    - Move the bar upward so its base sits one cell above the frame's interior floor, as in the original task.
    - Before placing it, flip the entire bar horizontally within its own bounding box; then translate the mirrored bar into the frame.
    - For each colored slot column, stack that column's colors immediately above the mirrored bar cells now occupying the same column.

    Key insight: the left-to-right correspondence is reversed by symmetry, so the slot columns must be supported by the bar's mirrored projection profile rather than by the original projection order.
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

    (_, slot_bbox, _), (_, bar_bbox, bar_pts) = grey_comps[0], grey_comps[1]
    sr0, sc0, sr1, sc1 = slot_bbox

    out = np.full_like(g, bg)
    out[sr0 : sr1 + 1, sc0 : sc1 + 1] = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region = out[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region[(slot_region != bg) & (slot_region != grey)] = bg

    bar_base_row = max(r for r, _ in bar_pts)
    slot_interior_bottom = sr1 - 1
    target_base_row = slot_interior_bottom - 1
    dy = target_base_row - bar_base_row

    mirrored_bar_pts = _mirror_bar_points(bar_pts, bar_bbox)

    col_min_row = {}
    for r, c in mirrored_bar_pts:
        col_min_row[c] = min(col_min_row.get(c, r), r)
    src_proj_cols = sorted([c for c, rmin in col_min_row.items() if rmin < bar_base_row])

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
    for r, c in mirrored_bar_pts:
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


# --------------- generator (mirrored output-first variant) ---------------

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


def _paint_slot_column(grid: np.ndarray, col: int, length: int, color: int,
                       count: int, height: int | None) -> None:
    cap = length - count
    grid[SLOT_TOP_ROW : SLOT_TOP_ROW + cap, col] = GREY
    if height is None:
        grid[SLOT_TOP_ROW + cap : SLOT_TOP_ROW + length, col] = color
        return
    visible = min(count, BAR_BASE_ABS_ROW - height)
    if visible > 0:
        row_end = BAR_BASE_ABS_ROW - height
        grid[row_end - visible + 1 : row_end + 1, col] = color


def _build_grid(bar_heights: list[int], slot_heights: list[int], colors: list[int],
                colored_counts: list[int], bar_c0: int, output_mode: bool) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    paste(grid, _bar_patch(bar_heights), BAR_R0, bar_c0)
    for col, length, color, count, height in zip(SLOT_COLS, SLOT_LENS, colors, colored_counts, slot_heights):
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
    mirrored_heights = list(reversed(heights))

    grid = _build_grid(heights, heights, colors, colored_counts, BAR_INPUT_C0, output_mode=False)
    expected = _build_grid(mirrored_heights, mirrored_heights, colors, colored_counts, BAR_OUTPUT_C0, output_mode=True)
    direct_expected = _build_grid(heights, heights, colors, colored_counts, BAR_OUTPUT_C0, output_mode=True)

    if np.array_equal(expected, direct_expected):
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
