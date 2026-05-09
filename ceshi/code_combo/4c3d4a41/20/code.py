# Combo variant of 4c3d4a41 (index 20)
# combo_spec: D1/3+D16/2
# Name: 镜像+旋转搜索

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg as _bg_color
from arc_utils import colors_present, get_bg, paste

# Design analysis
# combo_variant:
#   source_task: 4c3d4a41
#   combo_spec: D1/3 + D16/2
#   summary:
#     - D1/3 changes the docking relation: after we identify the usable orientation
#       of the movable bar, its projection profile must be mirrored left-right before
#       matching the slot columns.
#     - D16/2 changes the search process: the input bar may itself appear rotated, so
#       the solver cannot assume the observed component is already in the correct
#       docking orientation. It must search the four quarter-turns.
#   concretization:
#     - The frame, slot columns, palette, and output stacking rule stay the same as in
#       the base task.
#     - The left bar is generated from the original 4x7 shape, then rotated in the
#       input by k in {1,2,3}. This keeps the visual object recognizably the same but
#       destroys the old "just translate it" assumption.
#     - During solving, the bar is cropped to a local patch. For each candidate
#       rotation, the patch is rotated back to a trial orientation and then mirrored
#       left-right. Only one trial becomes a legal docking shape: a 4x7 bar with a
#       full bottom base row and four upward projection columns that can align to slot
#       cols [11,13,15,17] via one rigid horizontal shift.
#   behavior_trace:
#     - Example heights: [4,1,3,2]. The canonical bar has tall/short/tall/medium
#       projections on local cols [0,2,4,6].
#     - Suppose the input shows that bar after a 90-degree rotation. If we naively
#       mirror the observed patch, we get a 7x4 sideways object that cannot dock: it
#       has no full bottom base row and its support columns do not match the slot.
#     - Searching rotations fixes this: rotating the input patch by 270 degrees
#       recovers the canonical horizontal bar; mirroring then reverses the height
#       order to [2,3,1,4]. The output slot therefore uses those reversed heights
#       under cols [11,13,15,17], not the unreversed [4,1,3,2].
#   complexity_calibration:
#     - The rotation freedom is deliberately limited to the 4 quarter-turns. No extra
#       flips are introduced beyond the required mirror-after-rotation step.
#     - The slot still has exactly four odd columns and the bar still exposes exactly
#       four relevant projection columns, so the correspondence remains sparse enough
#       for humans to track.
#     - The generator rejects samples where mirror-vs-direct placement would produce
#       the same output, so the mirror dimension remains visible rather than latent.
#   implicit_assumptions_review:
#     - The base solver assumed the detected bar points could be translated directly in
#       global coordinates. That breaks once the input bar may be vertical or upside
#       down, so the new solver first normalizes the bar to a local patch and only
#       places transformed local points.
#     - The old dx computation assumed the observed projection columns were already the
#       intended ones. Now those columns are read only after choosing a candidate
#       rotation and applying the required mirror.
#     - The old code also relied on "bar base row = max row of observed points". That
#       is only meaningful after the correct rotation has been recovered, so base-row
#       checks move inside the candidate validator.
#     - The color restacking logic still survives unchanged once a valid transformed
#       placement has been found, because it only needs the final placed cells per
#       column.
#   placement_legality:
#     - Input legality: the rotated bar is pasted in the left half with anchors chosen
#       so every quarter-turn fits inside the 8x20 canvas and stays disconnected from
#       the frame.
#     - Output legality: only candidates whose cells all stay within the frame bbox
#       are accepted. Vertical 7x4 candidates fail naturally because they cannot meet
#       the full-base-row docking test and do not fit the column pattern.
#     - Slot columns remain separated by blank even columns, so stacking colors above
#       each supported bar column remains unambiguous.
#   visual_observability:
#     - Across examples, the left object visibly changes orientation while the final
#       object inside the frame is always horizontal. That gives the human solver an
#       external cue that a rotation search happened.
#     - The mirrored rule is visible in the skyline inside the frame: the order of
#       tall/short projections in the output is the left-right reverse of the canonical
#       bar recovered from the input.
#   cleanliness:
#     - No extra markers are introduced. The scene still consists of one frame, one
#       movable bar, and four colored slot columns.
#     - The generator keeps the bar and frame in disjoint regions, so the grid remains
#       readable despite the added transform search.
#   inferability_guard:
#     - Rotation is made non-degenerate by sampling input rotations only from {90,180,
#       270}. Mirror is made non-degenerate by rejecting samples whose mirrored output
#       equals the direct non-mirrored output.


def _bbox(pts):
    return bbox_coords(pts if isinstance(pts, np.ndarray) else np.asarray(list(pts), dtype=int))


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _most_common(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _pts_to_patch(bar_pts, bar_bbox):
    br0, bc0, br1, bc1 = map(int, bar_bbox)
    patch = np.zeros((br1 - br0 + 1, bc1 - bc0 + 1), dtype=bool)
    for r, c in bar_pts:
        patch[int(r) - br0, int(c) - bc0] = True
    return patch


def _projection_cols_from_patch(patch: np.ndarray):
    pts = np.argwhere(patch)
    if pts.size == 0:
        return None, None
    base_row = int(pts[:, 0].max())
    if not bool(np.all(patch[base_row, :])):
        return None, None
    col_min_row = {}
    for r, c in pts:
        rr = int(r)
        cc = int(c)
        col_min_row[cc] = min(col_min_row.get(cc, rr), rr)
    proj_cols = sorted(c for c, rmin in col_min_row.items() if rmin < base_row)
    return base_row, proj_cols


def _try_transformed_bar(slot_bbox, dst_cols, candidate_patch: np.ndarray):
    sr0, sc0, sr1, sc1 = map(int, slot_bbox)
    base_row, src_proj_cols = _projection_cols_from_patch(candidate_patch)
    if base_row is None or not src_proj_cols:
        return None
    if len(src_proj_cols) != len(dst_cols):
        return None

    deltas = [d - s for s, d in zip(src_proj_cols, dst_cols)]
    if len(set(deltas)) != 1:
        return None

    target_base_row = (sr1 - 1) - 1
    origin_r = target_base_row - base_row
    origin_c = deltas[0]

    placed = []
    for r, c in np.argwhere(candidate_patch):
        rr = origin_r + int(r)
        cc = origin_c + int(c)
        if not (sr0 <= rr <= sr1 and sc0 <= cc <= sc1):
            return None
        placed.append((rr, cc))
    return placed


def _search_bar_placement(bar_patch: np.ndarray, slot_bbox, dst_cols):
    for rot_k in range(4):
        rotated = np.rot90(bar_patch, rot_k)
        mirrored = np.fliplr(rotated)
        placed = _try_transformed_bar(slot_bbox, dst_cols, mirrored)
        if placed is not None:
            return rot_k, mirrored, placed
    return None, None, None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the usable orientation of a grey bar, mirror its projection profile, then dock it into the grey frame.

    Input structure:
    - The dominant color is background; the main structural color is grey and forms two components: a large right-hand frame and a smaller left-hand bar.
    - The bar may already be rotated by a quarter-turn multiple, so its observed silhouette is not guaranteed to be the docking silhouette.
    - Inside the frame, four colored columns indicate the destination columns that must end up supported by the bar after the correct transform.

    Rule:
    - Crop the movable bar, test all four rotations of that local patch, and for each rotation mirror it left-right.
    - Keep the unique transformed candidate whose full base can sit one cell above the frame floor and whose mirrored projection columns align with the colored slot columns by one horizontal shift.
    - Place that transformed bar into the frame and restack each slot column's colors immediately above the grey cells now occupying the same column.

    Key insight: solving requires two coupled steps rather than one translation: first undo the bar's input rotation by search, then apply the mirrored column correspondence before docking.
    """
    g = np.asarray(grid, dtype=int)

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
    sr0, sc0, sr1, sc1 = map(int, slot_bbox)

    out = np.full_like(g, bg)
    out[sr0 : sr1 + 1, sc0 : sc1 + 1] = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region = out[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region[(slot_region != bg) & (slot_region != grey)] = bg

    dst_cols = sorted(
        {
            int(c)
            for r, c in np.argwhere((g != bg) & (g != grey))
            if sr0 <= int(r) <= sr1 and sc0 <= int(c) <= sc1
        }
    )
    if not dst_cols:
        return out

    bar_patch = _pts_to_patch(bar_pts, bar_bbox)
    _rot_k, _candidate_patch, placed = _search_bar_placement(bar_patch, slot_bbox, dst_cols)
    if placed is None:
        return g.copy()

    for rr, cc in placed:
        out[rr, cc] = grey

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


# --------------- generator (rotation-search + mirrored docking) ---------------

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

BAR_R0 = 2
BAR_OUTPUT_C0 = 11
BAR_H, BAR_W = 4, 7
BAR_LOCAL_PROJ_COLS = [0, 2, 4, 6]
BAR_BASE_ABS_ROW = BAR_R0 + BAR_H - 1

INPUT_ROTATIONS = (1, 2, 3)
INPUT_BAR_ANCHORS = {
    0: (2, 1),
    1: (0, 2),
    2: (2, 1),
    3: (0, 2),
}
_ROTATION_CURSOR = 0

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
    input_bar_patch: np.ndarray,
    output_bar_patch: np.ndarray | None,
    slot_heights: list[int] | None,
    colors: list[int],
    colored_counts: list[int],
    input_anchor: tuple[int, int],
    output_mode: bool,
) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    paste(grid, input_bar_patch, *input_anchor)
    if output_mode:
        paste(grid, output_bar_patch, BAR_R0, BAR_OUTPUT_C0)
    for col, length, color, count, height in zip(
        SLOT_COLS,
        SLOT_LENS,
        colors,
        colored_counts,
        slot_heights if slot_heights is not None else [None] * len(SLOT_COLS),
    ):
        _paint_slot_column(grid, col, length, color, count, height if output_mode else None)
    return grid


def _sample_params(rng: np.random.Generator, forced_input_rot: int | None = None):
    colors = [int(v) for v in rng.choice(PALETTE, size=len(SLOT_COLS), replace=False)]
    colored_counts = [int(rng.integers(1, length + 1)) for length in SLOT_LENS]
    heights = [int(rng.integers(*HEIGHT0_RANGE))]
    heights.extend(int(rng.integers(*HEIGHT_REST_RANGE)) for _ in range(len(SLOT_COLS) - 1))
    input_rot = int(forced_input_rot if forced_input_rot is not None else rng.choice(INPUT_ROTATIONS))
    return heights, colors, colored_counts, input_rot


def _gen_attempt(rng: np.random.Generator, forced_input_rot: int | None = None) -> np.ndarray | None:
    heights, colors, colored_counts, input_rot = _sample_params(rng, forced_input_rot=forced_input_rot)
    mirrored_heights = list(reversed(heights))

    canonical_bar = _bar_patch(heights)
    rotated_input_bar = np.rot90(canonical_bar, input_rot)
    mirrored_output_bar = _bar_patch(mirrored_heights)
    direct_output_bar = _bar_patch(heights)
    input_anchor = INPUT_BAR_ANCHORS[input_rot]

    grid = _build_grid(
        input_bar_patch=rotated_input_bar,
        output_bar_patch=None,
        slot_heights=None,
        colors=colors,
        colored_counts=colored_counts,
        input_anchor=input_anchor,
        output_mode=False,
    )
    expected = _build_grid(
        input_bar_patch=np.zeros_like(rotated_input_bar),
        output_bar_patch=mirrored_output_bar,
        slot_heights=mirrored_heights,
        colors=colors,
        colored_counts=colored_counts,
        input_anchor=input_anchor,
        output_mode=True,
    )
    direct_expected = _build_grid(
        input_bar_patch=np.zeros_like(rotated_input_bar),
        output_bar_patch=direct_output_bar,
        slot_heights=heights,
        colors=colors,
        colored_counts=colored_counts,
        input_anchor=input_anchor,
        output_mode=True,
    )

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
    global _ROTATION_CURSOR
    rng = np.random.default_rng()
    forced_input_rot = int(INPUT_ROTATIONS[_ROTATION_CURSOR % len(INPUT_ROTATIONS)])
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng, forced_input_rot=forced_input_rot)
        if grid is not None:
            _ROTATION_CURSOR += 1
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_ATTEMPTS} attempts")
