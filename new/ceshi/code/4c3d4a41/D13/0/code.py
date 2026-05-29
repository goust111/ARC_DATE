# Variant of 4c3d4a41
# Dimension: D13 — 干扰项与噪声 (meta)
# Change: 无干扰项，网格中只有框架、杆和彩色像素三类对象... → 额外的灰色干扰物

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg as _bg_color
from arc_utils import colors_present, get_bg, paste


# Design analysis for the D13 variant
# -----------------------------------
# source_task: none
#
# concretization:
#   grid: still fixed 8x20 with bg=0 and grey=5
#   unchanged objects:
#     - a large grey semi-open frame on the right (largest grey component)
#     - a smaller grey bar on the left with a 7-cell base and 4 upward projections
#     - four colored slot columns inside the frame
#   new object family:
#     - 2 to 4 extra grey distractor components are added in the entrance corridor
#       between the bar and the frame, including empty cells just inside the frame
#       opening; each distractor is a tiny 1- or 2-cell connected component,
#       4-disconnected from the true frame and the true bar
#   output:
#     - unchanged from the original task: only the frame region remains; the bar is
#       inserted into the frame and the colored cells are stacked above the
#       projections; all distractors disappear with the cleared outside region
#
# behavior trace:
#   original input path:
#     - grey components = [frame, bar]
#     - solver can safely take the largest as frame and the second as bar
#   variant input path:
#     - grey components = [frame, bar, noise1, noise2, ...]
#     - the old "there are only two grey components" intuition is no longer valid
#     - the correct bar is still the only non-frame component with a long horizontal
#       base (row fill >= 5) and wide bbox (width >= 5); tiny distractors lack this
#     - after the bar is selected, the translation and color stacking logic stays the same
#
# complexity calibration:
#   - distractors are small enough that the bar remains the second-largest grey
#     component, which preserves the original core task
#   - but there are multiple extra grey blobs in the same left/right interaction lane,
#     so the visual scene is no longer "exactly one frame plus one bar"
#   - the distractor count is capped at 4 and each component area at 2, keeping the
#     nuisance obvious without turning the left half into clutter
#
# implicit-assumption audit of the original main():
#   - grey color detection via "most common non-bg" still holds because grey remains
#     the dominant structural color even after adding a few extra pixels
#   - selecting `grey_comps[1]` as the bar relied on there being only two meaningful
#     grey components; this is the main line that needs adaptation
#   - copying the whole frame bbox is no longer safe, because some distractors sit
#     inside the frame cavity and would survive into the output as stray grey pixels
#   - the output must therefore reconstruct the actual frame component itself, not
#     blindly preserve every grey cell inside the slot bbox
#   - projection extraction, horizontal alignment, and color stacking depend on the
#     chosen bar actually being the long comb-like object; the generator guarantees
#     that only the true bar satisfies the width/base-row signature
#
# placement legality:
#   - distractor templates occupy fixed safe cells near the entrance/interior gap and
#     never touch the bar, the frame, or each other under 4-connectivity
#   - they stay within bounds and off the colored slot columns, so they cannot
#     overwrite the column evidence used for alignment
#
# visual observability:
#   - every generated input has at least two extra grey components, so the D13 change
#     is visible in a small sample set
#   - the added objects are all grey, so no hidden color-id logic is introduced
#
# cleanliness:
#   - distractors stay in a narrow corridor and remain isolated; they do not overlap
#     with the slot columns and the solver clears them before placing the moved bar


def _bbox(pts):
    return bbox_coords(pts if isinstance(pts, np.ndarray) else np.asarray(list(pts), dtype=int))


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _most_common(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _component_info(pts):
    pts = np.asarray(pts, dtype=int)
    bbox = _bbox(pts)
    r0, c0, r1, c1 = bbox
    rows, row_counts = np.unique(pts[:, 0], return_counts=True)
    cols, col_counts = np.unique(pts[:, 1], return_counts=True)
    return {
        "pts": pts,
        "bbox": bbox,
        "area": int(len(pts)),
        "width": int(c1 - c0 + 1),
        "height": int(r1 - r0 + 1),
        "rows": rows,
        "row_counts": row_counts,
        "cols": cols,
        "col_counts": col_counts,
        "max_row_fill": int(row_counts.max()),
    }


def _pick_slot_and_bar(grey_components):
    infos = [_component_info(pts) for pts in grey_components]
    infos.sort(key=lambda d: (-d["area"], d["bbox"]))
    slot = infos[0]
    others = infos[1:]
    if not others:
        return slot, None

    # In the noisy variant, the bar is the only non-frame grey component with
    # a long horizontal footprint and a dense base row.
    def bar_key(info):
        return (
            info["width"] >= 5,
            info["max_row_fill"] >= 5,
            info["area"],
            info["width"],
            -info["height"],
        )

    bar = max(others, key=bar_key)
    return slot, bar


def main(grid: np.ndarray) -> np.ndarray:
    """
    Insert the grey comb-bar into the grey frame, align its prongs with the colored columns, and restack the colors above it.

    Input structure:
    - The dominant structural color is grey, forming one large right-side frame, one left-side comb-shaped bar, and several tiny extra grey distractor blobs in the corridor between them.
    - Colored cells appear only inside the frame and indicate the destination columns that the bar projections must line up with.

    Rule:
    - Ignore the small grey distractors and identify the true bar as the only non-frame grey component with a long horizontal base.
    - Slide that bar into the frame so its base sits one row above the interior floor, while shifting it horizontally until its projections match the colored columns.
    - Re-stack each column's colors immediately above the placed bar in the aligned column, and clear everything outside the frame region.

    Key insight: The extra grey specks are irrelevant noise; the usable object is the wide comb-like grey component whose projection pattern must still key into the colored frame columns.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _bg_color(g)
    other = g[g != bg]
    if other.size == 0:
        return g.copy()
    grey = _most_common(other)

    grey_comps = _cc4(g == grey)
    if len(grey_comps) < 2:
        return g.copy()

    slot_info, bar_info = _pick_slot_and_bar(grey_comps)
    if bar_info is None:
        return g.copy()

    sr0, sc0, sr1, sc1 = slot_info["bbox"]
    bar_pts = bar_info["pts"]

    out = np.full_like(g, bg)
    for r, c in slot_info["pts"]:
        out[r, c] = grey

    row_counts = {}
    for r, _ in bar_pts:
        row_counts[r] = row_counts.get(r, 0) + 1
    bar_base_row = max(row_counts, key=lambda r: (row_counts[r], r))
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

BAR_INPUT_C0 = 1
BAR_OUTPUT_C0 = 11
BAR_R0 = 2
BAR_H, BAR_W = 4, 7
BAR_LOCAL_PROJ_COLS = [0, 2, 4, 6]
BAR_BASE_ABS_ROW = BAR_R0 + BAR_H - 1

HEIGHT0_RANGE = (2, 5)
HEIGHT_REST_RANGE = (1, 5)
MAX_ATTEMPTS = 300

# Safe distractor templates: each entry is one independent 4-connected component.
# They live in the bar-to-frame entrance lane; some sit inside the frame cavity so
# a bbox-copying solver would preserve them incorrectly.
DISTRACTOR_COMPONENTS = (
    ((4, 10), (5, 10)),
    ((4, 12),),
    ((5, 14),),
    ((4, 16), (5, 16)),
)


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


def _add_distractors(grid: np.ndarray, which: list[int]) -> None:
    for idx in which:
        for r, c in DISTRACTOR_COMPONENTS[idx]:
            grid[r, c] = GREY


def _build_grid(heights: list[int], colors: list[int], colored_counts: list[int],
                bar_c0: int, output_mode: bool, distractor_ids: list[int] | None = None) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    paste(grid, _bar_patch(heights), BAR_R0, bar_c0)
    for col, length, color, count, height in zip(SLOT_COLS, SLOT_LENS, colors, colored_counts, heights):
        _paint_slot_column(grid, col, length, color, count, height if output_mode else None)
    if distractor_ids:
        _add_distractors(grid, distractor_ids)
    return grid


def _sample_params(rng: np.random.Generator) -> tuple[list[int], list[int], list[int], list[int]]:
    colors = [int(v) for v in rng.choice(PALETTE, size=len(SLOT_COLS), replace=False)]
    colored_counts = [int(rng.integers(1, L + 1)) for L in SLOT_LENS]
    heights = [int(rng.integers(*HEIGHT0_RANGE))]
    heights.extend(int(rng.integers(*HEIGHT_REST_RANGE)) for _ in range(len(SLOT_COLS) - 1))
    distractor_count = int(rng.integers(2, len(DISTRACTOR_COMPONENTS) + 1))
    distractor_ids = sorted(int(v) for v in rng.choice(len(DISTRACTOR_COMPONENTS), size=distractor_count, replace=False))
    return heights, colors, colored_counts, distractor_ids


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    heights, colors, colored_counts, distractor_ids = _sample_params(rng)
    grid = _build_grid(heights, colors, colored_counts, BAR_INPUT_C0, output_mode=False, distractor_ids=distractor_ids)
    expected = _build_grid(heights, colors, colored_counts, BAR_OUTPUT_C0, output_mode=True)

    if get_bg(grid) != BG:
        return None
    if sorted(colors_present(grid, exclude=[BG, GREY])) != sorted(colors):
        return None
    grey_components = _cc4(grid == GREY)
    if len(grey_components) < 4:
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
