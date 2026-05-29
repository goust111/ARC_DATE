# Combo variant of 4c3d4a41 (index 12)
# combo_spec: D5/0+D13/0
# Name: 保留杆原位轨迹+灰色干扰物

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg as _bg_color
from arc_utils import colors_present, get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# source_task:
#   - base task 4c3d4a41
#   - fused dimensions: D5/0 keep_source + D13/0 grey distractors
#
# concretization:
#   - Keep the original fixed 8x20 geometry so the object grammar remains easy to read:
#       one large grey semi-open frame on the right,
#       one grey comb-like bar on the left,
#       four colored slot columns inside the frame.
#   - Add 2 to 4 tiny grey distractor components in the corridor/opening region.
#     They are 4-disconnected from the true frame and the true bar, and some are
#     placed inside the frame cavity so "copy the whole frame bbox" is no longer safe.
#   - Add a new output-only trace color TRACE=1. After the grey bar moves into the
#     frame, its exact source silhouette is repainted in TRACE instead of disappearing.
#     The trace color is excluded from the slot-column palette, so it is visually
#     dedicated to the "where the bar used to be" role.
#
# behavior trace on a concrete example:
#   - Suppose the input bar heights are [4, 2, 1, 3], the colored counts are
#     [2, 4, 1, 3], and distractors are placed at {(4,10),(5,10)} and {(4,12)}.
#   - Original task:
#       1. pick the large right grey component as the frame
#       2. pick the left comb as the bar
#       3. move the bar into the frame opening and stack each color above the aligned prong
#       4. clear everything outside the frame
#   - Fused task:
#       1. still move the same true bar and restack colors the same way
#       2. but ignore the extra tiny grey components when deciding which grey object is the bar
#       3. after the move, repaint the bar's original silhouette in TRACE
#       4. clear the distractors instead of preserving them
#   - Net difference:
#       the output now has one extra colored source silhouette on the left, while the
#       input can no longer be solved by assuming "the second grey component is the bar"
#       or "every grey cell inside the frame bbox belongs in the output".
#
# complexity calibration:
#   - The distractors are deliberately tiny (area 1 or 2), so they create a real
#     identification step without competing with the true 7-wide bar.
#   - The trace is always present and always uses one dedicated color, making the
#     D5 change obvious across only 2-3 examples.
#   - The bar heights and column counts keep the original low-entropy structure:
#     four aligned projection columns and four aligned color columns. The fusion adds
#     only two orthogonal reasoning demands: object selection and source-retention.
#
# implicit-assumption audit of the original main():
#   - "largest grey component = frame" still holds and is preserved.
#   - "second-largest grey component = bar" no longer holds once distractors exist;
#     the bar must instead be selected by shape: wide bbox and a dense base row.
#   - "copy frame bbox then wipe colors" no longer holds because distractors can sit
#     inside the frame cavity. The combo solver must redraw only the true frame
#     component, then place the moved bar and colors explicitly.
#   - "outside the slot becomes bg" no longer holds because the bar source must remain
#     as a recolored trace. The solver must therefore keep one specific outside object
#     while removing all other outside grey pixels.
#   - "all non-grey, non-bg colors are slot evidence" still holds in the input because
#     TRACE never appears in generated inputs; only the output uses it.
#
# placement legality:
#   - The frame, bar, and slot columns keep their original non-overlapping geometry.
#   - Each distractor template sits in a safe cell set that is within bounds, off the
#     bar, and 4-disconnected from both the frame and every other template.
#   - The source trace is output-only and reuses the bar's old cells, so it never
#     collides with the moved bar or with the frame.
#
# visual observability:
#   - Every sample contains at least two grey distractor components, so the D13 change
#     is visibly present in the input scene.
#   - Every output contains the full trace silhouette, so the D5 change is visibly
#     present even if the bar motion itself looks otherwise familiar.
#   - The trace color is fixed and structurally tied to the exact old bar shape, which
#     gives a human solver enough support to infer "moved object leaves a ghost".
#
# cleanliness and inferability:
#   - The left half contains only two object families after fusion: one large comb and
#     a few tiny grey specks. Their scale contrast keeps the scene readable.
#   - The trace is a single-color silhouette with no overlap or secondary decoration,
#     so the output remains legible rather than cluttered.
#   - No hidden ordering depends on numeric color ids; the only color-role rule is
#     structural and repeated: TRACE marks the old bar position, slot colors mark the
#     destination columns, and grey remains the structural material.


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


def _bar_base_row(bar_pts: np.ndarray) -> int:
    row_counts = {}
    for r, _ in bar_pts:
        row_counts[int(r)] = row_counts.get(int(r), 0) + 1
    return max(row_counts, key=lambda r: (row_counts[r], r))


TRACE = 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the true grey comb into the right frame, leave a colored ghost at its source, and restack the slot colors above the moved comb.

    Input structure:
    - Grey is the dominant structural color and forms one large semi-open frame on the right, one wide comb-like bar on the left, and several tiny extra grey distractor blobs near the entrance corridor.
    - Four colored columns live inside the frame and identify which interior columns the bar projections must align with.

    Rule:
    - Identify the real bar by its long horizontal base, ignoring the tiny grey distractors.
    - Slide that bar into the frame so its base sits one cell above the interior floor and its projections line up with the colored columns; then stack each column's colors directly above the moved projection.
    - Repaint the bar's original silhouette with the dedicated trace color, while removing all distractors and any other exterior leftovers.

    Key insight: The meaningful moving object is still the only wide comb-shaped grey component, but unlike the original task its source does not vanish: it turns into a persistent trace after the comb is inserted into the frame.
    """
    g = np.asarray(grid, dtype=int)
    _, w = g.shape

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
        out[int(r), int(c)] = grey
    for r, c in bar_pts:
        out[int(r), int(c)] = TRACE

    bar_base_row = _bar_base_row(bar_pts)
    target_base_row = (sr1 - 1) - 1
    dy = target_base_row - bar_base_row

    col_min_row = {}
    for r, c in bar_pts:
        c = int(c)
        r = int(r)
        col_min_row[c] = min(col_min_row.get(c, r), r)
    src_proj_cols = sorted(c for c, rmin in col_min_row.items() if rmin < bar_base_row)

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
        rr = int(r) + dy
        cc = int(c) + dx
        if sr0 <= rr <= sr1 and sc0 <= cc <= sc1:
            out[rr, cc] = grey
            placed.append((rr, cc))

    placed_by_col = {}
    for rr, cc in placed:
        placed_by_col.setdefault(int(cc), []).append(int(rr))

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
        for i, color in enumerate(colors):
            rr = start_row + i
            if interior_top <= rr <= interior_bottom:
                out[rr, col] = color

    return out


BG = 0
GREY = 5
PALETTE = np.array([2, 3, 4, 6, 7, 8, 9], dtype=int)

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

# Keep all four prongs visible so the frame-column alignment remains observable
# across a small number of ARC training examples.
HEIGHT0_RANGE = (2, 5)
HEIGHT_REST_RANGE = (2, 5)
MAX_ATTEMPTS = 300

# Each tuple is one tiny 4-connected grey distractor component.
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


def _bar_patch(heights: list[int], color: int = GREY) -> np.ndarray:
    patch = np.zeros((BAR_H, BAR_W), dtype=int)
    patch[-1, :] = color
    for local_c, height in zip(BAR_LOCAL_PROJ_COLS, heights):
        patch[BAR_H - height : BAR_H, local_c] = color
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


def _add_distractors(grid: np.ndarray, which: list[int]) -> None:
    for idx in which:
        for r, c in DISTRACTOR_COMPONENTS[idx]:
            grid[r, c] = GREY


def _build_input_grid(
    heights: list[int],
    colors: list[int],
    colored_counts: list[int],
    distractor_ids: list[int],
) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    paste(grid, _bar_patch(heights, GREY), BAR_R0, BAR_INPUT_C0)
    for col, length, color, count in zip(SLOT_COLS, SLOT_LENS, colors, colored_counts):
        _paint_slot_column(grid, col, length, color, count, None)
    _add_distractors(grid, distractor_ids)
    return grid


def _build_output_grid(
    heights: list[int],
    colors: list[int],
    colored_counts: list[int],
) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    paste(grid, _bar_patch(heights, TRACE), BAR_R0, BAR_INPUT_C0, transparent=BG)
    paste(grid, _bar_patch(heights, GREY), BAR_R0, BAR_OUTPUT_C0, transparent=BG)
    for col, length, color, count, height in zip(SLOT_COLS, SLOT_LENS, colors, colored_counts, heights):
        _paint_slot_column(grid, col, length, color, count, height)
    return grid


def _sample_params(
    rng: np.random.Generator,
) -> tuple[list[int], list[int], list[int], list[int]]:
    colors = [int(v) for v in rng.choice(PALETTE, size=len(SLOT_COLS), replace=False)]
    colored_counts = [int(rng.integers(1, length + 1)) for length in SLOT_LENS]
    heights = [int(rng.integers(*HEIGHT0_RANGE))]
    heights.extend(int(rng.integers(*HEIGHT_REST_RANGE)) for _ in range(len(SLOT_COLS) - 1))
    distractor_count = int(rng.integers(2, len(DISTRACTOR_COMPONENTS) + 1))
    distractor_ids = sorted(
        int(v) for v in rng.choice(len(DISTRACTOR_COMPONENTS), size=distractor_count, replace=False)
    )
    return heights, colors, colored_counts, distractor_ids


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    heights, colors, colored_counts, distractor_ids = _sample_params(rng)
    grid = _build_input_grid(heights, colors, colored_counts, distractor_ids)
    expected = _build_output_grid(heights, colors, colored_counts)

    if get_bg(grid) != BG:
        return None
    if sorted(colors_present(grid, exclude=[BG, GREY])) != sorted(colors):
        return None
    if len(_cc4(grid == GREY)) < 4:
        return None
    if colors_present(expected, exclude=[BG, GREY]) != sorted(colors + [TRACE]):
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
