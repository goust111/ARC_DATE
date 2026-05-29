# Variant of 4c3d4a41
# Dimension: D17 — 颜色语义与角色分配 (meta)
# Change: 灰色(5)是结构色（框架和杆），背景色(0)，4种非灰非背景色用于彩色列... → 杆和框架使用不同结构色

import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import bbox_coords, components_mask, get_bg as _bg_color
    from arc_utils import colors_present, get_bg, paste
except ModuleNotFoundError:  # Allow direct file import from the variant folder.
    sys.path.append(str(Path(__file__).resolve().parents[5] / "solvers"))
    from arc_utils import bbox_coords, components_mask, get_bg as _bg_color
    from arc_utils import colors_present, get_bg, paste


# Design analysis for the D17 variant
# a) Concrete behavior trace:
#    - The frame stays the same open rectangle on the right, but it remains grey (5) while the movable bar is blue (1).
#    - The input still contains four colored slot columns inside the frame and one bar outside the frame.
#    - Original solver path: "most common non-bg color" -> find two grey components -> largest is frame, second is bar.
#    - Variant path: shared-color grouping is invalid because blue bar and grey frame are no longer in the same component pool.
#      The solver must instead detect the frame by its hollow/open rectangle geometry and the bar by its full-width base plus
#      upward posts. After those objects are found, the translation and color stacking logic remains unchanged.
#    - Small example at the key branch:
#      original input: frame=5, bar=5, dst cols found by non-{0,5} cells, pasted bar overwrites with 5.
#      variant input: frame=5, bar=1, dst cols are still the non-{0,5} cells in the frame, but the pasted bar must overwrite
#      with 1. Reusing the old unconditional "paint grey" step would silently erase the new D17 distinction.
# b) Complexity calibration:
#    - Only one new visible degree of freedom is introduced: structural roles split across two colors.
#    - Geometry, spacing, counts, and movement stay fixed so the task does not branch into multiple unrelated sub-rules.
#    - The bar color is fixed to blue instead of sampled dynamically. That keeps the variant visually obvious without adding a
#      second hidden inference problem about color remapping.
# c) Implicit-assumption audit of the original main():
#    - `grey = _most_common(other)`: invalid, because the bar is no longer the same color as the frame.
#    - `grey_comps = comps_of_color(grey)`: invalid, because this loses the bar component entirely.
#    - `slot_region[(slot_region != bg) & (slot_region != grey)] = bg`: must become frame-color aware, otherwise the frame
#      clearing rule could accidentally remove or preserve the wrong cells when the structural colors differ.
#    - `out[rr, cc] = grey`: invalid, because the moved bar must retain bar color rather than being recolored as the frame.
#    - `g[r, col] != bg and g[r, col] != grey`: must ignore frame color specifically, not "the only structure color".
# d) Placement legality:
#    - The frame bbox remains rows 0-7, cols 9-19 and the input bar bbox remains rows 2-5, cols 1-7, so no overlap occurs.
#    - The translated bar lands at cols 11-17 and base row 5, fully inside the frame interior footprint used by the original task.
#    - Slot columns stay at cols 11,13,15,17, matching the bar post spacing and avoiding ambiguity in horizontal alignment.
# e) Visual observability:
#    - In 2-3 examples a human can now see a consistent two-color structural split: the right-side container is grey while the
#      moving object is blue and remains blue after insertion.
#    - The movement rule stays readable because the bar shape and the column alignment pattern are unchanged.
# f) Visual cleanliness:
#    - The bar still enters the frame through the left opening and only overlaps the four intended column positions.
#    - Colored columns remain vertical, single-color stacks; the only overlay inside the frame is the same one as the original
#      task, now with a blue bar instead of a grey bar.
# g) Color-rule inferability:
#    - There is no arbitrary recoloring. The frame keeps frame color, the bar keeps bar color, and the four slot columns keep
#      their original colors and order. Color ids are not used for sorting or hidden branching.
# 9) New solver assumptions and generator guarantees:
#    - Assumption: exactly one monochrome component matches the open-frame geometry. Guaranteed by the fixed frame patch.
#    - Assumption: exactly one monochrome component matches the full-base bar geometry. Guaranteed by the fixed bar patch.
#    - Assumption: the bar color is absent from the four slot columns, so "non-bg and non-frame" cells in the slot are exactly
#      the target columns. Guaranteed by sampling column colors from a palette that excludes both 0, 1, and 5.
#    - Assumption: the four destination columns and the four bar posts are equally spaced and require a single dx shift.
#      Guaranteed by using the same column coordinates as the base task.


def _bbox(pts):
    return bbox_coords(pts if isinstance(pts, np.ndarray) else np.asarray(list(pts), dtype=int))


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _local_mask(pts: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in pts:
        mask[int(r) - r0, int(c) - c0] = True
    return mask


def _is_frame_component(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 5 or w < 7:
        return False
    if not mask[0, :].all() or not mask[-1, :].all():
        return False
    if not mask[:, -1].all():
        return False
    if not mask[0, 0] or not mask[-1, 0]:
        return False
    if (~mask[1:-1, 0]).sum() == 0:
        return False
    return True


def _is_bar_component(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 2 or w < 5:
        return False
    if not mask[-1, :].all():
        return False
    if not mask[:-1, :].any():
        return False
    tall_cols = 0
    for c in range(w):
        rows = np.flatnonzero(mask[:, c])
        if rows.size == 0:
            return False
        top = int(rows[0])
        if not mask[top:, c].all():
            return False
        if top < h - 1:
            tall_cols += 1
    return tall_cols >= 2


def _find_frame_and_bar(g: np.ndarray, bg: int):
    comps = []
    for color in colors_present(g, exclude=[bg]):
        for pts in _cc4(g == color):
            bbox = _bbox(pts)
            comps.append(
                {
                    "color": int(color),
                    "pts": pts,
                    "bbox": bbox,
                    "size": int(len(pts)),
                    "mask": _local_mask(pts, bbox),
                }
            )

    frame_candidates = [rec for rec in comps if _is_frame_component(rec["mask"])]
    if not frame_candidates:
        return None, None
    frame_candidates.sort(key=lambda rec: (-rec["size"], rec["bbox"]))
    frame = frame_candidates[0]

    bar_candidates = [rec for rec in comps if rec is not frame and _is_bar_component(rec["mask"])]
    if not bar_candidates:
        return frame, None
    bar_candidates.sort(key=lambda rec: (-rec["size"], rec["bbox"]))
    return frame, bar_candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide a colored bar into a differently colored frame, align its posts with the slot columns, and restack each column above the bar.

    Input structure:
    - Background is 0; the right-hand frame is one structural color, and the detached bar is a different structural color.
    - The frame encloses four vertical colored columns whose occupied columns mark where the bar posts should land.
    - The bar is a separate component with a full horizontal base and upward posts spaced like the slot columns.

    Rule:
    - Identify the open frame and the movable bar by shape rather than by shared color.
    - Translate the bar into the frame so its base sits one row above the frame floor and its posts align with the colored columns.
    - For each aligned column, keep the original top-to-bottom color order and place as many colored cells as fit directly above the moved bar, while the bar keeps its own color.

    Key insight:
    - The structural objects no longer share a color, so the reliable cue is geometric role: the hollow rectangle is the destination frame and the full-base comb shape is the bar to insert.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _bg_color(g)
    frame, bar = _find_frame_and_bar(g, bg)
    if frame is None or bar is None:
        return g.copy()

    frame_color = frame["color"]
    bar_color = bar["color"]
    sr0, sc0, sr1, sc1 = frame["bbox"]
    bar_pts = bar["pts"]

    out = np.full_like(g, bg)
    out[sr0 : sr1 + 1, sc0 : sc1 + 1] = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region = out[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region[(slot_region != bg) & (slot_region != frame_color)] = bg

    bar_base_row = max(int(r) for r, _ in bar_pts)
    slot_interior_bottom = sr1 - 1
    target_base_row = slot_interior_bottom - 1
    dy = target_base_row - bar_base_row

    col_min_row = {}
    for r, c in bar_pts:
        c = int(c)
        r = int(r)
        col_min_row[c] = min(col_min_row.get(c, r), r)
    src_proj_cols = sorted(c for c, rmin in col_min_row.items() if rmin < bar_base_row)

    dst_cols_set = set()
    for r, c in np.argwhere((g != bg) & (g != frame_color)):
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
            out[rr, cc] = bar_color
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

        rows = [r for r in range(sr0, sr1 + 1) if g[r, col] != bg and g[r, col] != frame_color]
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


# --------------- generator ---------------

BG = 0
BAR_COLOR = 1
FRAME_COLOR = 5
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

HEIGHT0_RANGE = (2, 5)
HEIGHT_REST_RANGE = (1, 5)
MAX_ATTEMPTS = 300


def _frame_patch() -> np.ndarray:
    patch = np.zeros((FRAME_H, FRAME_W), dtype=int)
    patch[[0, -1], :] = FRAME_COLOR
    patch[:, -1] = FRAME_COLOR
    patch[:, 0] = FRAME_COLOR
    patch[list(LEFT_WALL_OPEN_ROWS), 0] = BG
    return patch


def _bar_patch(heights: list[int]) -> np.ndarray:
    patch = np.zeros((BAR_H, BAR_W), dtype=int)
    patch[-1, :] = BAR_COLOR
    for local_c, height in zip(BAR_LOCAL_PROJ_COLS, heights):
        patch[BAR_H - height : BAR_H, local_c] = BAR_COLOR
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
    grid[SLOT_TOP_ROW : SLOT_TOP_ROW + cap, col] = FRAME_COLOR
    if height is None:
        grid[SLOT_TOP_ROW + cap : SLOT_TOP_ROW + length, col] = color
        return
    visible = min(count, BAR_BASE_ABS_ROW - height)
    if visible > 0:
        row_end = BAR_BASE_ABS_ROW - height
        grid[row_end - visible + 1 : row_end + 1, col] = color


def _build_grid(
    heights: list[int],
    colors: list[int],
    colored_counts: list[int],
    bar_c0: int,
    output_mode: bool,
) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    if not output_mode:
        paste(grid, _bar_patch(heights), BAR_R0, bar_c0, transparent=BG)
    for col, length, color, count, height in zip(SLOT_COLS, SLOT_LENS, colors, colored_counts, heights):
        _paint_slot_column(grid, col, length, color, count, height if output_mode else None)
    if output_mode:
        paste(grid, _bar_patch(heights), BAR_R0, bar_c0, transparent=BG)
    return grid


def _sample_params(rng: np.random.Generator) -> tuple[list[int], list[int], list[int]]:
    colors = [int(v) for v in rng.choice(PALETTE, size=len(SLOT_COLS), replace=False)]
    colored_counts = [int(rng.integers(1, length + 1)) for length in SLOT_LENS]
    heights = [int(rng.integers(*HEIGHT0_RANGE))]
    heights.extend(int(rng.integers(*HEIGHT_REST_RANGE)) for _ in range(len(SLOT_COLS) - 1))
    return heights, colors, colored_counts


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    heights, colors, colored_counts = _sample_params(rng)
    grid = _build_grid(heights, colors, colored_counts, BAR_INPUT_C0, output_mode=False)
    expected = _build_grid(heights, colors, colored_counts, BAR_OUTPUT_C0, output_mode=True)

    if get_bg(grid) != BG:
        return None
    if BAR_COLOR in colors or FRAME_COLOR in colors:
        return None
    if sorted(colors_present(grid, exclude=[BG, FRAME_COLOR, BAR_COLOR])) != sorted(colors):
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
