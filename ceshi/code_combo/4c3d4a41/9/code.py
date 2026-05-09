# Combo variant of 4c3d4a41 (index 9)
# combo_spec: D4/2+D17/1
# Name: 凸起着色重映射+双结构色

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import bbox_coords, components_mask, get_bg as _bg_color
    from arc_utils import colors_present, get_bg, paste
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[4] / "solvers"))
    from arc_utils import bbox_coords, components_mask, get_bg as _bg_color
    from arc_utils import colors_present, get_bg, paste


# Design analysis for combo D4/2 + D17/1
# a) Concrete behavior trace:
#    - Base task: frame and bar are both grey; the solver finds two grey components, inserts the bar into the frame,
#      and preserves each slot column's own color above the matched projection.
#    - Combo input: the frame stays grey, but the bar base uses a second structure color and each projection's visible
#      vertical part has its own distinct color. The four slot columns still carry four original colors.
#    - Combo output: the bar keeps both its base color and its per-projection colors after docking. For each aligned
#      slot column, only the visible stack length is inherited from the original geometry; the color is replaced by the
#      matched projection color.
#    - Small example: if a slot column originally has 3 red cells and the matched projection is teal with height 3,
#      then only 2 cells fit above the inserted projection. Original task outputs 2 red cells; this combo outputs 2
#      teal cells while the inserted bar itself still shows its separate structure/base color.
# b) Complexity calibration:
#    - Full disjointness between slot colors and projection colors is impossible here because ARC offers only 10 colors
#      and we already reserve one for background plus two structure colors. The generator therefore uses the same set of
#      4 non-structure colors twice: once for the slot columns and once for the projections, but in a deranged order.
#      This keeps the recolor rule obvious because every aligned column changes color, while color variety stays small.
#    - Projection heights are limited to {2,3,4} so every projection exposes at least one colored cell above the base;
#      otherwise the recolor source could become invisible.
#    - At least one column is truncated by the bar height and at least one is not, so the samples still teach both the
#      geometric visibility rule and the recoloring rule together.
# c) Implicit-assumption audit of the original main():
#    - `grey = _most_common(other)` and "second largest grey component is the bar" both fail because the bar is no
#      longer monochrome and no longer shares the frame color.
#    - The bar must now be detected as a multicolor connected component left of the frame, while the frame is still best
#      identified by its open-rectangle shape among monochrome components.
#    - Clearing the slot region can no longer remove "all non-bg/non-grey" indiscriminately as a structural rule; it
#      must preserve frame color only, because bar color and projection colors are all non-grey.
#    - Copying original slot colors into the output is incorrect under D4/2. The solver should keep only per-column
#      counts, then repaint using the destination projection color inferred from the moved bar.
# d) Placement legality:
#    - The fixed frame bbox remains rows 0..7, cols 9..19; the input bar remains rows 2..5, cols 1..7. These objects
#      are disjoint in input space, and the translated bar lands entirely inside the frame interior at cols 11..17.
#    - Output drawing pastes the bar after the slot columns so tall colored projections correctly overwrite any frame-cap
#      cells that would otherwise occupy the same coordinates.
# e) Visual observability:
#    - A human can simultaneously see four role layers: grey frame, blue base bar, colored projections, and differently
#      colored slot columns. The output visibly preserves the bar colors and swaps each column to the matched projection.
#    - Because projection colors are a derangement of the slot-column colors, every sample shows a nontrivial recolor in
#      every column instead of relying on rare accidental mismatches.
# f) Visual cleanliness:
#    - The bar is still a single comb-like object with contiguous columns, so the projection colors read as markings on
#      one object rather than as separate floating tokens.
#    - The only overpainting inside the frame is the intended docking overlap; visible colors remain confined to the
#      aligned odd columns above the inserted projections.
# g) Change inferability:
#    - The recolor mapping is structural rather than numeric: output column i takes the color of the projection that
#      lands directly beneath it. The generator keeps projection colors distinct so the positional mapping is legible.
# 9) New solver assumptions and generator guarantees:
#    - Assumption: exactly one monochrome component matches the open-frame geometry. Guaranteed by the fixed frame patch.
#    - Assumption: exactly one multicolor non-background component lies fully to the left of the frame and matches the
#      full-base bar geometry. Guaranteed by the fixed left-hand bar placement.
#    - Assumption: every projection has a visible colored segment above the base, so a source projection color can be
#      recovered per projection column. Guaranteed by sampling heights only from {2,3,4}.
#    - Assumption: inside the frame, non-background and non-frame cells belong only to the slot columns in the input.
#      Guaranteed because the bar is never pasted inside the frame during input generation.
#    - Assumption: the four projection columns and the four destination slot columns admit one shared horizontal shift.
#      Guaranteed by using the original task's fixed odd-column spacing on both objects.


def _bbox(pts):
    arr = pts if isinstance(pts, np.ndarray) else np.asarray(list(pts), dtype=int)
    return bbox_coords(arr)


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
    mono_comps = []
    for color in colors_present(g, exclude=[bg]):
        for pts in _cc4(g == color):
            bbox = _bbox(pts)
            mono_comps.append(
                {
                    "color": int(color),
                    "pts": pts,
                    "bbox": bbox,
                    "size": int(len(pts)),
                    "mask": _local_mask(pts, bbox),
                }
            )

    frame_candidates = [rec for rec in mono_comps if _is_frame_component(rec["mask"])]
    if not frame_candidates:
        return None, None
    frame_candidates.sort(key=lambda rec: (-rec["size"], rec["bbox"]))
    frame = frame_candidates[0]
    _fr0, fc0, _fr1, _fc1 = frame["bbox"]

    fg_comps = []
    for pts in _cc4(g != bg):
        bbox = _bbox(pts)
        fg_comps.append(
            {
                "pts": pts,
                "bbox": bbox,
                "size": int(len(pts)),
                "mask": _local_mask(pts, bbox),
            }
        )

    bar_candidates = []
    for rec in fg_comps:
        _br0, bc0, _br1, bc1 = rec["bbox"]
        if bc1 < fc0 and _is_bar_component(rec["mask"]):
            bar_candidates.append(rec)
    if not bar_candidates:
        return frame, None
    bar_candidates.sort(key=lambda rec: (-rec["size"], rec["bbox"]))
    return frame, bar_candidates[0]


def _projection_columns_and_colors(
    g: np.ndarray,
    bar_pts: np.ndarray,
    bg: int,
    bar_base_color: int,
) -> tuple[list[int], dict[int, int]]:
    bar_base_row = max(int(r) for r, _ in bar_pts)
    col_min_row: dict[int, int] = {}
    proj_color_by_src: dict[int, int] = {}
    for r, c in bar_pts:
        r = int(r)
        c = int(c)
        col_min_row[c] = min(col_min_row.get(c, r), r)
        if r < bar_base_row:
            val = int(g[r, c])
            if val != bg and val != bar_base_color:
                proj_color_by_src.setdefault(c, val)
    src_proj_cols = sorted(c for c, rmin in col_min_row.items() if rmin < bar_base_row)
    return src_proj_cols, proj_color_by_src


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock a multicolor bar into a differently colored frame, then repaint each surviving slot column with the matched projection color.

    Input structure:
    - A grey half-open frame sits on the right and contains four grey-capped colored columns.
    - A separate bar sits on the left: its horizontal base uses one structure color, while each upward projection carries its own distinct non-structure color above the base.
    - The slot-column colors are not the same role as the projection colors; they are only markers for which columns the projections must align to.

    Rule:
    - Find the open frame and the multicolor bar by shape, then translate the bar so its base rests one row above the frame floor and its projections align with the slot columns.
    - Keep the inserted bar's original colors unchanged, including the different base color and the per-projection colors.
    - For each aligned slot column, keep only as many colored cells as fit above the inserted projection, but repaint those surviving cells to the color of that aligned projection rather than the slot column's old color.

    Key insight:
    - The geometry still decides where the bar goes and how many cells survive, but the output colors now come from the bar itself: the frame, the bar base, the projection colors, and the original slot colors play four separate visual roles.
    """
    g = np.asarray(grid, dtype=int)
    _h, w = g.shape

    bg = _bg_color(g)
    frame, bar = _find_frame_and_bar(g, bg)
    if frame is None or bar is None:
        return g.copy()

    frame_color = int(frame["color"])
    sr0, sc0, sr1, sc1 = frame["bbox"]
    bar_pts = bar["pts"]
    bar_bbox = bar["bbox"]
    bar_base_row = max(int(r) for r, _ in bar_pts)
    bar_base_color = int(g[bar_base_row, bar_bbox[1]])

    out = np.full_like(g, bg)
    out[sr0 : sr1 + 1, sc0 : sc1 + 1] = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region = out[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region[(slot_region != bg) & (slot_region != frame_color)] = bg

    slot_interior_bottom = sr1 - 1
    target_base_row = slot_interior_bottom - 1
    dy = target_base_row - bar_base_row

    src_proj_cols, proj_color_by_src = _projection_columns_and_colors(g, bar_pts, bg, bar_base_color)
    if not src_proj_cols or len(src_proj_cols) != len(proj_color_by_src):
        return g.copy()

    dst_cols = []
    colored_counts_by_col: dict[int, int] = {}
    for col in range(sc0, sc1 + 1):
        count = 0
        for r in range(sr0, sr1 + 1):
            val = int(g[r, col])
            if val != bg and val != frame_color:
                count += 1
        if count > 0:
            dst_cols.append(col)
            colored_counts_by_col[col] = count

    dx = None
    if len(src_proj_cols) == len(dst_cols):
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

    placed_by_col: dict[int, list[int]] = {}
    for r, c in bar_pts:
        rr = int(r) + dy
        cc = int(c) + dx
        if sr0 <= rr <= sr1 and sc0 <= cc <= sc1:
            out[rr, cc] = int(g[int(r), int(c)])
            placed_by_col.setdefault(cc, []).append(rr)

    proj_color_by_dst = {}
    for src_col, color in proj_color_by_src.items():
        dst_col = src_col + dx
        if sc0 <= dst_col <= sc1:
            proj_color_by_dst[dst_col] = color

    interior_top = sr0 + 1
    interior_bottom = sr1 - 1
    for col in dst_cols:
        if col not in placed_by_col or col not in proj_color_by_dst:
            continue
        bar_top = min(placed_by_col[col])
        desired_bottom = bar_top - 1
        if desired_bottom < interior_top:
            continue
        max_len = desired_bottom - interior_top + 1
        visible = min(colored_counts_by_col[col], max_len)
        if visible <= 0:
            continue
        start_row = desired_bottom - visible + 1
        for rr in range(start_row, desired_bottom + 1):
            if interior_top <= rr <= interior_bottom:
                out[rr, col] = proj_color_by_dst[col]

    return out


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

HEIGHT_RANGE = (2, 5)
MAX_ATTEMPTS = 300


def _frame_patch() -> np.ndarray:
    patch = np.zeros((FRAME_H, FRAME_W), dtype=int)
    patch[[0, -1], :] = FRAME_COLOR
    patch[:, -1] = FRAME_COLOR
    patch[:, 0] = FRAME_COLOR
    patch[list(LEFT_WALL_OPEN_ROWS), 0] = BG
    return patch


def _bar_patch(heights: list[int], proj_colors: list[int]) -> np.ndarray:
    patch = np.zeros((BAR_H, BAR_W), dtype=int)
    patch[-1, :] = BAR_COLOR
    for local_c, height, color in zip(BAR_LOCAL_PROJ_COLS, heights, proj_colors):
        patch[BAR_H - height : BAR_H - 1, local_c] = color
    return patch


def _paint_input_slot_column(
    grid: np.ndarray,
    col: int,
    length: int,
    color: int,
    count: int,
) -> None:
    cap = length - count
    grid[SLOT_TOP_ROW : SLOT_TOP_ROW + cap, col] = FRAME_COLOR
    grid[SLOT_TOP_ROW + cap : SLOT_TOP_ROW + length, col] = color


def _paint_output_slot_column(
    grid: np.ndarray,
    col: int,
    length: int,
    color: int,
    count: int,
    height: int,
) -> None:
    cap = length - count
    grid[SLOT_TOP_ROW : SLOT_TOP_ROW + cap, col] = FRAME_COLOR
    visible = min(count, BAR_BASE_ABS_ROW - height)
    if visible <= 0:
        return
    row_end = BAR_BASE_ABS_ROW - height
    grid[row_end - visible + 1 : row_end + 1, col] = color


def _build_grid(
    heights: list[int],
    slot_colors: list[int],
    proj_colors: list[int],
    colored_counts: list[int],
    bar_c0: int,
    output_mode: bool,
) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    if not output_mode:
        paste(grid, _bar_patch(heights, proj_colors), BAR_R0, bar_c0, transparent=BG)
    for col, length, slot_color, proj_color, count, height in zip(
        SLOT_COLS,
        SLOT_LENS,
        slot_colors,
        proj_colors,
        colored_counts,
        heights,
    ):
        if output_mode:
            _paint_output_slot_column(grid, col, length, proj_color, count, height)
        else:
            _paint_input_slot_column(grid, col, length, slot_color, count)
    if output_mode:
        paste(grid, _bar_patch(heights, proj_colors), BAR_R0, bar_c0, transparent=BG)
    return grid


def _sample_params(
    rng: np.random.Generator,
) -> tuple[list[int], list[int], list[int], list[int]]:
    while True:
        slot_colors = [int(v) for v in rng.choice(PALETTE, size=len(SLOT_COLS), replace=False)]
        proj_colors = [int(v) for v in rng.permutation(slot_colors)]
        if any(a == b for a, b in zip(slot_colors, proj_colors)):
            continue

        colored_counts = [int(rng.integers(1, length + 1)) for length in SLOT_LENS]
        heights = [int(rng.integers(*HEIGHT_RANGE)) for _ in SLOT_COLS]

        visible_caps = [BAR_BASE_ABS_ROW - height for height in heights]
        truncated = [count > cap for count, cap in zip(colored_counts, visible_caps)]
        if len(set(heights)) < 2:
            continue
        if not any(truncated):
            continue
        if not any(not flag for flag in truncated):
            continue
        return heights, slot_colors, proj_colors, colored_counts


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    heights, slot_colors, proj_colors, colored_counts = _sample_params(rng)
    grid = _build_grid(
        heights,
        slot_colors,
        proj_colors,
        colored_counts,
        BAR_INPUT_C0,
        output_mode=False,
    )
    expected = _build_grid(
        heights,
        slot_colors,
        proj_colors,
        colored_counts,
        BAR_OUTPUT_C0,
        output_mode=True,
    )

    if get_bg(grid) != BG:
        return None
    if sorted(colors_present(grid, exclude=[BG, FRAME_COLOR, BAR_COLOR])) != sorted(slot_colors):
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
