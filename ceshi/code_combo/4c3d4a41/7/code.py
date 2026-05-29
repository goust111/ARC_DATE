# Combo variant of 4c3d4a41 (index 7)
# combo_spec: D1/0+D8/2
# Name: 颜色匹配+多余凸起选择
#
# Design analysis
# a) Concrete behavior and trace:
#    - Keep the original 8x20 canvas, the right-side grey frame, the same 4 slot columns
#      at cols [11,13,15,17], and the same "move bar into frame, then stack colors above it"
#      output grammar.
#    - The bar is now a colored bar: a grey base plus 5 or 6 colored projections. Exactly
#      4 projection colors also appear in the frame; those are the useful projections. The
#      remaining 1 or 2 projection colors are distractors that never appear in the frame.
#    - To make the two dimensions genuinely interact, the useful 4 projections are not fixed
#      to one absolute subset. The bar always contains the 5-column backbone [0,2,4,6,8].
#      The useful subset is either the left block [0,2,4,6] or the right block [2,4,6,8].
#      With 6 projections, one extra adjacent tooth is added near the distractor side:
#        * left-active layouts: [0,2,4,6,8] or [0,2,4,6,7,8]
#        * right-active layouts: [0,2,4,6,8] or [0,1,2,4,6,8]
#    - Small trace:
#        old task: 4 grey projections -> align by absolute columns -> dx is fixed by geometry.
#        fused task: 5-6 colored projections -> identify the 4 colors shared with the frame ->
#                    those 4 columns imply one common dx; unmatched colors are discarded.
#      The difference is visible because the same 5-column backbone admits both dx=11
#      (left-active) and dx=9 (right-active); only color correspondence tells which block is
#      the real one.
# b) Complexity calibration:
#    - The frame still has only 4 colored columns, so the scene remains readable.
#    - The bar adds only 1 or 2 distractor projections, keeping the hypothesis space small:
#      humans only need to decide which side's 4-projection block is valid.
#    - Slot colors are unique, and bar colors are unique as well. This avoids ambiguous
#      many-to-one color matches while preserving a real matching step.
# c) Implicit-assumption audit against the original solver:
#    - "Bar is the second-largest grey component" breaks because projections are colored; the
#      solver must find the bar as the second-largest non-background connected component.
#    - "All projection columns are targets" breaks because 1 or 2 projections are distractors.
#    - "Projection count equals slot-column count" breaks because the bar has 5-6 projections
#      but the frame has 4 target columns.
#    - "Horizontal offset can be inferred from column overlap alone" breaks because the
#      5-column backbone supports two plausible 4-column alignments.
# d) Placement legality:
#    - The input bar stays at cols 0..8, so it remains separate from the frame at cols 9..19.
#    - Any projection at absolute col 8 must have height 2; taller teeth there would touch the
#      frame wall at col 9 and merge the two objects. The generator enforces this.
#    - The output keeps only the matched 7-wide core, which always lands at cols 11..17 inside
#      the frame interior.
# e) Visual observability:
#    - The frame plainly shows 4 colors, while the bar shows 5 or 6 projection colors.
#      Solvers can visually detect that only the shared colors matter.
#    - Because both left-active and right-active samples are generated, examples reveal that
#      absolute position alone is not sufficient; the chosen 4-color subset determines the move.
# f) Visual cleanliness:
#    - Distractors remain on one side of the input bar and disappear in the output, so the scene
#      still consists of two clear assemblies instead of interleaved clutter inside the frame.
# g) Color-inference constraints:
#    - The color rule is structural rather than numeric: "shared color between bar and frame"
#      is the only reason a projection participates. Distractor colors are sampled from the same
#      visible palette but are absent from the frame, which gives a direct visual cue.
# h) Generator guarantees for the new solver:
#    - There are always exactly 4 slot colors and exactly 4 bar colors that match them.
#    - All matched projections imply the same dx, and no distractor color appears in the frame.
#    - The useful matched span is always width 7, so cropping to the minimal matched core is safe.
# i) New solver assumptions to audit:
#    - Each slot column contains a single non-grey color. The generator preserves this.
#    - Each bar projection uses at most one non-grey color above the grey base. The generator
#      preserves this by painting each projection as a solid color.
#    - The 4 shared colors are sufficient to determine the correct dx by majority/mode count.
#      The generator preserves this by making every shared color unique.

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


def _projection_columns_from_points(bar_pts: np.ndarray, base_row: int) -> list[tuple[int, list[int]]]:
    cols: dict[int, list[int]] = {}
    for r, c in bar_pts:
        cols.setdefault(int(c), []).append(int(r))
    return sorted((c, sorted(rows)) for c, rows in cols.items() if min(rows) < base_row)


def _slot_column_info(
    g: np.ndarray,
    bg: int,
    grey: int,
    slot_bbox: tuple[int, int, int, int],
) -> list[tuple[int, int, list[int]]]:
    sr0, sc0, sr1, sc1 = slot_bbox
    info = []
    for c in range(sc0, sc1 + 1):
        rows = [r for r in range(sr0, sr1 + 1) if int(g[r, c]) != bg and int(g[r, c]) != grey]
        if not rows:
            continue
        colors = {int(g[r, c]) for r in rows}
        if len(colors) != 1:
            continue
        info.append((int(c), int(next(iter(colors))), rows))
    info.sort(key=lambda t: t[0])
    return info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Insert a colored multi-projection bar into the frame by matching only the projections whose colors appear in the slot.

    Input structure:
    - Background plus two disconnected non-background assemblies: a right-side grey frame that contains four colored slot columns, and a left-side bar with a grey base and five or six colored upward projections.
    - Exactly four projection colors reappear in the slot columns; the remaining one or two projection colors are distractors with no slot partner.

    Rule:
    - Move the bar upward so its base sits one cell above the frame interior floor.
    - Use color correspondence to find which four projections are valid, then choose the horizontal shift whose matched projections all land under the same-colored slot columns.
    - Keep only the minimal contiguous bar core spanning those matched projections, discard unmatched edge teeth, and restack each slot column's colored cells immediately above its aligned projection.

    Key insight:
    - This is no longer pure column alignment. The bar offers more candidate teeth than the frame can accept, so the correct translation comes from the four shared colors, and the unmatched colored teeth are ignored.
    """
    g = np.asarray(grid, dtype=int)
    _h, w = g.shape

    bg = _bg_color(g)
    other = g[g != bg]
    if other.size == 0:
        return g.copy()
    grey = _most_common(other)

    comps = [(len(pts), _bbox(pts), pts) for pts in _cc4(g != bg)]
    comps.sort(key=lambda t: (-t[0], t[1]))
    if len(comps) < 2:
        return g.copy()

    (_, slot_bbox, _slot_pts), (_, _bar_bbox, bar_pts) = comps[0], comps[1]
    sr0, sc0, sr1, sc1 = slot_bbox

    out = np.full_like(g, bg)
    out[sr0 : sr1 + 1, sc0 : sc1 + 1] = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region = out[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region[(slot_region != bg) & (slot_region != grey)] = bg

    bar_base_row = max(int(r) for r, _ in bar_pts)
    target_base_row = (sr1 - 1) - 1
    dy = target_base_row - bar_base_row

    proj_cols = _projection_columns_from_points(bar_pts, bar_base_row)
    proj_info = []
    for c, rows in proj_cols:
        proj_colors = {
            int(g[r, c])
            for r in rows
            if r < bar_base_row and int(g[r, c]) != bg and int(g[r, c]) != grey
        }
        if len(proj_colors) != 1:
            continue
        proj_info.append((c, int(next(iter(proj_colors)))))

    slot_cols = _slot_column_info(g, bg, grey, slot_bbox)
    slot_by_color = {color: (col, rows) for col, color, rows in slot_cols}

    dx_scores: dict[int, int] = {}
    for src_col, color in proj_info:
        if color not in slot_by_color:
            continue
        cand_dx = int(slot_by_color[color][0] - src_col)
        dx_scores[cand_dx] = dx_scores.get(cand_dx, 0) + 1

    matched = []
    if dx_scores:
        dx = max(dx_scores.items(), key=lambda kv: (kv[1], -abs(kv[0]), -kv[0]))[0]
        matched = [
            (src_col, color, slot_by_color[color][0], slot_by_color[color][1])
            for src_col, color in proj_info
            if color in slot_by_color and int(slot_by_color[color][0] - src_col) == dx
        ]
    else:
        dx = 0

    if not matched:
        dst_cols = [col for col, _color, _rows in slot_cols]
        src_cols = [col for col, _color in proj_info]
        best = None
        dst_set = set(dst_cols)
        for cand_dx in range(-w, w + 1):
            shifted = [c + cand_dx for c in src_cols]
            overlap = sum(c in dst_set for c in shifted)
            key = (overlap, -abs(cand_dx), -cand_dx)
            if best is None or key > best[0]:
                best = (key, cand_dx)
        dx = 0 if best is None else int(best[1])
        matched = [
            (src_col, color, slot_by_color[color][0], slot_by_color[color][1])
            for src_col, color in proj_info
            if color in slot_by_color and int(slot_by_color[color][0] - src_col) == dx
        ]

    matched_src_cols = [src_col for src_col, _color, _dst_col, _rows in matched]
    if matched_src_cols:
        keep_c0 = min(matched_src_cols)
        keep_c1 = max(matched_src_cols)
    else:
        all_cols = [int(c) for _, c in bar_pts]
        keep_c0 = min(all_cols)
        keep_c1 = max(all_cols)

    bar_patch_h = 4
    crop_r0 = int(target_base_row - (bar_patch_h - 1))
    crop_r1 = int(target_base_row)
    crop_c0 = int(keep_c0 + dx)
    crop_c1 = int(keep_c1 + dx)
    for rr in range(max(sr0, crop_r0), min(sr1, crop_r1) + 1):
        for cc in range(max(sc0, crop_c0), min(sc1, crop_c1) + 1):
            out[rr, cc] = bg

    placed = []
    for r, c in bar_pts:
        if not (keep_c0 <= int(c) <= keep_c1):
            continue
        rr = int(r + dy)
        cc = int(c + dx)
        if sr0 <= rr <= sr1 and sc0 <= cc <= sc1:
            out[rr, cc] = int(g[r, c])
            placed.append((rr, cc))

    placed_by_col: dict[int, list[int]] = {}
    for rr, cc in placed:
        placed_by_col.setdefault(int(cc), []).append(int(rr))

    interior_top = sr0 + 1
    interior_bottom = sr1 - 1

    for _src_col, color, dst_col, rows in matched:
        if dst_col not in placed_by_col:
            continue
        bar_top = min(placed_by_col[dst_col])
        desired_bottom = bar_top - 1
        if desired_bottom < interior_top:
            continue

        colors = [int(g[r, dst_col]) for r in rows]
        max_len = desired_bottom - interior_top + 1
        if max_len <= 0:
            continue
        if len(colors) > max_len:
            colors = colors[-max_len:]

        start_row = max(desired_bottom - len(colors) + 1, interior_top)
        for i, colv in enumerate(colors):
            rr = start_row + i
            if interior_top <= rr <= interior_bottom:
                out[rr, dst_col] = int(colv)

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
BAR_R0 = 2
BAR_H, BAR_W = 4, 9
BAR_BASE_ABS_ROW = BAR_R0 + BAR_H - 1

OUTPUT_CORE_C0 = 11
OUTPUT_CORE_W = 7
OUTPUT_CORE_LOCAL_COLS = [0, 2, 4, 6]

RIGHT_TOUCH_COL = 8
RIGHT_TOUCH_HEIGHT = 2

ACTIVE_HEIGHT_RANGE = (2, 5)
EXTRA_HEIGHT_RANGE = (2, 5)
MAX_ATTEMPTS = 300


def _frame_patch() -> np.ndarray:
    patch = np.zeros((FRAME_H, FRAME_W), dtype=int)
    patch[[0, -1], :] = GREY
    patch[:, -1] = GREY
    patch[:, 0] = GREY
    patch[list(LEFT_WALL_OPEN_ROWS), 0] = BG
    return patch


def _bar_patch(specs: dict[int, tuple[int, int]], width: int) -> np.ndarray:
    patch = np.zeros((BAR_H, width), dtype=int)
    patch[-1, :] = GREY
    for local_c, (height, color) in specs.items():
        patch[BAR_H - height : BAR_H - 1, local_c] = int(color)
    return patch


def _active_core_patch(active_heights: list[int], active_colors: list[int]) -> np.ndarray:
    specs = {
        local_c: (height, color)
        for local_c, height, color in zip(OUTPUT_CORE_LOCAL_COLS, active_heights, active_colors)
    }
    return _bar_patch(specs, OUTPUT_CORE_W)


def _paint_slot_input(grid: np.ndarray, col: int, length: int, color: int, count: int) -> None:
    cap = length - count
    grid[SLOT_TOP_ROW : SLOT_TOP_ROW + cap, col] = GREY
    grid[SLOT_TOP_ROW + cap : SLOT_TOP_ROW + length, col] = int(color)


def _paint_slot_output(grid: np.ndarray, col: int, color: int, count: int, height: int) -> None:
    visible = min(count, BAR_BASE_ABS_ROW - height)
    if visible <= 0:
        return
    row_end = BAR_BASE_ABS_ROW - height
    grid[row_end - visible + 1 : row_end + 1, col] = int(color)


def _layout(active_side: str, extra_count: int) -> tuple[list[int], list[int], list[int]]:
    if active_side == "left":
        active_cols = [0, 2, 4, 6]
        extras = [8] if extra_count == 1 else [7, 8]
    else:
        active_cols = [2, 4, 6, 8]
        extras = [0] if extra_count == 1 else [0, 1]
    full_cols = sorted(active_cols + extras)
    return full_cols, active_cols, extras


def _sample_params(
    rng: np.random.Generator,
) -> tuple[str, int, list[int], list[int], list[int], list[int], list[int]]:
    active_side = str(rng.choice(["left", "right"]))
    extra_count = int(rng.integers(1, 3))
    slot_colors = [int(v) for v in rng.choice(PALETTE, size=len(SLOT_COLS), replace=False)]
    colored_counts = [int(rng.integers(1, length + 1)) for length in SLOT_LENS]

    full_cols, active_cols, extras = _layout(active_side, extra_count)
    distractor_pool = [int(v) for v in PALETTE if int(v) not in slot_colors]
    distractor_colors = [int(v) for v in rng.choice(distractor_pool, size=len(extras), replace=False)]

    height_by_col: dict[int, int] = {}
    for local_c in active_cols:
        if local_c == RIGHT_TOUCH_COL:
            height_by_col[local_c] = RIGHT_TOUCH_HEIGHT
        else:
            height_by_col[local_c] = int(rng.integers(*ACTIVE_HEIGHT_RANGE))
    for local_c in extras:
        if local_c == RIGHT_TOUCH_COL:
            height_by_col[local_c] = RIGHT_TOUCH_HEIGHT
        else:
            height_by_col[local_c] = int(rng.integers(*EXTRA_HEIGHT_RANGE))

    active_heights = [height_by_col[local_c] for local_c in active_cols]
    extra_heights = [height_by_col[local_c] for local_c in extras]
    return (
        active_side,
        extra_count,
        slot_colors,
        colored_counts,
        active_heights,
        distractor_colors,
        extra_heights,
    )


def _build_input(
    active_side: str,
    extra_count: int,
    slot_colors: list[int],
    colored_counts: list[int],
    active_heights: list[int],
    distractor_colors: list[int],
    extra_heights: list[int],
) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)

    full_cols, active_cols, extras = _layout(active_side, extra_count)
    specs: dict[int, tuple[int, int]] = {}
    for local_c, color, height in zip(active_cols, slot_colors, active_heights):
        specs[local_c] = (height, color)
    for local_c, color, height in zip(extras, distractor_colors, extra_heights):
        specs[local_c] = (height, color)

    paste(grid, _bar_patch(specs, BAR_W), BAR_R0, BAR_INPUT_C0)
    for col, length, color, count in zip(SLOT_COLS, SLOT_LENS, slot_colors, colored_counts):
        _paint_slot_input(grid, col, length, color, count)
    return grid


def _build_expected(
    slot_colors: list[int],
    colored_counts: list[int],
    active_heights: list[int],
) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)

    for col, length, count in zip(SLOT_COLS, SLOT_LENS, colored_counts):
        cap = length - count
        grid[SLOT_TOP_ROW : SLOT_TOP_ROW + cap, col] = GREY

    paste(grid, _active_core_patch(active_heights, slot_colors), BAR_R0, OUTPUT_CORE_C0)
    for col, color, count, height in zip(SLOT_COLS, slot_colors, colored_counts, active_heights):
        _paint_slot_output(grid, col, color, count, height)
    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    params = _sample_params(rng)
    (
        active_side,
        extra_count,
        slot_colors,
        colored_counts,
        active_heights,
        distractor_colors,
        extra_heights,
    ) = params
    grid = _build_input(
        active_side,
        extra_count,
        slot_colors,
        colored_counts,
        active_heights,
        distractor_colors,
        extra_heights,
    )
    expected = _build_expected(slot_colors, colored_counts, active_heights)

    if get_bg(grid) != BG:
        return None
    if sorted(colors_present(grid, exclude=[BG, GREY])) != sorted(slot_colors + distractor_colors):
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
