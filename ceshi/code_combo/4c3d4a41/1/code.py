# Combo variant of 4c3d4a41 (index 1)
# combo_spec: D1/0+D14/2
# Name: 颜色匹配对齐+灰帽变色

# Design analysis
# a) Concrete behavior and trace:
#    - Preserve the original visual grammar: fixed 8x20 canvas, semi-open grey frame on the
#      right, 7-cell bar on the left, and colored slot columns inside the frame.
#    - Fuse the two requested changes by using the D1 geometry: the frame now contains 5 active
#      columns at cols [10,12,14,16,18], while the bar still has only 4 projections.
#    - Exactly one consecutive 4-column block in the frame matches the left->right colors painted
#      on the 4 bar projections; the remaining end column is a distractor and receives no bar
#      projection in the output.
#    - Small trace:
#         input:
#           slot colors by column = [A,B,C,D,E]
#           bar projection colors = [B,C,D,E]
#           so the correct docking block is cols [12,14,16,18], not the left block.
#           suppose the matched column at col 14 has input cap/color = [grey,grey,C,C] and the
#           aligned projection height leaves rows 1..3 above it.
#         old base task:
#           align by absolute position, so the bar would incorrectly choose the left block.
#         D1-only behavior:
#           choose the right block correctly, but keep only the original colored count stacked low,
#           e.g. [grey,C,C] above the projection.
#         fused behavior:
#           choose the right block by color match, then recolor the whole reachable segment above
#           the projection, e.g. [C,C,C].
#      The fused output therefore differs in two linked places: which 4 frame columns are active,
#      and how each matched column absorbs its own color upward.
# b) Complexity calibration:
#    - The only alignment freedom is binary: left block or right block. That is enough to force
#      real color matching without making the search space noisy.
#    - Five slot columns with distinct colors keep the color evidence visible, while four bar
#      projections keep the task close to the source puzzle.
#    - Projection heights stay in {2,3,4}. Height 1 would remove the colored part of a projection
#      and break the visual cue required for D1.
# c) Old-assumption audit:
#    - The original solver assumes every colored slot column is a destination. That fails here
#      because one end column is intentionally unmatched.
#    - The original solver also assumes the bar is a grey component. That fails because the bar
#      projections are recolored; the bar must be found as the second-largest non-background
#      connected component.
#    - The original stacking rule preserves grey caps. That fails because matched columns must now
#      repaint the entire reachable segment above each aligned projection with that column's color.
#    - Distinct slot colors and bar colors are now structural assumptions: they make the matching
#      window unique and human-readable. The generator enforces them.
# d) Placement legality:
#    - The frame remains fixed at rows 0..7, cols 9..19. Active columns at 10,12,14,16,18 do not
#      overlap each other and stay inside the frame interior.
#    - The bar can legally dock only at col 10 or col 12, which keeps its width-7 base fully
#      inside interior cols 10..18.
#    - Output painting is restricted to interior rows 1..6 and matched active columns only.
# e) Visual observability:
#    - Humans can see the same four colors on the bar and inside one consecutive frame block.
#      When the block is the right one, absolute-position heuristics visibly fail.
#    - The generator additionally requires matched columns where the reachable segment is taller
#      than the original colored stack, so the grey-hat recoloring is visible in the examples.
# f) Visual cleanliness:
#    - The distractor is always an end column, so it does not split the matched block.
#    - Each matched output column becomes a solid monochrome stripe above a same-colored projection,
#      which is cleaner than mixing multiple rules inside one column.
# g) Color-inference constraints:
#    - Slot colors are all distinct, and the bar uses exactly one consecutive 4-color subsequence
#      of them. This gives enough structure to infer the color-matching offset from a few examples.
#    - Recoloring is local and self-explanatory: a column only propagates its own existing color,
#      never a global palette mapping.

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
    Dock a color-coded bar into a grey frame by matching projection colors, then let each matched column absorb its own color upward.

    Input structure:
    - bg plus two disconnected non-bg assemblies: a right-side grey frame that already contains five colored slot columns, and a left-side bar whose grey base supports four upward projections painted in distinct colors.
    - Exactly one consecutive 4-column block inside the frame has the same left-to-right color order as the bar projections; the remaining end column is a distractor.

    Rule:
    - Move the bar into the frame so its base sits one cell above the frame interior floor.
    - Choose the horizontal offset by matching projection colors to the unique consecutive 4-column frame block with the same color sequence, then paste the whole bar there.
    - For each matched column, repaint every interior cell from the slot top down to the cell just above the aligned projection with that column's own color, so any grey cap in that reachable segment disappears into a solid monochrome column.

    Key insight: The colored projections first tell you which four frame columns are the true docking targets, and only after that alignment is known can you apply the second rule that transfers each matched column's color upward through its grey cap.
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

    bar_cols: dict[int, list[int]] = {}
    for r, c in bar_pts:
        bar_cols.setdefault(int(c), []).append(int(r))

    proj_info = []
    for c in sorted(bar_cols):
        rows = sorted(bar_cols[c])
        if rows[0] >= bar_base_row:
            continue
        colors = {
            int(g[r, c])
            for r in rows
            if r < bar_base_row and int(g[r, c]) != bg and int(g[r, c]) != grey
        }
        if len(colors) != 1:
            continue
        proj_info.append((int(c), int(next(iter(colors)))))

    slot_cols = []
    for c in range(sc0, sc1 + 1):
        rows = [r for r in range(sr0, sr1 + 1) if int(g[r, c]) != bg and int(g[r, c]) != grey]
        if not rows:
            continue
        colors = {int(g[r, c]) for r in rows}
        if len(colors) != 1:
            continue
        slot_cols.append((int(c), int(next(iter(colors)))))
    slot_cols.sort(key=lambda t: t[0])

    matched_slot_cols: list[tuple[int, int]] = []
    dx = None
    proj_colors = [color for _, color in proj_info]

    if proj_info and len(slot_cols) >= len(proj_info):
        need = len(proj_info)
        for start in range(len(slot_cols) - need + 1):
            window = slot_cols[start : start + need]
            if [color for _, color in window] != proj_colors:
                continue
            deltas = [dst_col - src_col for (src_col, _), (dst_col, _) in zip(proj_info, window)]
            if len(set(deltas)) == 1:
                dx = int(deltas[0])
                matched_slot_cols = window
                break

    if dx is None:
        by_color = {color: col for col, color in slot_cols}
        if proj_info and all(color in by_color for _, color in proj_info):
            deltas = [by_color[color] - src_col for src_col, color in proj_info]
            if len(set(deltas)) == 1:
                dx = int(deltas[0])
                matched_slot_cols = [(by_color[color], color) for _, color in proj_info]

    if dx is None:
        dst_cols = [col for col, _color in slot_cols]
        src_proj_cols = [col for col, _color in proj_info]
        best = (-1, 0)
        dst_set = set(dst_cols)
        for cand_dx in range(-w, w + 1):
            score = sum((c + cand_dx) in dst_set for c in src_proj_cols)
            if score > best[0]:
                best = (score, cand_dx)
        dx = int(best[1])
        placed_targets = {c + dx for c, _color in proj_info}
        matched_slot_cols = [info for info in slot_cols if info[0] in placed_targets]

    placed = []
    for r, c in bar_pts:
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

    for col, col_color in matched_slot_cols:
        if col not in placed_by_col:
            continue
        bar_top = min(placed_by_col[col])
        desired_bottom = min(bar_top - 1, interior_bottom)
        if desired_bottom < interior_top:
            continue
        out[interior_top : desired_bottom + 1, col] = int(col_color)

    return out


# --------------- generator ---------------

BG = 0
GREY = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)

H, W = 8, 20

FRAME_R0, FRAME_C0 = 0, 9
FRAME_H, FRAME_W = 8, 11
LEFT_WALL_OPEN_ROWS = (4, 5)

SLOT_COLS = [10, 12, 14, 16, 18]
SLOT_LENS = [3, 4, 4, 4, 3]
SLOT_TOP_ROW = 1

MATCH_BLOCK_STARTS = [0, 1]
MATCH_BLOCK_LEN = 4

BAR_INPUT_C0 = 1
BAR_R0 = 2
BAR_H, BAR_W = 4, 7
BAR_LOCAL_PROJ_COLS = [0, 2, 4, 6]
BAR_BASE_ABS_ROW = BAR_R0 + BAR_H - 1

HEIGHT_RANGE = (2, 5)
MAX_ATTEMPTS = 300
_NEXT_BRANCH_INDEX = 0


def _available_cells(height: int) -> int:
    return BAR_BASE_ABS_ROW - height


def _frame_patch() -> np.ndarray:
    patch = np.zeros((FRAME_H, FRAME_W), dtype=int)
    patch[[0, -1], :] = GREY
    patch[:, -1] = GREY
    patch[:, 0] = GREY
    patch[list(LEFT_WALL_OPEN_ROWS), 0] = BG
    return patch


def _bar_patch(heights: list[int], proj_colors: list[int]) -> np.ndarray:
    patch = np.zeros((BAR_H, BAR_W), dtype=int)
    patch[-1, :] = GREY
    for local_c, height, color in zip(BAR_LOCAL_PROJ_COLS, heights, proj_colors):
        patch[BAR_H - height : BAR_H - 1, local_c] = int(color)
    return patch


def _paint_slot_input(grid: np.ndarray, col: int, length: int, color: int, count: int) -> None:
    cap = length - count
    grid[SLOT_TOP_ROW : SLOT_TOP_ROW + cap, col] = GREY
    grid[SLOT_TOP_ROW + cap : SLOT_TOP_ROW + length, col] = int(color)


def _paint_slot_output(grid: np.ndarray, col: int, color: int, height: int) -> None:
    row_end = BAR_BASE_ABS_ROW - height
    if row_end >= SLOT_TOP_ROW:
        grid[SLOT_TOP_ROW : row_end + 1, col] = int(color)


def _build_input(
    heights: list[int],
    slot_colors: list[int],
    colored_counts: list[int],
    block_start: int,
) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    for col, length, color, count in zip(SLOT_COLS, SLOT_LENS, slot_colors, colored_counts):
        _paint_slot_input(grid, col, length, color, count)
    bar_colors = slot_colors[block_start : block_start + MATCH_BLOCK_LEN]
    paste(grid, _bar_patch(heights, bar_colors), BAR_R0, BAR_INPUT_C0)
    return grid


def _build_expected(
    heights: list[int],
    slot_colors: list[int],
    colored_counts: list[int],
    block_start: int,
) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)

    for col, length, count in zip(SLOT_COLS, SLOT_LENS, colored_counts):
        cap = length - count
        grid[SLOT_TOP_ROW : SLOT_TOP_ROW + cap, col] = GREY

    bar_colors = slot_colors[block_start : block_start + MATCH_BLOCK_LEN]
    bar_output_c0 = SLOT_COLS[block_start]
    paste(grid, _bar_patch(heights, bar_colors), BAR_R0, bar_output_c0)

    matched_cols = SLOT_COLS[block_start : block_start + MATCH_BLOCK_LEN]
    for col, color, height in zip(matched_cols, bar_colors, heights):
        _paint_slot_output(grid, col, color, height)

    return grid


def _sample_params(
    rng: np.random.Generator, forced_block_start: int | None = None
) -> tuple[list[int], list[int], list[int], int]:
    while True:
        slot_colors = [int(v) for v in rng.choice(PALETTE, size=len(SLOT_COLS), replace=False)]
        colored_counts = [int(rng.integers(1, length + 1)) for length in SLOT_LENS]
        heights = [int(rng.integers(*HEIGHT_RANGE)) for _ in BAR_LOCAL_PROJ_COLS]
        if forced_block_start is None:
            block_start = int(rng.choice(MATCH_BLOCK_STARTS))
        else:
            block_start = int(forced_block_start)

        matched_counts = colored_counts[block_start : block_start + MATCH_BLOCK_LEN]
        diff_cols = [
            i
            for i, (count, height) in enumerate(zip(matched_counts, heights))
            if count < _available_cells(height)
        ]
        if len(diff_cols) < 2:
            continue
        if block_start == 0 and heights[0] == 4:
            continue
        if block_start == 1 and heights[-1] == 4:
            continue
        return heights, slot_colors, colored_counts, block_start


def _gen_attempt(rng: np.random.Generator, forced_block_start: int | None = None) -> np.ndarray | None:
    heights, slot_colors, colored_counts, block_start = _sample_params(rng, forced_block_start)
    grid = _build_input(heights, slot_colors, colored_counts, block_start)
    expected = _build_expected(heights, slot_colors, colored_counts, block_start)

    if get_bg(grid) != BG:
        return None
    if sorted(colors_present(grid, exclude=[BG, GREY])) != sorted(slot_colors):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _NEXT_BRANCH_INDEX

    rng = np.random.default_rng()
    forced_block_start = MATCH_BLOCK_STARTS[_NEXT_BRANCH_INDEX % len(MATCH_BLOCK_STARTS)]
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng, forced_block_start)
        if grid is not None:
            # Alternate the two color-match offsets so small train sets expose both branches.
            _NEXT_BRANCH_INDEX += 1
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_ATTEMPTS} attempts")
