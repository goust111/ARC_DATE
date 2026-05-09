# Variant of 4c3d4a41
# Dimension: D18 — 彩色像素堆叠位置计算 (transform)
# Change: 彩色像素紧贴凸起顶部下方堆叠，受框架顶部内壁限制，多余像素被截断（保留底部的）... → 彩色像素从框架顶部开始向下堆叠

from pathlib import Path
import sys

import numpy as np

SOLVER_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_UTILS) not in sys.path:
    sys.path.append(str(SOLVER_UTILS))

from arc_utils import bbox_coords, colors_present, components_mask, get_bg, get_bg as _bg_color, paste


# Design analysis for the D18 variant
# a) Concretization and behavior trace:
#    - The input keeps the original geometry: an 8x20 canvas, a grey frame on the right, and
#      a separate grey bar on the left with four upward projections. Inside the frame, the four
#      odd columns (11, 13, 15, 17) still contain a grey cap followed by a monochrome colored run.
#    - Only the transform stage changes. Example: if a destination column has one colored cell
#      and the aligned bar projection leaves four free rows above it, the original task placed
#      that color at the bottom of the free space; this variant places it at row 1 instead.
#      The colored block now grows downward from the frame ceiling, so a visible gap to the bar
#      can remain below it.
#    - The changed stage is the per-column repaint after the bar has been translated. Detecting
#      the frame, detecting the bar, and aligning projection columns are unchanged.
# b) Complexity calibration:
#    - The original four target columns, fixed frame, and varied projection heights are retained,
#      so the bar-to-column alignment problem stays unambiguous.
#    - To keep the new anchor visually observable, the generator rejects samples where every
#      column completely fills its available space; at least one column must satisfy count < 5-h.
# c) Implicit-assumption review:
#    - Choosing the largest two grey components still assumes exactly one frame component and one
#      bar component; the generator preserves that separation.
#    - Destination columns are still read from non-grey colors inside the frame, so each slot
#      column must contain at least one colored cell in the input; the generator enforces that.
#    - Horizontal alignment still assumes evenly spaced projection columns and colored columns;
#      both remain at odd columns with a constant offset of +10.
#    - The new repaint logic assumes the frame interior starts at sr0+1 and that the translated
#      bar contributes a lowest blocking row per destination column; the fixed geometry ensures it.
#    - The solver still preserves original grey cap cells because it only clears non-grey colors
#      from the slot before repainting. This remains valid in the variant and keeps D18 isolated
#      to the colored-stack anchor rather than changing the frame's other markings.
# d) Placement legality:
#    - The frame bbox is fixed at rows 0-7, cols 9-19. The translated bar lands at cols 11-17,
#      entirely inside the frame opening, with no overlap outside the intended slot.
#    - Output colors are clipped to the rows strictly above the aligned bar top, so they cannot
#      cross into the grey bar or leave the frame interior.
# e) Visual observability:
#    - Humans can see the difference from 2-3 examples because the same colored count can appear
#      either near the bar (old rule) or attached to the ceiling (new rule). The generator forces
#      at least one such ceiling-anchored gap in every sample.
# f) Visual cleanliness:
#    - Each active column still uses one color, and the only overlap is the intended overwrite of
#      preserved grey caps by the new top-anchored colors, so the output stays easy to parse.
# g) Color-change inferability:
#    - No color remapping is introduced. Colors are copied column-wise, and position alone explains
#      the transform, which matches ARC's small-example inference requirements.


def _bbox(pts):
    arr = pts if isinstance(pts, np.ndarray) else np.asarray(list(pts), dtype=int)
    return bbox_coords(arr)


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _most_common(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the grey bar into the grey frame, align the projections to the colored columns, then re-stack colors from the ceiling downward.

    Input structure:
    - The background surrounds two grey connected components: a large frame on the right and a smaller bar on the left with upward projections.
    - Inside the frame, four colored columns indicate which bar projection columns matter and how many colored cells each column carries.

    Rule:
    - Translate the bar into the frame so its base sits on the same row as before and its projections line up with the colored slot columns.
    - For each aligned colored column, move that column's colored cells to the top of the frame interior and stack them downward, clipping at the first bar cell below.

    Key insight: The bar still determines which columns interact, but in this variant the frame's interior ceiling, not the bar top, is the anchor for the colored stack.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

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
        rr = int(r)
        cc = int(c)
        if sr0 <= rr <= sr1 and sc0 <= cc <= sc1:
            dst_cols_set.add(cc)
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

    for col in dst_cols:
        if col not in placed_by_col:
            continue

        bar_top = min(placed_by_col[col])
        max_len = bar_top - interior_top
        if max_len <= 0:
            continue

        rows = [r for r in range(sr0, sr1 + 1) if g[r, col] != bg and g[r, col] != grey]
        if not rows:
            continue
        colors = [int(g[r, col]) for r in rows]
        visible = colors[:max_len]

        for i, colv in enumerate(visible):
            out[interior_top + i, col] = colv

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


def _paint_input_slot_column(grid: np.ndarray, col: int, length: int, color: int, count: int) -> None:
    cap = length - count
    grid[SLOT_TOP_ROW : SLOT_TOP_ROW + cap, col] = GREY
    grid[SLOT_TOP_ROW + cap : SLOT_TOP_ROW + length, col] = color


def _paint_output_slot_column(grid: np.ndarray, col: int, length: int, color: int, count: int, height: int) -> None:
    cap = length - count
    grid[SLOT_TOP_ROW : SLOT_TOP_ROW + cap, col] = GREY
    visible = min(count, BAR_BASE_ABS_ROW - height)
    if visible > 0:
        grid[SLOT_TOP_ROW : SLOT_TOP_ROW + visible, col] = color


def _build_input_grid(heights: list[int], colors: list[int], colored_counts: list[int]) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    paste(grid, _bar_patch(heights), BAR_R0, BAR_INPUT_C0)
    for col, length, color, count in zip(SLOT_COLS, SLOT_LENS, colors, colored_counts):
        _paint_input_slot_column(grid, col, length, color, count)
    return grid


def _build_output_grid(heights: list[int], colors: list[int], colored_counts: list[int]) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    paste(grid, _bar_patch(heights), BAR_R0, BAR_OUTPUT_C0)
    for col, length, color, count, height in zip(SLOT_COLS, SLOT_LENS, colors, colored_counts, heights):
        _paint_output_slot_column(grid, col, length, color, count, height)
    return grid


def _sample_params(rng: np.random.Generator) -> tuple[list[int], list[int], list[int]]:
    colors = [int(v) for v in rng.choice(PALETTE, size=len(SLOT_COLS), replace=False)]
    colored_counts = [int(rng.integers(1, length + 1)) for length in SLOT_LENS]
    heights = [int(rng.integers(*HEIGHT0_RANGE))]
    heights.extend(int(rng.integers(*HEIGHT_REST_RANGE)) for _ in range(len(SLOT_COLS) - 1))
    return heights, colors, colored_counts


def _variant_observable(heights: list[int], colored_counts: list[int]) -> bool:
    return any(count < (BAR_BASE_ABS_ROW - height) for count, height in zip(colored_counts, heights))


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    heights, colors, colored_counts = _sample_params(rng)
    if not _variant_observable(heights, colored_counts):
        return None

    grid = _build_input_grid(heights, colors, colored_counts)
    expected = _build_output_grid(heights, colors, colored_counts)

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
