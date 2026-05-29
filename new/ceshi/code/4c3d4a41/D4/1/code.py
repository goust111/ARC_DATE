# Variant of 4c3d4a41
# Dimension: D4 — 变换：彩色像素的重排规则 (transform)
# Change: 彩色像素在杆进入后堆叠在对应凸起顶部上方，从框架内顶部开始向下排列，数量受可用空间限制... → 彩色像素填充凸起之间的空隙

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg as _bg_color
from arc_utils import colors_present, get_bg, paste

# Design analysis for the D4 variant
# a) Concrete behavior:
#    The frame, bar, alignment rule, and four source color columns stay the same.
#    Only the post-alignment color placement changes. After the bar is inserted,
#    the relevant empty cells are the three bounded cavities between neighboring
#    projections, not the vertical space above each projection. For a pair of
#    adjacent source columns i and i+1, concatenate their colored cells
#    (left column first, then right column) and place that sequence into the
#    cavity between the aligned projections i and i+1, scanning top->bottom.
#    Small example: if source counts are [2,1] with colors [A,B] and the cavity
#    has three cells, the gap becomes [A,A,B] from top to bottom; the original
#    task would instead have produced a stack above one projection only.
# b) Complexity calibration:
#    The geometry is still fixed and visually clean, but the destination is now
#    three pairwise cavities rather than four independent columns. This is a
#    visible rule change because output colors move to even columns 12/14/16
#    instead of staying on 11/13/15/17.
# c) Old assumptions that break:
#    Any logic that iterates per destination column and searches "bar_top" above
#    that same column is tied to the old stacking rule. In the variant, colors
#    must be redistributed by adjacent projection pairs, so the solver now
#    computes projection tops first, then cavity cells between neighboring tops.
# d) Placement legality:
#    The bar translation is unchanged, so the frame/bar fit remains legal.
#    The generator samples projection heights from {3,4} and derives source
#    counts so each source column fits within its slot length and each cavity
#    has enough room for the intended pairwise fill.
# e) Visual observability:
#    The new rule is visible in 2-3 examples because colors end up inside the
#    inter-projection gaps, with middle source colors appearing in two adjacent
#    cavities. No color-id-based ordering is used; only left-to-right position
#    and top-to-bottom source order matter.
# f) Visual cleanliness:
#    Output overlays only two element types inside the slot: the grey bar and
#    colored cavity fills. Because the cavities are disjoint and fully inside
#    the frame interior, the result stays easy to parse.
# g) Color inferability:
#    Each slot column remains monochrome, so each cavity clearly inherits its
#    colors from its two neighboring columns. Distinct colors across the four
#    source columns make this correspondence legible.
# h) Non-degeneracy target:
#    The generator measures cavity capacities from the chosen heights and uses
#    them to derive source counts, ensuring the new gap-fill mechanic actually
#    controls the output instead of accidentally collapsing back to the old
#    "same-column stacking" behavior.
# i) New solver assumptions checked against generation:
#    - exactly one large grey frame component and one smaller grey bar component
#    - four aligned projection columns after translation
#    - source colored columns sorted left->right correspond to those projections
#    - each cavity is the cells strictly between adjacent projections and above
#      the base row. The generator preserves all of these.


def _bbox(pts):
    return bbox_coords(pts if isinstance(pts, np.ndarray) else np.asarray(list(pts), dtype=int))


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _most_common(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide a grey toothed bar into a grey frame, then fill the cavities between neighboring teeth with color.

    Input structure:
    - The background surrounds two grey 4-connected structures: a large frame on the right and a smaller bar on the left.
    - Inside the frame, four monochrome colored columns mark the destination alignment columns for the bar projections.

    Rule:
    - Move the bar into the frame so its base sits one row above the frame floor and its projections align with the colored guide columns.
    - Clear the original colored guide columns, then treat each adjacent pair of source columns as a color sequence for the cavity between the matching neighboring projections.
    - Fill each cavity top-to-bottom using the left source column first and the right source column second, clipping if the cavity is smaller than the combined sequence.

    Key insight:
    - After alignment, the meaningful empty regions are the bounded gaps between neighboring projections, so the colored columns become material for those cavities instead of stacks above single projections.
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

    bar_base_row = max(int(r) for r, _ in bar_pts)
    slot_interior_bottom = sr1 - 1
    target_base_row = slot_interior_bottom - 1
    dy = target_base_row - bar_base_row

    col_min_row = {}
    for r, c in bar_pts:
        r = int(r)
        c = int(c)
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
        rr = int(r) + dy
        cc = int(c) + dx
        if sr0 <= rr <= sr1 and sc0 <= cc <= sc1:
            out[rr, cc] = grey
            placed.append((rr, cc))

    placed_by_col = {}
    for rr, cc in placed:
        placed_by_col.setdefault(cc, []).append(rr)

    projection_cols = []
    for col, rows in placed_by_col.items():
        if min(rows) < target_base_row:
            projection_cols.append(col)
    projection_cols.sort()

    source_sequences = []
    for col in dst_cols:
        rows = [r for r in range(sr0, sr1 + 1) if g[r, col] != bg and g[r, col] != grey]
        source_sequences.append([int(g[r, col]) for r in rows])

    pair_count = min(len(projection_cols) - 1, len(source_sequences) - 1)
    for i in range(max(pair_count, 0)):
        left_proj = projection_cols[i]
        right_proj = projection_cols[i + 1]
        left_top = min(placed_by_col[left_proj])
        right_top = min(placed_by_col[right_proj])
        gap_top = max(left_top, right_top)
        fill_cells = [
            (rr, cc)
            for rr in range(gap_top, target_base_row)
            for cc in range(left_proj + 1, right_proj)
            if sr0 < rr < sr1 and sc0 < cc < sc1
        ]
        if not fill_cells:
            continue

        seq = source_sequences[i] + source_sequences[i + 1]
        if not seq:
            continue
        seq = seq[: len(fill_cells)]
        for (rr, cc), colv in zip(fill_cells, seq):
            out[rr, cc] = int(colv)

    return out


# --------------- generator ---------------

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

VALID_HEIGHTS = [
    (3, 3, 3, 3),
    (3, 3, 3, 4),
    (3, 3, 4, 3),
    (3, 3, 4, 4),
    (3, 4, 3, 3),
    (3, 4, 3, 4),
    (3, 4, 4, 4),
    (4, 3, 3, 3),
    (4, 3, 3, 4),
    (4, 3, 4, 3),
    (4, 3, 4, 4),
    (4, 4, 3, 3),
    (4, 4, 3, 4),
    (4, 4, 4, 3),
    (4, 4, 4, 4),
]
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


def _paint_slot_caps(grid: np.ndarray, col: int, length: int, count: int) -> None:
    cap = length - count
    grid[SLOT_TOP_ROW : SLOT_TOP_ROW + cap, col] = GREY


def _gap_capacities(heights: list[int]) -> list[int]:
    return [max(0, min(heights[i], heights[i + 1]) - 1) for i in range(len(heights) - 1)]


def _solve_counts_from_heights(heights: list[int], rng: np.random.Generator) -> list[int] | None:
    caps = _gap_capacities(heights)
    sols = []
    for n1 in (1, 2):
        n2 = caps[0] - n1
        n3 = caps[1] - n2
        n4 = caps[2] - n3
        counts = [n1, n2, n3, n4]
        if all(1 <= n <= 2 for n in counts):
            sols.append(counts)
    if not sols:
        return None
    return [int(v) for v in sols[int(rng.integers(0, len(sols)))]]


def _build_input_grid(heights: list[int], colors: list[int], counts: list[int], bar_c0: int) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    paste(grid, _bar_patch(heights), BAR_R0, bar_c0, transparent=BG)
    for col, length, color, count in zip(SLOT_COLS, SLOT_LENS, colors, counts):
        _paint_input_slot_column(grid, col, length, color, count)
    return grid


def _build_expected_grid(heights: list[int], colors: list[int], counts: list[int]) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    for col, length, count in zip(SLOT_COLS, SLOT_LENS, counts):
        _paint_slot_caps(grid, col, length, count)
    paste(grid, _bar_patch(heights), BAR_R0, BAR_OUTPUT_C0, transparent=BG)

    proj_cols = [BAR_OUTPUT_C0 + c for c in BAR_LOCAL_PROJ_COLS]
    tops = [BAR_BASE_ABS_ROW - h + 1 for h in heights]
    for i in range(len(proj_cols) - 1):
        left_proj = proj_cols[i]
        right_proj = proj_cols[i + 1]
        gap_top = max(tops[i], tops[i + 1])
        fill_cells = [
            (rr, cc)
            for rr in range(gap_top, BAR_BASE_ABS_ROW)
            for cc in range(left_proj + 1, right_proj)
        ]
        seq = [colors[i]] * counts[i] + [colors[i + 1]] * counts[i + 1]
        assert len(seq) == len(fill_cells)
        for (rr, cc), colv in zip(fill_cells, seq):
            grid[rr, cc] = int(colv)
    return grid


def _sample_params(rng: np.random.Generator) -> tuple[list[int], list[int], list[int]] | None:
    heights = [int(v) for v in VALID_HEIGHTS[int(rng.integers(0, len(VALID_HEIGHTS)))]]
    counts = _solve_counts_from_heights(heights, rng)
    if counts is None:
        return None
    colors = [int(v) for v in rng.choice(PALETTE, size=len(SLOT_COLS), replace=False)]
    return heights, colors, counts


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    params = _sample_params(rng)
    if params is None:
        return None
    heights, colors, counts = params
    grid = _build_input_grid(heights, colors, counts, BAR_INPUT_C0)
    expected = _build_expected_grid(heights, colors, counts)

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
