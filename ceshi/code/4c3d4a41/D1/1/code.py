# Variant of 4c3d4a41
# Dimension: D1 — 匹配条件：凸起-列对齐方式 (match_cond)
# Change: 杆的凸起列与框架内彩色列按绝对列位置一一对齐(projection_slot_alignment + column_alignment) → 凸起高度与彩色列长度匹配

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg as _bg_color
from arc_utils import colors_present, get_bg, paste

# ---------------------------------------------------------------------------
# Design analysis for the D1 variant
# ---------------------------------------------------------------------------
#
# variant_spec:
#   kept_geometry:
#     grid: 8x20 fixed
#     frame: same right-side half-open grey frame, bbox rows 0..7 cols 9..19
#     bar: same detached 7-wide base with 4 upward projections, moved into the
#       slot with the same final placement as the original task
#     slot_columns: same 4 fixed slot columns at absolute cols 11,13,15,17
#   core_change:
#     old_match:
#       The original solver identified the destination columns directly from the
#       colored columns already present in the slot, then horizontally aligned
#       the bar so each projection landed under the column at the same x-position.
#     new_match:
#       The bar always goes to its standard final position inside the frame.
#       The matching step is now between:
#         - projection height h (counted as grey cells in that projection,
#           including the base row), and
#         - colored-count n in an input slot column.
#       A projection and a color column pair iff h == n.
#       The output therefore permutes the colors across the 4 slot columns.
#   concrete_trace:
#     example_input:
#       bar_heights_left_to_right: [1, 4, 2, 3]
#       colored_counts_left_to_right: [2, 3, 4, 1]
#       colors_left_to_right: [A, B, C, D]
#     old_rule_effect:
#       The colors would stay in the same columns [A, B, C, D]; only vertical
#       restacking would happen.
#     new_rule_effect:
#       height 1 takes the column with 1 colored cell -> color D
#       height 4 takes the column with 4 colored cells -> color C
#       height 2 takes the column with 2 colored cells -> color A
#       height 3 takes the column with 3 colored cells -> color B
#       output color order becomes [D, C, A, B].
#   complexity_calibration:
#     - Exact equality is the only matching condition; there is no color-id
#       sorting, no hidden arithmetic, and no multiple transforms.
#     - To keep the pairing unambiguous, the generator uses the four distinct
#       heights/counts {1,2,3,4}. This yields a one-to-one permutation.
#     - The generator also forbids the identity permutation, so samples do not
#       collapse back to the original column-alignment behavior.
#   implicit_assumptions_review:
#     - Original assumption: the set of colored columns determines where the bar
#       should be shifted. This is removed; the bar is placed at its canonical
#       centered location inside the frame instead.
#     - Original assumption: colors remain in their source columns. This is no
#       longer valid; colors are reassigned by count==height.
#     - Original output copied interior grey caps from the input. That becomes
#       misleading once colors move sideways, so the solver clears the slot
#       interior and rebuilds only frame + bar + reassigned colors.
#   placement_legality:
#     - Input still uses the fixed original frame and detached bar, so there is
#       no risk of overlap outside the known layout.
#     - Counts are sampled as a permutation of {1,2,3,4}; the only count 4 is
#       placed in one of the two middle slot columns, the only columns with
#       length 4, preventing impossible input columns.
#   visual_observability:
#     - Human-visible cue in the input: each slot column has a clearly countable
#       colored run, and each bar projection has a clearly countable height.
#     - Human-visible cue in the output: colors appear above projections in a
#       different horizontal order from the input, and that order matches the
#       height-count equality rather than the original positions.
#   cleanliness:
#     - Clearing the slot interior avoids leftover grey caps from the source
#       columns, which would otherwise obscure the new sideways permutation.


def _bbox(pts):
    return bbox_coords(pts if isinstance(pts, np.ndarray) else np.asarray(list(pts), dtype=int))


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _most_common(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _projection_data(bar_pts: np.ndarray, bar_bbox: tuple[int, int, int, int]) -> tuple[int, list[tuple[int, int]]]:
    """Return bar base row and [(src_col, height)] for the 4 designated projection columns."""
    base_row = max(int(r) for r, _ in bar_pts)
    col_min_row = {}
    for r, c in bar_pts:
        r = int(r)
        c = int(c)
        col_min_row[c] = min(col_min_row.get(c, r), r)
    _br0, bc0, _br1, bc1 = bar_bbox
    data = []
    for c in range(int(bc0), int(bc1) + 1, 2):
        top = col_min_row[int(c)]
        data.append((int(c), int(base_row - top + 1)))
    return int(base_row), data


def _slot_columns(g: np.ndarray, bg: int, grey: int, slot_bbox: tuple[int, int, int, int]) -> list[dict]:
    """Collect per-column colored data inside the slot, ordered left-to-right."""
    sr0, sc0, sr1, sc1 = slot_bbox
    cols = []
    for c in range(sc0 + 1, sc1):
        rows = [r for r in range(sr0 + 1, sr1) if int(g[r, c]) != bg and int(g[r, c]) != grey]
        if not rows:
            continue
        cols.append(
            {
                "col": int(c),
                "count": int(len(rows)),
                "colors": [int(g[r, c]) for r in rows],
            }
        )
    return cols


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the grey bar into the frame, then match colored columns to projections by equal size.

    Input structure:
    - bg and a grey structural color form two main pieces: a large right-side open frame and a detached horizontal bar with four upward projections.
    - Inside the frame, four colored columns sit in fixed slot positions; each column may have grey cells above it, and the number of colored cells is the column's numeric signature.

    Rule:
    - Move the bar into its standard location inside the frame.
    - Measure each projection's height, count the colored pixels in each input slot column, and pair them by exact equality.
    - Repaint each matched color stack directly above the corresponding projection; if equal sizes repeat, resolve them left-to-right within that size group.

    Key insight: The decisive cue is not x-alignment anymore; it is the equality between projection height and colored-column size, so the output can reorder the colors horizontally.
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
    sr0, sc0, sr1, sc1 = slot_bbox
    br0, bc0, br1, bc1 = bar_bbox

    out = np.full_like(g, bg)
    out[sr0 : sr1 + 1, sc0 : sc1 + 1] = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
    if sr1 - sr0 >= 2 and sc1 - sc0 >= 2:
        out[sr0 + 1 : sr1, sc0 + 1 : sc1] = bg

    bar_base_row, proj_data = _projection_data(bar_pts, bar_bbox)
    slot_interior_bottom = sr1 - 1
    target_base_row = slot_interior_bottom - 1
    dy = target_base_row - bar_base_row

    bar_w = bc1 - bc0 + 1
    slot_w = sc1 - sc0 + 1
    target_bar_left = sc0 + (slot_w - bar_w) // 2
    dx = target_bar_left - bc0

    placed = []
    for r, c in bar_pts:
        rr = int(r + dy)
        cc = int(c + dx)
        if sr0 <= rr <= sr1 and sc0 <= cc <= sc1:
            out[rr, cc] = grey
            placed.append((rr, cc))

    placed_by_col = {}
    for rr, cc in placed:
        placed_by_col.setdefault(int(cc), []).append(int(rr))

    slot_cols = _slot_columns(g, bg, grey, slot_bbox)
    by_count: dict[int, list[dict]] = {}
    for info in slot_cols:
        by_count.setdefault(int(info["count"]), []).append(info)

    interior_top = sr0 + 1
    interior_bottom = sr1 - 1

    for src_col, height in proj_data:
        out_col = int(src_col + dx)
        matches = by_count.get(int(height), [])
        if not matches or out_col not in placed_by_col:
            continue
        info = matches.pop(0)
        bar_top = min(placed_by_col[out_col])
        desired_bottom = bar_top - 1
        if desired_bottom < interior_top:
            continue

        colors = list(info["colors"])
        max_len = desired_bottom - interior_top + 1
        if max_len <= 0:
            continue
        if len(colors) > max_len:
            colors = colors[-max_len:]

        start_row = desired_bottom - len(colors) + 1
        for i, colv in enumerate(colors):
            rr = int(start_row + i)
            if interior_top <= rr <= interior_bottom:
                out[rr, out_col] = int(colv)

    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

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

HEIGHT_VALUES = [1, 2, 3, 4]
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


def _paint_output_stack(grid: np.ndarray, col: int, color: int, count: int, height: int) -> None:
    bar_top = BAR_BASE_ABS_ROW - height + 1
    desired_bottom = bar_top - 1
    if desired_bottom < SLOT_TOP_ROW:
        return
    visible = min(count, desired_bottom - SLOT_TOP_ROW + 1)
    if visible <= 0:
        return
    start_row = desired_bottom - visible + 1
    grid[start_row : desired_bottom + 1, col] = color


def _build_input(heights: list[int], colors: list[int], counts: list[int]) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    paste(grid, _bar_patch(heights), BAR_R0, BAR_INPUT_C0)
    for col, length, color, count in zip(SLOT_COLS, SLOT_LENS, colors, counts):
        _paint_input_slot_column(grid, col, length, color, count)
    return grid


def _build_expected(heights: list[int], colors: list[int], counts: list[int]) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    paste(grid, _bar_patch(heights), BAR_R0, BAR_OUTPUT_C0)

    count_to_color = {int(count): int(color) for count, color in zip(counts, colors)}
    out_cols = [BAR_OUTPUT_C0 + local_c for local_c in BAR_LOCAL_PROJ_COLS]
    for out_col, height in zip(out_cols, heights):
        _paint_output_stack(grid, out_col, count_to_color[int(height)], int(height), int(height))
    return grid


def _sample_counts(rng: np.random.Generator) -> list[int]:
    mid_with_four = int(rng.integers(0, 2))
    middle_counts = [0, 0]
    middle_counts[mid_with_four] = 4
    remaining = [1, 2, 3]
    rng.shuffle(remaining)
    middle_counts[1 - mid_with_four] = remaining.pop()
    outer_counts = remaining
    rng.shuffle(outer_counts)
    return [int(outer_counts[0]), int(middle_counts[0]), int(middle_counts[1]), int(outer_counts[1])]


def _sample_params(rng: np.random.Generator) -> tuple[list[int], list[int], list[int]]:
    colors = [int(v) for v in rng.choice(PALETTE, size=len(SLOT_COLS), replace=False)]
    heights = [int(v) for v in rng.permutation(np.array(HEIGHT_VALUES, dtype=int))]
    counts = _sample_counts(rng)
    return heights, colors, counts


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    heights, colors, counts = _sample_params(rng)
    if heights == counts:
        return None

    grid = _build_input(heights, colors, counts)
    expected = _build_expected(heights, colors, counts)

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
