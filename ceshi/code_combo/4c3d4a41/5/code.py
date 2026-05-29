# Combo variant of 4c3d4a41 (index 5)
# combo_spec: D1/1+D10/0
# Name: 高度匹配对齐+高度由列属性决定

import numpy as np

from arc_utils import bbox_coords, colors_present, components_mask, get_bg as _bg_color, paste

# ---------------------------------------------------------------------------
# Design analysis for the combo variant
# ---------------------------------------------------------------------------
#
# a) Concrete behavior trace
#    - Original task:
#      The solver detected which slot columns already contained color, shifted
#      the detached bar horizontally until its four projections landed under
#      those same absolute columns, and then lifted each column's own colors
#      straight upward above the projection in that same x-position.
#    - This combo:
#      The bar no longer uses the colored columns as horizontal anchors. It is
#      always inserted into the frame's canonical centered position. Matching is
#      between two visible numeric properties:
#        * projection height h, counted in grey cells including the base row
#        * colored-count n in a slot column
#      A projection receives the color stack from the unique slot column whose
#      n == h.
#    - Example:
#        input slot counts, left->right: [2, 4, 1, 3]
#        input slot colors, left->right: [A, B, C, D]
#        derived bar heights, left->right: [3, 1, 4, 2]
#      Original behavior would keep [A, B, C, D] under the same columns.
#      Combo behavior matches by equality:
#        height 3 <- column D
#        height 1 <- column C
#        height 4 <- column B
#        height 2 <- column A
#      so the output color order becomes [D, C, B, A].
#
# b) Complexity calibration
#    - The input keeps the original fixed geometry, so the puzzle still has one
#      frame, one detached bar, and four countable colored columns.
#    - To make equality matching unique and easy to read, the generator uses
#      the four distinct counts/heights {1,2,3,4}. This avoids ambiguous
#      many-to-one matches and gives each sample a full permutation signal.
#    - To prevent collapse back to the original positional rule, the generator
#      rejects the identity permutation between source column counts and target
#      projection heights.
#
# c) Implicit-assumption review against the original solver
#    - Original assumption: destination x-positions are inferred from the input
#      colored columns. Broken by D1/1, so the combo solver instead computes the
#      canonical centered placement from frame width and bar width.
#    - Original assumption: colors stay in their source columns. Broken by the
#      equality match, so the combo solver must extract per-column color stacks
#      first, then repaint them at matched projection columns.
#    - Original assumption: bar heights are independent random clutter. Broken
#      by D10/0, so the generator must build heights from the sampled slot
#      counts rather than sampling them independently.
#    - New solver assumption: each projection height finds exactly one column of
#      the same colored-count. The generator enforces counts/heights as a shared
#      permutation of {1,2,3,4}, so that assumption always holds.
#
# d) Placement legality
#    - Slot columns remain at fixed absolute cols 11,13,15,17 with legal lengths
#      [3,4,4,3]. The only count 4 is always assigned to one of the two middle
#      columns, the only positions that can physically host four colored cells.
#    - The bar keeps the original 7-wide base at rows 2..5 in the input and is
#      translated to cols 11..17 in the output, which fits entirely inside the
#      known frame interior without overlap beyond intended overwriting.
#
# e) Visual observability
#    - Humans can count both signatures directly from the input: projection
#      heights and colored run lengths.
#    - Because the generator rejects identity matches, the horizontal order of
#      colors in the output differs from the input, making the size-match cue
#      visible over just a few examples.
#
# f) Visual cleanliness
#    - The solver clears the slot interior before rebuilding it. That removes
#      stale grey caps from source columns, so the output only shows frame, bar,
#      and matched color stacks.
#    - Each output column contains at most one solid color above one grey
#      projection, preserving the same clean layering as the original task.
#
# g) Inferability of the change
#    - No hidden use of color ids, ordering by palette value, or arbitrary
#      tie-breakers is needed. The only mapping signal is geometric count
#      equality, and the generator uses four distinct counts so the permutation
#      can be inferred structurally from 2-3 examples.


def _bbox(pts):
    return bbox_coords(pts if isinstance(pts, np.ndarray) else np.asarray(list(pts), dtype=int))


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _most_common(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _projection_data(
    bar_pts: np.ndarray, bar_bbox: tuple[int, int, int, int]
) -> tuple[int, list[tuple[int, int]]]:
    """Return bar base row and ordered (source_col, height) pairs."""
    base_row = max(int(r) for r, _ in bar_pts)
    col_min_row: dict[int, int] = {}
    for r, c in bar_pts:
        rr = int(r)
        cc = int(c)
        col_min_row[cc] = min(col_min_row.get(cc, rr), rr)

    _br0, bc0, _br1, bc1 = bar_bbox
    data = []
    for c in range(int(bc0), int(bc1) + 1, 2):
        top = col_min_row[int(c)]
        data.append((int(c), int(base_row - top + 1)))
    return int(base_row), data


def _slot_columns(
    g: np.ndarray, bg: int, grey: int, slot_bbox: tuple[int, int, int, int]
) -> list[dict]:
    """Collect colored stacks inside the slot, ordered left-to-right."""
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
    Insert the grey bar into the frame and reorder the colored stacks by matching heights to counts.

    Input structure:
    - The background and a single grey structural color form two separated objects: a large right-hand half-open frame and a detached horizontal bar with four upward projections.
    - Inside the frame, four vertical slot columns contain contiguous colored runs; the number of colored cells in each run is the column's visible size signature.

    Rule:
    - Move the bar into its standard centered location inside the frame so the base sits one row above the interior floor.
    - Measure every projection height, count the colored cells in every input slot column, and pair columns to projections only when the two counts are exactly equal.
    - Repaint each matched color stack immediately above its projection, truncating only if the projection leaves less vertical room than the input stack length.

    Key insight:
    - Horizontal position in the input is no longer the correspondence signal. The decisive relation is that each projection's grey height was derived from one slot column's colored count, so solving the task means recovering that one-to-one equality match.
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
    _br0, bc0, _br1, bc1 = bar_bbox

    out = np.full_like(g, bg)
    out[sr0 : sr1 + 1, sc0 : sc1 + 1] = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
    if sr1 - sr0 >= 2 and sc1 - sc0 >= 2:
        out[sr0 + 1 : sr1, sc0 + 1 : sc1] = bg

    bar_base_row, proj_data = _projection_data(bar_pts, bar_bbox)
    target_base_row = (sr1 - 1) - 1
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

    placed_by_col: dict[int, list[int]] = {}
    for rr, cc in placed:
        placed_by_col.setdefault(int(cc), []).append(int(rr))

    slot_cols = _slot_columns(g, bg, grey, slot_bbox)
    by_count = {int(info["count"]): info for info in slot_cols}

    interior_top = sr0 + 1
    interior_bottom = sr1 - 1

    for src_col, height in proj_data:
        out_col = int(src_col + dx)
        info = by_count.get(int(height))
        if info is None or out_col not in placed_by_col:
            continue

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
        color = count_to_color[int(height)]
        _paint_output_stack(grid, int(out_col), int(color), int(height), int(height))
    return grid


def _sample_counts(rng: np.random.Generator) -> list[int]:
    """Sample a legal placement of the unique counts {1,2,3,4} onto [3,4,4,3]."""
    mid_with_four = int(rng.integers(0, 2))
    middle_counts = [0, 0]
    middle_counts[mid_with_four] = 4

    remaining = [1, 2, 3]
    rng.shuffle(remaining)
    middle_counts[1 - mid_with_four] = remaining.pop()

    outer_counts = remaining
    rng.shuffle(outer_counts)
    return [int(outer_counts[0]), int(middle_counts[0]), int(middle_counts[1]), int(outer_counts[1])]


def _derive_heights_from_counts(rng: np.random.Generator, counts: list[int]) -> list[int]:
    """Heights are a non-identity permutation of the sampled colored-counts."""
    heights = [int(v) for v in counts]
    while True:
        rng.shuffle(heights)
        if any(h != c for h, c in zip(heights, counts)):
            return heights.copy()


def _sample_params(rng: np.random.Generator) -> tuple[list[int], list[int], list[int]]:
    colors = [int(v) for v in rng.choice(PALETTE, size=len(SLOT_COLS), replace=False)]
    counts = _sample_counts(rng)
    heights = _derive_heights_from_counts(rng, counts)
    return heights, colors, counts


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    heights, colors, counts = _sample_params(rng)
    grid = _build_input(heights, colors, counts)
    expected = _build_expected(heights, colors, counts)

    if _bg_color(grid) != BG:
        return None
    if sorted(colors_present(grid, exclude=[BG, GREY])) != sorted(colors):
        return None
    if heights == counts:
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
