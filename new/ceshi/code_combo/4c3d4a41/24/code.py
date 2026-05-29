# Combo variant of 4c3d4a41 (index 24)
# combo_spec: D4/3+D18/1
# Name: 高度决定数量+顶部截断

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import bbox_coords, colors_present, components_mask, get_bg, get_bg as _bg_color, paste


# Design analysis for the combo variant
# a) Concrete behavior:
#    - The structural layout is unchanged: a grey frame sits on the right, a grey bar with four projections sits on the left,
#      and four slot columns inside the frame mark the destination columns.
#    - The fused change only affects how colors above the moved bar are rebuilt.
#      Example: if one input slot column contains top->bottom colors [2, 8, 4] and the aligned projection height is 2,
#      the original solver family would have taken a size based on the available stack region and, when forced to clip,
#      kept the lower end [8, 4]. The fused rule instead keeps exactly `height` cells and takes the upper end [2, 8].
#    - So the output column count comes from geometry, while the surviving color order comes from the input column's top segment.
# b) Complexity calibration:
#    - Heights are restricted to {1, 2}. In the fixed 8x20 geometry, taller projections would leave fewer than `height`
#      cells above the bar, making "output count = height" physically impossible.
#    - Each generated example contains both heights, at least three columns with input_count != height, and at least two columns
#      where top-truncation and bottom-truncation would produce different visible outputs.
#    - At least one column also keeps `input_count == height`, so the no-truncation branch remains observable instead of
#      degenerating into a pure "always clip" dataset.
#    - Multi-color slot columns are short contiguous stacks, so the new truncation direction is visible without adding new objects.
# c) Hidden-assumption audit on the original solver:
#    - Original code assumed the output colored count was tied to the number of colored input cells. That is broken by D4/3.
#      The new solver reads the input only for color order and gets the count from projection height.
#    - Original clipping used `colors[-max_len:]`, which silently preferred bottom pixels. D18/1 inverts that to top retention,
#      so the new solver uses `colors[:height]`.
#    - Original generator used monochrome columns, which made top-vs-bottom clipping observationally identical. The new generator
#      deliberately samples multi-color stacks so the inverted truncation direction actually changes the output.
# d) Placement legality:
#    - The frame, bar width, and translation remain the original fixed geometry, so the moved bar still lands fully inside the slot.
#    - With heights in {1, 2}, every destination column has at least 3 cells of clearance above the bar, enough to place exactly
#      `height` colored cells at the top and grey fill below them.
#    - Input colored stack lengths are always sampled in [height, slot_length], so the solver never needs to invent missing colors.
# e) Visual observability:
#    - A human can compare each input column with the output shaft above the aligned projection and see two simultaneous cues:
#      shorter/taller projections decide how many cells survive, and the survivors are taken from the top of the input stack.
#    - Requiring at least two columns with `top_h != bottom_h` ensures the output would visibly differ from bottom-retention.
# f) Visual cleanliness:
#    - Each slot column remains a single vertical lane with no overlaps; output shafts are rebuilt as top color cells followed by grey.
#    - The empty spacer columns inside the frame are preserved, so the four active columns remain easy to track.
# g) Color-rule audit:
#    - No logic depends on numeric color ids. Colors are copied only by vertical order within each slot column.
#    - Repeated colors across different columns are allowed, but truncation is always explained by position, not by palette value.
# 9) New solver assumptions checked against the generator:
#    - Largest grey component is the frame and second largest is the bar: guaranteed by the fixed layout.
#    - Exactly four destination columns exist inside the frame and line up with the four projection columns: guaranteed by construction.
#    - Every destination column has at least `height` colored cells available: guaranteed by sampling `len(seq) >= height`.
#    - The fused behavior is non-degenerate: guaranteed by requiring visible top-vs-bottom differences and by rejecting samples whose
#      top-truncation output equals the legacy bottom-truncation output.


def _bbox(pts):
    return bbox_coords(pts if isinstance(pts, np.ndarray) else np.asarray(list(pts), dtype=int))


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _most_common(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _select_kept_colors(seq: list[int], height: int, *, keep_top: bool) -> list[int]:
    if not seq or height <= 0:
        return []
    if len(seq) >= height:
        return list(seq[:height] if keep_top else seq[-height:])
    # Fallback only for malformed inputs; generator guarantees len(seq) >= height.
    return list(seq) + [int(seq[-1])] * (height - len(seq))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide a grey bar into a grey frame, then rebuild each colored shaft from projection height and top-retained colors.

    Input structure:
    - The most common color is the background and the dominant non-background color is the grey structure.
    - Grey forms two disconnected components: a rectangular frame on the right and a horizontal bar with four upward projections on the left.
    - Inside the frame, four vertical slot columns contain contiguous colored stacks whose column positions indicate the target projection alignment.

    Rule:
    - Move the bar into the frame so its base rests one row above the frame interior bottom, and shift it horizontally until the projection columns align with the colored slot columns.
    - Clear the original colored stacks inside the frame, then rebuild each aligned shaft above the moved bar.
    - For each destination column, keep exactly as many colored cells as the aligned projection's height, place them flush against the top of the frame interior, and when truncation is needed keep the top part of the input color sequence rather than the bottom part.

    Key insight: The input slot columns carry an ordered color sequence, but geometry decides the final count; each projection height selects how many leading colors survive.
    """
    g = np.asarray(grid, dtype=int)
    _, w = g.shape

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

    out = np.full_like(g, bg)
    out[sr0 : sr1 + 1, sc0 : sc1 + 1] = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region = out[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region[(slot_region != bg) & (slot_region != grey)] = bg

    bar_base_row = max(int(r) for r, _ in bar_pts)
    target_base_row = (sr1 - 1) - 1
    dy = target_base_row - bar_base_row

    _br0, bc0, _br1, _bc1 = bar_bbox
    base_cols = sorted(int(c) for r, c in bar_pts if int(r) == bar_base_row)
    src_proj_cols = base_cols[::2] if len(base_cols) >= 4 else list(range(int(bc0), int(bc0) + 7, 2))

    col_min_row: dict[int, int] = {}
    for r, c in bar_pts:
        r = int(r)
        c = int(c)
        col_min_row[c] = min(col_min_row.get(c, r), r)
    src_heights = {
        int(c): int(bar_base_row - col_min_row.get(int(c), bar_base_row) + 1)
        for c in src_proj_cols
    }

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

    placed: list[tuple[int, int]] = []
    for r, c in bar_pts:
        rr = int(r) + dy
        cc = int(c) + dx
        if sr0 <= rr <= sr1 and sc0 <= cc <= sc1:
            out[rr, cc] = grey
            placed.append((rr, cc))

    placed_by_col: dict[int, list[int]] = {}
    for rr, cc in placed:
        placed_by_col.setdefault(int(cc), []).append(int(rr))

    dst_heights = {int(c + dx): int(h) for c, h in src_heights.items()}
    interior_top = sr0 + 1

    for col in dst_cols:
        if col not in placed_by_col or col not in dst_heights:
            continue

        seq = [int(g[r, col]) for r in range(sr0, sr1 + 1) if g[r, col] != bg and g[r, col] != grey]
        if not seq:
            continue

        bar_top = min(placed_by_col[col])
        desired_bottom = bar_top - 1
        if desired_bottom < interior_top:
            continue

        height = dst_heights[col]
        if interior_top + height - 1 > desired_bottom:
            continue

        kept = _select_kept_colors(seq, height, keep_top=True)
        out[interior_top : desired_bottom + 1, col] = grey
        for i, color in enumerate(kept):
            out[interior_top + i, col] = color

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

HEIGHT_VALUES = (1, 2)
MAX_ATTEMPTS = 400


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


def _paint_input_slot_column(grid: np.ndarray, col: int, length: int, seq: list[int]) -> None:
    cap = length - len(seq)
    grid[SLOT_TOP_ROW : SLOT_TOP_ROW + cap, col] = GREY
    for i, color in enumerate(seq):
        grid[SLOT_TOP_ROW + cap + i, col] = int(color)


def _paint_output_slot_column(grid: np.ndarray, col: int, seq: list[int], height: int, *, keep_top: bool) -> None:
    desired_bottom = BAR_BASE_ABS_ROW - height
    grid[SLOT_TOP_ROW : desired_bottom + 1, col] = GREY
    kept = _select_kept_colors(seq, height, keep_top=keep_top)
    for i, color in enumerate(kept):
        grid[SLOT_TOP_ROW + i, col] = int(color)


def _build_input(heights: list[int], sequences: list[list[int]]) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    paste(grid, _bar_patch(heights), BAR_R0, BAR_INPUT_C0)
    for col, length, seq in zip(SLOT_COLS, SLOT_LENS, sequences):
        _paint_input_slot_column(grid, col, length, seq)
    return grid


def _build_output(heights: list[int], sequences: list[list[int]], *, keep_top: bool) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    paste(grid, _bar_patch(heights), BAR_R0, BAR_OUTPUT_C0)
    for col, seq, height in zip(SLOT_COLS, sequences, heights):
        _paint_output_slot_column(grid, col, seq, height, keep_top=keep_top)
    return grid


def _visible_top_trunc_cols(heights: list[int], sequences: list[list[int]]) -> int:
    return sum(
        int(len(seq) > height and _select_kept_colors(seq, height, keep_top=True) != _select_kept_colors(seq, height, keep_top=False))
        for height, seq in zip(heights, sequences)
    )


def _variant_metrics(heights: list[int], sequences: list[list[int]]) -> dict[str, int]:
    different_count_cols = sum(int(len(seq) != height) for height, seq in zip(heights, sequences))
    reduction_cols = sum(int(len(seq) > height) for height, seq in zip(heights, sequences))
    visible_top_trunc_cols = _visible_top_trunc_cols(heights, sequences)
    exact_match_cols = sum(int(len(seq) == height) for height, seq in zip(heights, sequences))
    return {
        "different_count_cols": different_count_cols,
        "reduction_cols": reduction_cols,
        "visible_top_trunc_cols": visible_top_trunc_cols,
        "exact_match_cols": exact_match_cols,
    }


def _sample_sequences(rng: np.random.Generator, heights: list[int]) -> list[list[int]]:
    sequences: list[list[int]] = []
    for length, height in zip(SLOT_LENS, heights):
        count = int(rng.integers(height, length + 1))
        seq = [int(v) for v in rng.choice(PALETTE, size=count, replace=False)]
        sequences.append(seq)
    return sequences


def _sample_params(rng: np.random.Generator) -> tuple[list[int], list[list[int]]]:
    while True:
        heights = [int(rng.choice(HEIGHT_VALUES)) for _ in SLOT_COLS]
        if set(heights) != set(HEIGHT_VALUES):
            continue

        sequences = _sample_sequences(rng, heights)
        metrics = _variant_metrics(heights, sequences)
        expected = _build_output(heights, sequences, keep_top=True)
        legacy = _build_output(heights, sequences, keep_top=False)

        if metrics["different_count_cols"] < 3:
            continue
        if metrics["reduction_cols"] < 2:
            continue
        if metrics["exact_match_cols"] < 1:
            continue
        if metrics["visible_top_trunc_cols"] < 2:
            continue
        if np.array_equal(expected, legacy):
            continue
        return heights, sequences


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    heights, sequences = _sample_params(rng)
    grid = _build_input(heights, sequences)
    expected = _build_output(heights, sequences, keep_top=True)

    visible_colors = sorted({color for seq in sequences for color in seq})
    if get_bg(grid) != BG:
        return None
    if sorted(colors_present(grid, exclude=[BG, GREY])) != visible_colors:
        return None
    if np.array_equal(expected, _build_output(heights, sequences, keep_top=False)):
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
