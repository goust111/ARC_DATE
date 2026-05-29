# Variant of 4c3d4a41
# Dimension: D18 — 彩色像素堆叠位置计算 (transform)
# Change: 彩色像素紧贴凸起顶部下方堆叠，受框架顶部内壁限制，多余像素被截断（保留底部的）... → 彩色像素居中放置

"""
Design analysis for the D18 variant.

a) Concrete behavior and trace
   - The scene keeps the original objects: a grey open-left frame on the right, a grey bar with four upward
     projections on the left, and four colored slot columns inside the frame at fixed odd columns.
   - The bar still translates into the frame so each projection column aligns with one colored slot column.
   - The changed step is only the final vertical placement of the colored cells inside each aligned column.
   - Original logic:
       available rows = from the frame's inner top row down to the cell just above the projection top
       colors kept = bottom-most min(original_count, available_rows) cells
       placement = flush to the bottom of that gap, touching the projection
   - New logic:
       available rows = same gap as above
       colors kept = same truncation rule as before
       placement = center the kept block inside that gap
     Example with interior rows 1..4 available and 2 colored cells:
       original output rows = 3,4
       new output rows = 2,3
     The same counts are preserved, but the block is no longer anchored to the projection.

b) Complexity calibration
   - The geometry stays fixed, so object detection does not become ambiguous.
   - The new degree of freedom is the vertical offset inside each column gap. This is visible without adding
     new object types or hidden metadata.
   - Pure centering can visually collapse back to the original behavior when the gap is exactly filled.
     To avoid weak samples, the generator forces at least two columns to have strict spare space above and below
     the final color block.

c) Implicit assumption audit against the original main()
   - Copying only the frame bbox and clearing colored cells is still valid: the transform still happens entirely
     inside the slot.
   - The bar translation and projection-column alignment are unchanged.
   - The original line `start_row = desired_bottom - len(colors) + 1` depended on bottom anchoring. That
     assumption is no longer valid; it is replaced by a centered offset computed from the full free gap.
   - The original truncation `colors = colors[-max_len:]` remains valid. The variant changes position, not which
     subset survives when the gap is too small.
   - Discrete centering can leave one extra empty cell on only one side when the slack is odd. This code uses
     floor division, so the extra cell stays below the block. The generator does not rely only on that marginal
     case; it guarantees columns with at least one empty cell on both sides.

d) Placement legality
   - Frame bbox, slot columns, and bar geometry stay within the original 8x20 canvas.
   - The translated bar still lands at rows 2..5 and cols 11..17, inside the frame opening.
   - Centered color placement only uses rows between the inner top wall and the projection top minus one, so it
     never overlaps the bar or escape the frame.
   - Generator-side expected outputs are built using the same placement formula as the solver.

e) Visual observability
   - The difference from the original task is directly visible: color blocks float away from both the top wall
     and the projection in at least two columns per sample.
   - No rule depends on color id ordering, arithmetic on color ids, or any invisible attribute.

f) Visual neatness
   - The variant keeps the same sparse four-column layout, so the output remains easy to parse.
   - Grey caps from the input can still remain in some rows, but only in the same narrow slot columns as before;
     no extra overlapping regions are introduced.

g) Color-change inferability
   - There is no new color mapping. Each slot column keeps its own input color, and only vertical placement
     changes.
"""

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


def _centered_start(interior_top: int, desired_bottom: int, visible_len: int) -> int:
    """Center a visible stack inside the available gap, biasing one extra slack cell downward when needed."""
    gap_len = desired_bottom - interior_top + 1
    free = gap_len - visible_len
    return interior_top + (free // 2)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide a grey bar into a grey frame, align its projections to the colored slot columns, then re-place colors.

    Input structure:
    - The most common non-background color is a grey structure split into two 4-connected components:
      a large right-side frame and a smaller left-side bar with four upward projections.
    - Inside the frame, four colored columns indicate which bar projection columns must be matched.

    Rule:
    - Translate the bar upward into the frame so its base sits one cell above the frame's interior floor.
    - Shift it horizontally so the projection columns align with the colored columns in the frame.
    - For each aligned colored column, keep the same top-to-bottom color order, truncate from the top side if the
      gap is too short, and then center the surviving color stack within the space between the frame's inner top
      wall and the top of the matched projection.

    Key insight:
    - The frame and bar still determine the same target columns as in the original task; the only new reasoning is
      that the color block is positioned by measuring the whole vertical gap and centering within it instead of
      anchoring directly against the projection.
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
        rr = r + dy
        cc = c + dx
        if sr0 <= rr <= sr1 and sc0 <= cc <= sc1:
            out[rr, cc] = grey
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

        rows = [r for r in range(sr0, sr1 + 1) if g[r, col] != bg and g[r, col] != grey]
        if not rows:
            continue
        colors = [int(g[r, col]) for r in rows]

        gap_len = desired_bottom - interior_top + 1
        if gap_len <= 0:
            continue
        if len(colors) > gap_len:
            colors = colors[-gap_len:]

        start_row = _centered_start(interior_top, desired_bottom, len(colors))
        for i, colv in enumerate(colors):
            rr = start_row + i
            if interior_top <= rr <= interior_bottom:
                out[rr, col] = colv

    return out


# --- Palette ---
BG = 0
GREY = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)

# --- Canvas geometry (fixed per spec) ---
H, W = 8, 20

# --- Frame (right-hand slot) ---
FRAME_R0, FRAME_C0 = 0, 9
FRAME_H, FRAME_W = 8, 11
LEFT_WALL_OPEN_ROWS = (4, 5)

# --- Slot columns inside the frame ---
SLOT_COLS = [11, 13, 15, 17]
SLOT_LENS = [3, 4, 4, 3]
SLOT_TOP_ROW = 1

# --- Bar geometry ---
BAR_INPUT_C0 = 1
BAR_OUTPUT_C0 = 11
BAR_R0 = 2
BAR_H, BAR_W = 4, 7
BAR_LOCAL_PROJ_COLS = [0, 2, 4, 6]
BAR_BASE_ABS_ROW = BAR_R0 + BAR_H - 1

# --- Sampling parameters ---
HEIGHT0_RANGE = (2, 5)
HEIGHT_REST_RANGE = (1, 5)
MAX_ATTEMPTS = 300
STRICT_CENTERED_COLS = 2


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


def _projection_gap(height: int) -> int:
    return BAR_BASE_ABS_ROW - height


def _centered_span(height: int, count: int) -> tuple[int, int] | None:
    gap_len = _projection_gap(height)
    if gap_len <= 0:
        return None
    visible = min(count, gap_len)
    desired_bottom = SLOT_TOP_ROW + gap_len - 1
    start = _centered_start(SLOT_TOP_ROW, desired_bottom, visible)
    return start, visible


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

    span = _centered_span(height, count)
    if span is None:
        return
    start, visible = span
    if visible > 0:
        grid[start : start + visible, col] = color


def _build_grid(
    heights: list[int],
    colors: list[int],
    colored_counts: list[int],
    bar_c0: int,
    output_mode: bool,
) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    paste(grid, _bar_patch(heights), BAR_R0, bar_c0)
    for col, length, color, count, height in zip(SLOT_COLS, SLOT_LENS, colors, colored_counts, heights):
        _paint_slot_column(grid, col, length, color, count, height if output_mode else None)
    return grid


def _strict_center_options(height: int, length: int) -> list[int]:
    gap_len = _projection_gap(height)
    upper = min(length, gap_len - 2)
    if upper < 1:
        return []
    return list(range(1, upper + 1))


def _sample_params(rng: np.random.Generator) -> tuple[list[int], list[int], list[int]]:
    colors = [int(v) for v in rng.choice(PALETTE, size=len(SLOT_COLS), replace=False)]

    for _ in range(MAX_ATTEMPTS):
        heights = [int(rng.integers(*HEIGHT0_RANGE))]
        heights.extend(int(rng.integers(*HEIGHT_REST_RANGE)) for _ in range(len(SLOT_COLS) - 1))

        strict_options = [_strict_center_options(h, L) for h, L in zip(heights, SLOT_LENS)]
        eligible = [i for i, opts in enumerate(strict_options) if opts]
        if len(eligible) < STRICT_CENTERED_COLS:
            continue

        colored_counts = [int(rng.integers(1, L + 1)) for L in SLOT_LENS]
        chosen = rng.choice(eligible, size=STRICT_CENTERED_COLS, replace=False)
        for idx in chosen:
            colored_counts[int(idx)] = int(rng.choice(strict_options[int(idx)]))
        return heights, colors, colored_counts

    raise RuntimeError("failed to sample heights/counts with visible centered columns")


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    heights, colors, colored_counts = _sample_params(rng)
    grid = _build_grid(heights, colors, colored_counts, BAR_INPUT_C0, output_mode=False)
    expected = _build_grid(heights, colors, colored_counts, BAR_OUTPUT_C0, output_mode=True)

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
