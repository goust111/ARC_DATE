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
    Fit a grey bar into a grey frame by aligning projections with colored columns, then stack colors above.

    Input structure:
    - bg, a grey structural color forming two components: a large rectangular frame (slot) and a smaller bar with upward projections.
    - Colored pixels inside the slot mark target columns.

    Rule:
    - Move the bar upward into the slot so its bottom edge sits one cell above the slot's interior bottom.
    - Shift it horizontally so the bar's upward projections align column-wise with the colored pixels in the slot.
    - Stack the colored pixels immediately above the bar projections in each column.

    Key insight: The bar's projections act as keys that must align with the colored-column "locks" in the slot.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    # Background: most frequent. "Grey" structural color: most frequent non-bg.
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

    # Slot is the largest grey component (the right-side frame); bar is the next largest.
    (_, slot_bbox, _), (_, bar_bbox, bar_pts) = grey_comps[0], grey_comps[1]
    sr0, sc0, sr1, sc1 = slot_bbox

    # Base output: keep only the slot bbox content, clear all colored pixels inside it.
    out = np.full_like(g, bg)
    out[sr0 : sr1 + 1, sc0 : sc1 + 1] = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region = out[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region[(slot_region != bg) & (slot_region != grey)] = bg

    # Compute bar translation: bottom of bar sits 1 cell above slot interior bottom.
    bar_base_row = max(r for r, _ in bar_pts)
    slot_interior_bottom = sr1 - 1
    target_base_row = slot_interior_bottom - 1
    dy = target_base_row - bar_base_row

    # Determine bar projection columns (columns with any bar pixel above the base row).
    col_min_row = {}
    for r, c in bar_pts:
        col_min_row[c] = min(col_min_row.get(c, r), r)
    src_proj_cols = sorted([c for c, rmin in col_min_row.items() if rmin < bar_base_row])

    # Determine destination columns from colored pixels in the original slot.
    dst_cols_set = set()
    for r, c in np.argwhere((g != bg) & (g != grey)):
        r = int(r)
        c = int(c)
        if sr0 <= r <= sr1 and sc0 <= c <= sc1:
            dst_cols_set.add(c)
    dst_cols = sorted(dst_cols_set)

    # Compute a robust horizontal shift aligning projection columns to colored columns.
    dx = None
    if src_proj_cols and len(src_proj_cols) == len(dst_cols):
        deltas = [d - s for s, d in zip(src_proj_cols, dst_cols)]
        if len(set(deltas)) == 1:
            dx = deltas[0]
    if dx is None:
        # Brute pick dx that maximizes overlap of shifted projection cols with dst cols.
        best = (-1, 0)
        dst_set = set(dst_cols)
        for cand_dx in range(-w, w + 1):
            score = sum((c + cand_dx) in dst_set for c in src_proj_cols)
            if score > best[0]:
                best = (score, cand_dx)
        dx = best[1]

    # Paste bar into the slot (overwrite with grey).
    placed = []
    for r, c in bar_pts:
        rr = r + dy
        cc = c + dx
        if sr0 <= rr <= sr1 and sc0 <= cc <= sc1:
            out[rr, cc] = grey
            placed.append((rr, cc))

    # Move colored pixels: for each affected column, stack them immediately above the top of the bar in that column.
    # Use colors from the original grid (inside slot bbox), ordered top->bottom.
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

        # Extract original colored pixels in this column (ignore bg/grey).
        rows = [r for r in range(sr0, sr1 + 1) if g[r, col] != bg and g[r, col] != grey]
        if not rows:
            continue
        colors = [int(g[r, col]) for r in rows]  # top->bottom order by construction

        max_len = desired_bottom - interior_top + 1
        if max_len <= 0:
            continue
        if len(colors) > max_len:
            colors = colors[-max_len:]

        start_row = desired_bottom - len(colors) + 1
        start_row = max(start_row, interior_top)

        # Paint (overwrite) within the interior only.
        for i, colv in enumerate(colors):
            rr = start_row + i
            if interior_top <= rr <= interior_bottom:
                out[rr, col] = colv

    return out

# --------------- generator (attempt 3 (Codex)) ---------------

# --- Palette ---
BG = 0
GREY = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)  # colors != bg and != grey

# --- Canvas geometry (fixed per spec) ---
H, W = 8, 20

# --- Frame (right-hand slot) ---
FRAME_R0, FRAME_C0 = 0, 9            # top-left of frame patch
FRAME_H, FRAME_W = 8, 11             # frame patch dimensions
LEFT_WALL_OPEN_ROWS = (4, 5)         # rows where the left wall of the frame is open

# --- Slot columns inside the frame ---
SLOT_COLS = [11, 13, 15, 17]         # absolute col indices
SLOT_LENS = [3, 4, 4, 3]             # per-col length counted from row 1
SLOT_TOP_ROW = 1                     # topmost interior row of slot columns

# --- Bar geometry ---
BAR_INPUT_C0 = 1                     # left column of bar in the input (pre-translation)
BAR_OUTPUT_C0 = 11                   # left column of bar in the expected output (post-translation)
BAR_R0 = 2                           # top row where bar patch is pasted
BAR_H, BAR_W = 4, 7                  # bar patch dimensions (4 rows tall, base at local row 3)
BAR_LOCAL_PROJ_COLS = [0, 2, 4, 6]   # bar projection columns within the patch
BAR_BASE_ABS_ROW = BAR_R0 + BAR_H - 1  # row 5, absolute row of bar base after paste

# --- Sampling parameters ---
HEIGHT0_RANGE = (2, 5)               # rng.integers(low, high) -> {2,3,4} for first proj
HEIGHT_REST_RANGE = (1, 5)           # rng.integers(low, high) -> {1,2,3,4} for other projs
MAX_ATTEMPTS = 300


def _frame_patch() -> np.ndarray:
    """Grey frame with top/bottom/right walls closed and a 2-cell gap in the left wall."""
    patch = np.zeros((FRAME_H, FRAME_W), dtype=int)
    patch[[0, -1], :] = GREY                       # top and bottom rows
    patch[:, -1] = GREY                            # right wall
    patch[:, 0] = GREY                             # left wall (solid...)
    patch[list(LEFT_WALL_OPEN_ROWS), 0] = BG       # ...with the opening carved out
    return patch


def _bar_patch(heights: list[int]) -> np.ndarray:
    """Horizontal bar base with 4 upward grey projections at local cols {0,2,4,6}."""
    patch = np.zeros((BAR_H, BAR_W), dtype=int)
    patch[-1, :] = GREY                            # full base row
    for local_c, height in zip(BAR_LOCAL_PROJ_COLS, heights):
        patch[BAR_H - height : BAR_H, local_c] = GREY
    return patch


def _paint_slot_column(grid: np.ndarray, col: int, length: int, color: int,
                       count: int, height: int | None) -> None:
    """Paint one slot column.

    Input mode (height=None): top (length-count) cells grey caps, bottom `count` cells the color.
    Output mode (height given): grey cap then up to (5-height) visible colored cells flush to top.
    """
    cap = length - count
    grid[SLOT_TOP_ROW : SLOT_TOP_ROW + cap, col] = GREY
    if height is None:
        grid[SLOT_TOP_ROW + cap : SLOT_TOP_ROW + length, col] = color
        return
    visible = min(count, BAR_BASE_ABS_ROW - height)  # = min(count, 5 - height)
    if visible > 0:
        row_end = BAR_BASE_ABS_ROW - height          # inclusive last colored row
        grid[row_end - visible + 1 : row_end + 1, col] = color


def _build_grid(heights: list[int], colors: list[int], colored_counts: list[int],
                bar_c0: int, output_mode: bool) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    paste(grid, _bar_patch(heights), BAR_R0, bar_c0)
    for col, length, color, count, height in zip(SLOT_COLS, SLOT_LENS, colors, colored_counts, heights):
        _paint_slot_column(grid, col, length, color, count, height if output_mode else None)
    return grid


def _sample_params(rng: np.random.Generator) -> tuple[list[int], list[int], list[int]]:
    colors = [int(v) for v in rng.choice(PALETTE, size=len(SLOT_COLS), replace=False)]
    colored_counts = [int(rng.integers(1, L + 1)) for L in SLOT_LENS]
    heights = [int(rng.integers(*HEIGHT0_RANGE))]
    heights.extend(int(rng.integers(*HEIGHT_REST_RANGE)) for _ in range(len(SLOT_COLS) - 1))
    return heights, colors, colored_counts


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
