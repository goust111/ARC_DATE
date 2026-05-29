# Variant of 4c3d4a41
# Dimension: D5 — 移动方向 (trajectory)
# Change: 杆水平移动（从左侧移入右侧框架），彩色像素垂直移动（向上堆叠）；原位置清空... → 垂直移动进入框架

import numpy as np

from arc_utils import bbox_coords, colors_present, components_mask, get_bg, get_bg as _bg_color, paste


# Design analysis for the D5 variant
# a) Concrete behavior:
#    The bar keeps its original horizontal shape with upward projections, but it now hangs above a top-open U-frame
#    and drops straight downward into it. To make that possible on the same 8x20 canvas, the frame is shortened and
#    the colored marker columns are lowered. The output keeps the frame and its grey caps, removes the original bar,
#    drops the bar two rows, then stacks the colored pixels in the same columns above the dropped projections.
# b) Complexity calibration:
#    The variant keeps four distinct target columns and four independent projection heights, so the input still has a
#    non-trivial column-matching structure. Only the frame height and marker lengths are compressed enough to make the
#    vertical entrance feasible on the fixed canvas.
# c) Hidden assumptions from the original solver that must change:
#    - Copying the whole slot bbox was safe before because the bar stayed completely outside that bbox. Here the bar
#      partially intrudes through the top opening, so those old bar pixels must be cleared explicitly.
#    - The original interior top was one row below the frame's top wall. Here the opening is on the top edge, so the
#      accessible stacking region starts at the frame's top row itself.
#    - Frame-top grey cells adjacent to the opening would fuse with tall bar projections into one component, so the
#      top edge is reduced to two side posts only.
# d) Placement legality:
#    The frame occupies rows 2..7, cols 9..19. The input bar occupies rows 0..3, cols 11..17, staying separated from
#    both the frame and the lowered slot markers. Marker columns start at row 5 so no grey cap can connect to the bar.
# e) Visual observability:
#    In every sample at least one projection has height 1 and at least one has height >= 2, so some colors rise higher
#    than others after the drop. At least one of the two middle columns has count 2, ensuring a visible multi-cell
#    colored stack rather than reducing everything to single pixels.
# f) Visual cleanliness:
#    The opening is wide, the bar uses the same four odd columns as before, and the marker columns remain sparse inside
#    the frame. No region relies on hidden ordering by color id.
# g) New solver assumptions and generator guarantees:
#    - The largest grey component is the frame and the second largest is the bar.
#    - The bar is horizontally centered over the top opening, so a pure vertical drop is the intended move.
#    - Colored markers live only in the four target columns and only below the hanging bar in the input.
#    The generator enforces all three conditions directly.


def _bbox(pts):
    return bbox_coords(pts if isinstance(pts, np.ndarray) else np.asarray(list(pts), dtype=int))


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _most_common(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop a grey projection bar vertically through a top opening, then stack the colors above it.

    Input structure:
    - bg plus a grey structural color form two main components: a top-open U-frame and a smaller horizontal bar
      hanging above the opening with upward projections.
    - Four colored marker columns sit lower inside the frame and indicate which projection columns matter.

    Rule:
    - Remove the bar from its input position and drop it straight down so its base ends one row above the frame floor.
    - Keep the bar centered on the top opening rather than sliding it sideways from outside the frame.
    - For each colored target column, move the colored pixels upward so they stack immediately above the dropped
      projection in that same column.

    Key insight: The same projection-to-column correspondence is preserved, but the motion cue changes from a lateral
    insertion into a vertical descent through the frame's top opening.
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

    out = np.full_like(g, bg)
    out[sr0 : sr1 + 1, sc0 : sc1 + 1] = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region = out[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region[(slot_region != bg) & (slot_region != grey)] = bg

    # The hanging input bar partly intrudes into the slot bbox, so clear that old footprint explicitly.
    for r, c in bar_pts:
        if sr0 <= r <= sr1 and sc0 <= c <= sc1:
            out[r, c] = bg

    bar_base_row = max(r for r, _ in bar_pts)
    slot_interior_bottom = sr1 - 1
    target_base_row = slot_interior_bottom - 1
    dy = target_base_row - bar_base_row

    opening_cols = [c for c in range(sc0, sc1 + 1) if g[sr0, c] == bg]
    opening_center = (min(opening_cols) + max(opening_cols)) // 2 if opening_cols else (sc0 + sc1) // 2
    bar_center = (bar_bbox[1] + bar_bbox[3]) // 2
    dx = opening_center - bar_center

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

    stack_top = sr0
    stack_bottom = sr1 - 1
    dst_cols = sorted(
        {
            int(c)
            for r, c in np.argwhere((g != bg) & (g != grey))
            if sr0 <= r <= sr1 and sc0 <= c <= sc1
        }
    )

    for col in dst_cols:
        if col not in placed_by_col:
            continue
        bar_top = min(placed_by_col[col])
        desired_bottom = bar_top - 1
        if desired_bottom < stack_top:
            continue

        rows = [r for r in range(sr0, sr1 + 1) if g[r, col] != bg and g[r, col] != grey]
        if not rows:
            continue
        colors = [int(g[r, col]) for r in rows]

        max_len = desired_bottom - stack_top + 1
        if len(colors) > max_len:
            colors = colors[-max_len:]

        start_row = max(stack_top, desired_bottom - len(colors) + 1)
        for i, colv in enumerate(colors):
            rr = start_row + i
            if stack_top <= rr <= stack_bottom:
                out[rr, col] = colv

    return out


# --------------- generator ---------------

BG = 0
GREY = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)

H, W = 8, 20

FRAME_R0, FRAME_C0 = 2, 9
FRAME_H, FRAME_W = 6, 11

SLOT_COLS = [11, 13, 15, 17]
SLOT_LENS = [1, 2, 2, 1]
INPUT_SLOT_TOP_ROW = 5

BAR_INPUT_R0 = 0
BAR_OUTPUT_R0 = 2
BAR_C0 = 11
BAR_H, BAR_W = 4, 7
BAR_LOCAL_PROJ_COLS = [0, 2, 4, 6]
BAR_OUTPUT_BASE_ROW = BAR_OUTPUT_R0 + BAR_H - 1
OUTPUT_STACK_TOP_ROW = FRAME_R0

HEIGHT0_RANGE = (2, 4)   # {2, 3}; keep the first projection visibly taller
HEIGHT_REST_RANGE = (1, 4)  # {1, 2, 3}
MAX_ATTEMPTS = 300


def _frame_patch() -> np.ndarray:
    """Top-open U-frame: only two top posts, side walls, and a closed bottom."""
    patch = np.zeros((FRAME_H, FRAME_W), dtype=int)
    patch[0, 0] = GREY
    patch[0, -1] = GREY
    patch[1:-1, 0] = GREY
    patch[1:-1, -1] = GREY
    patch[-1, :] = GREY
    return patch


def _bar_patch(heights: list[int]) -> np.ndarray:
    """Horizontal bar with upward projections at the four odd columns."""
    patch = np.zeros((BAR_H, BAR_W), dtype=int)
    patch[-1, :] = GREY
    for local_c, height in zip(BAR_LOCAL_PROJ_COLS, heights):
        patch[BAR_H - height : BAR_H, local_c] = GREY
    return patch


def _paint_input_column(grid: np.ndarray, col: int, length: int, color: int, count: int) -> None:
    cap = length - count
    grid[INPUT_SLOT_TOP_ROW : INPUT_SLOT_TOP_ROW + cap, col] = GREY
    grid[INPUT_SLOT_TOP_ROW + cap : INPUT_SLOT_TOP_ROW + length, col] = color


def _paint_output_structure(grid: np.ndarray, col: int, length: int, count: int) -> None:
    cap = length - count
    grid[INPUT_SLOT_TOP_ROW : INPUT_SLOT_TOP_ROW + cap, col] = GREY


def _paint_output_colors(grid: np.ndarray, col: int, color: int, count: int, height: int) -> None:
    desired_bottom = BAR_OUTPUT_BASE_ROW - height
    max_len = desired_bottom - OUTPUT_STACK_TOP_ROW + 1
    if max_len <= 0:
        return
    visible = min(count, max_len)
    start_row = desired_bottom - visible + 1
    grid[start_row : desired_bottom + 1, col] = color


def _build_input(heights: list[int], colors: list[int], colored_counts: list[int]) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    paste(grid, _bar_patch(heights), BAR_INPUT_R0, BAR_C0)
    for col, length, color, count in zip(SLOT_COLS, SLOT_LENS, colors, colored_counts):
        _paint_input_column(grid, col, length, color, count)
    return grid


def _build_expected(heights: list[int], colors: list[int], colored_counts: list[int]) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    paste(grid, _bar_patch(heights), BAR_OUTPUT_R0, BAR_C0)
    for col, length, color, count, height in zip(SLOT_COLS, SLOT_LENS, colors, colored_counts, heights):
        _paint_output_structure(grid, col, length, count)
        _paint_output_colors(grid, col, color, count, height)
    return grid


def _sample_params(rng: np.random.Generator) -> tuple[list[int], list[int], list[int]]:
    colors = [int(v) for v in rng.choice(PALETTE, size=len(SLOT_COLS), replace=False)]
    colored_counts = [int(rng.integers(1, length + 1)) for length in SLOT_LENS]

    heights = [int(rng.integers(*HEIGHT0_RANGE))]
    heights.extend(int(rng.integers(*HEIGHT_REST_RANGE)) for _ in range(len(SLOT_COLS) - 1))

    if not any(h == 1 for h in heights):
        idx = int(rng.integers(1, len(heights)))
        heights[idx] = 1
    if not any(h >= 2 for h in heights):
        heights[0] = 2

    if colored_counts[1] == 1 and colored_counts[2] == 1:
        colored_counts[int(rng.choice([1, 2]))] = 2

    return heights, colors, colored_counts


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    heights, colors, colored_counts = _sample_params(rng)
    grid = _build_input(heights, colors, colored_counts)
    expected = _build_expected(heights, colors, colored_counts)

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
