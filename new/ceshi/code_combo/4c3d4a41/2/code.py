# Combo variant of 4c3d4a41 (index 2)
# combo_spec: D1/3+D5/2
# Name: 镜像对齐+垂直进入

import numpy as np

from arc_utils import bbox_coords, colors_present, components_mask, get_bg, get_bg as _bg_color, paste

# Design analysis
# a) Concrete behavior and trace:
#    The base task used a left-side opening and a horizontal slide. D5/2 rotates that
#    cue into a vertical one: the grey frame becomes a top-open U, the bar hangs above
#    the opening, and the solved state is obtained by dropping the bar straight down.
#    D1/3 then changes which projection supports each colored column: before landing,
#    the whole bar is mirrored left-right inside its own 4x7 bbox, so slot col 11 is
#    supported by the original rightmost projection, slot col 13 by the next one, etc.
#    Example with input heights [3,1,2,1]:
#    - plain D5/2 output skyline at cols [11,13,15,17] would use [3,1,2,1];
#    - fused output skyline uses [1,2,1,3] after the vertical drop.
#    Colors themselves stay in the same four frame columns; only the supporting grey
#    profile is mirrored while the motion remains vertical.
# b) Complexity calibration:
#    The canvas stays 8x20 with exactly one frame, one bar, and four colored columns.
#    To keep the variant readable on a fixed canvas, the frame uses the compact D5/2
#    geometry: rows 2..7, cols 9..19, opening along the top edge, marker lengths [1,2,2,1].
#    Heights are still sampled from a small discrete set, and generator rejection keeps
#    only cases where mirrored-vs-direct drop is visually different without introducing
#    a fixed left/right height bias that would dilute the mirror cue.
# c) Hidden assumptions from the original solver that break:
#    - "Frame copy is enough" fails because the hanging bar already overlaps the frame bbox.
#      The solver must explicitly clear old bar pixels inside that bbox before repainting.
#    - "Interior stacking starts below a solid top wall" fails because the opening is on
#      row sr0, so the accessible stacking region starts at sr0 itself.
#    - "Bar placement is a pure translation" fails because the fused rule mirrors the bar
#      first and only then drops it.
#    - "Projection i supports slot column i" fails because the support order is reversed.
# d) Placement legality:
#    The frame occupies rows 2..7, cols 9..19. The input bar occupies rows 0..3,
#    cols 11..17, separated from the frame by the open roof and from marker columns by
#    starting those markers at row 5. After dropping, the mirrored bar lands at rows 2..5
#    within the frame interior, with base row 5 and no out-of-bounds cells.
# e) Visual observability:
#    Every accepted sample has a non-palindromic height profile, so the mirrored skyline
#    differs from the direct-drop skyline. The generator also forces at least one height-1
#    projection and one taller projection, plus at least one middle column with two colored
#    cells, making both the drop direction and the mirror reversal visible in few examples.
# f) Visual cleanliness:
#    Only three regions carry information: the hanging bar above, the U-frame, and four
#    sparse marker columns inside. Even columns remain empty, so each colored stack reads
#    as a separate lane and the mirrored support order is easy to inspect.
# g) New solver assumptions and generator guarantees:
#    - Largest grey component is the U-frame; second-largest is the hanging bar.
#    - The top opening is wide and centered, so opening-center alignment gives the intended
#      vertical drop without horizontal ambiguity.
#    - Colored markers appear only in the four odd interior columns and use one color each.
#    - The fused effect is guaranteed by rejecting inputs whose mirrored drop matches the
#      non-mirrored drop. This directly prevents degeneration back to plain D5/2.


def _bbox(pts):
    return bbox_coords(pts if isinstance(pts, np.ndarray) else np.asarray(list(pts), dtype=int))


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _most_common(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _mirror_bar_points(bar_pts, bar_bbox):
    """Reflect the bar left-right inside its own bbox before placement."""
    _br0, bc0, _br1, bc1 = bar_bbox
    return [(int(r), int(bc0 + bc1 - c)) for r, c in bar_pts]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop a grey projection bar through a top opening after mirroring it, then restack colors above it.

    Input structure:
    - The dominant background surrounds two grey components: a top-open U-frame lower on the right and a smaller horizontal bar hanging above that opening.
    - Four colored marker columns sit inside the frame; they keep their columns, but the bar that will support them is not used in its original left-to-right order.

    Rule:
    - Remove the hanging bar from its input position, flip it horizontally within its own bounding box, and then drop the mirrored shape straight down through the top opening.
    - Stop the dropped bar so its base ends one row above the frame floor, without lateral sliding other than centering on the opening.
    - For each colored frame column, move that column's colors upward so they stack immediately above the mirrored projection now sitting underneath it.

    Key insight: two spatial changes must be combined at once: the motion is vertical rather than side-entry, and the supporting projection order is reversed by a left-right mirror before the bar lands.
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

    for r, c in bar_pts:
        if sr0 <= r <= sr1 and sc0 <= c <= sc1:
            out[r, c] = bg

    bar_base_row = max(r for r, _ in bar_pts)
    target_base_row = (sr1 - 1) - 1
    dy = target_base_row - bar_base_row

    mirrored_bar_pts = _mirror_bar_points(bar_pts, bar_bbox)

    opening_cols = [c for c in range(sc0, sc1 + 1) if g[sr0, c] == bg]
    opening_center = (min(opening_cols) + max(opening_cols)) // 2 if opening_cols else (sc0 + sc1) // 2
    bar_center = (bar_bbox[1] + bar_bbox[3]) // 2
    dx = opening_center - bar_center

    placed = []
    for r, c in mirrored_bar_pts:
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
        if max_len <= 0:
            continue
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

HEIGHT_RANGE = (1, 4)
MAX_ATTEMPTS = 300


def _frame_patch() -> np.ndarray:
    """Top-open U-frame with two roof posts, side walls, and a closed floor."""
    patch = np.zeros((FRAME_H, FRAME_W), dtype=int)
    patch[0, 0] = GREY
    patch[0, -1] = GREY
    patch[1:-1, 0] = GREY
    patch[1:-1, -1] = GREY
    patch[-1, :] = GREY
    return patch


def _bar_patch(heights: list[int]) -> np.ndarray:
    """Horizontal bar with upward projections at the four odd local columns."""
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
    mirrored_heights = list(reversed(heights))
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    paste(grid, _bar_patch(mirrored_heights), BAR_OUTPUT_R0, BAR_C0)
    for col, length, color, count, height in zip(SLOT_COLS, SLOT_LENS, colors, colored_counts, mirrored_heights):
        _paint_output_structure(grid, col, length, count)
        _paint_output_colors(grid, col, color, count, height)
    return grid


def _build_direct_drop_expected(heights: list[int], colors: list[int], colored_counts: list[int]) -> np.ndarray:
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

    heights = [int(rng.integers(*HEIGHT_RANGE)) for _ in range(len(SLOT_COLS))]

    if not any(h == 1 for h in heights):
        heights[int(rng.integers(len(heights)))] = 1
    if not any(h >= 2 for h in heights):
        heights[int(rng.integers(len(heights)))] = 2
    if heights == list(reversed(heights)):
        heights[-1] = 1 if heights[-1] != 1 else 2
        if heights == list(reversed(heights)):
            heights[0] = 3 if heights[0] != 3 else 2

    if colored_counts[1] == 1 and colored_counts[2] == 1:
        colored_counts[int(rng.choice([1, 2]))] = 2

    return heights, colors, colored_counts


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    heights, colors, colored_counts = _sample_params(rng)
    grid = _build_input(heights, colors, colored_counts)
    expected = _build_expected(heights, colors, colored_counts)
    direct_expected = _build_direct_drop_expected(heights, colors, colored_counts)

    if np.array_equal(expected, direct_expected):
        return None
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
