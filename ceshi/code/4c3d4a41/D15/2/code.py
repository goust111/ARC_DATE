# Variant of 4c3d4a41
# Dimension: D15 — src-dst基数关系 (multi_step)
# Change: 1:1关系 - 1个杆对1个框架... → 多杆对1框架(N:1)

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg as _bg_color
from arc_utils import colors_present, get_bg, paste


# Design analysis for the D15 variant
# -----------------------------------
# a) Concrete behavior
#    The right-hand grey frame and the four colored slot columns stay unchanged.
#    The change is on the source side: instead of one 4-projection bar, the input now
#    contains 2 or 3 disconnected grey mini-bars. Each mini-bar has 1 or 2 upward
#    projections, and the total number of projections across all mini-bars is still 4.
#    In the output, these mini-bars are inserted into the same frame one after another:
#    the topmost source mini-bar claims the leftmost remaining destination columns, the
#    next source mini-bar claims the next destination columns, and so on. Within each
#    claimed region the old per-column rule is unchanged: the projection column aligns
#    to the colored slot column, and the colored pixels stack above that projection.
#    Example trace: under the old logic a single source bar with 4 projections aligned
#    to all four slot columns at once. Under the variant, if the source pieces have
#    projection counts [2,1,1] from top to bottom, then the top piece aligns to the
#    first two slot columns, the middle piece aligns to the third slot column, and the
#    bottom piece aligns to the fourth slot column.
# b) Complexity calibration
#    The grouping is restricted to [2,2], [2,1,1], [1,2,1], or [1,1,2]. This is enough
#    to make the N:1 composition visible without creating too many tiny ambiguous cases.
#    Every piece has at least one true upward projection because all sampled projection
#    heights are >= 2. The top-to-bottom ordering is repeated structurally in each grid,
#    so the multi-step behavior is observable from a few examples.
# c) Review of original assumptions
#    The original solver assumed exactly one movable bar: it took the second-largest grey
#    component, computed one dx, and pasted one shape. Those assumptions no longer hold.
#    The new solver must:
#    - collect every grey component except the largest frame component,
#    - sort them by vertical position instead of selecting one,
#    - allocate destination columns cumulatively according to each piece's projection
#      count, rather than aligning all destination columns at once.
#    The original color-moving code was mostly safe because it already operated per
#    column; it now reads the placed columns produced by all pieces together.
# d) Placement legality
#    Source pieces are drawn in fixed, non-overlapping staging slots on the left side of
#    the canvas. The top slot uses a lower max height so it fits above the frame and does
#    not touch the lower source pieces. Output placement uses consecutive destination
#    groups over the fixed slot columns [11,13,15,17], so pieces never overlap inside the
#    frame either.
# e) Visual observability
#    The key new signal is visible: there are multiple disconnected source bars, and the
#    output contains multiple disconnected inserted bar segments inside one frame. The
#    ordering rule uses visible geometry only: source vertical order and left-to-right
#    destination order. No invisible color-id ordering is used.
# f) Visual cleanliness
#    The source area uses only 2 or 3 pieces with ample separation, avoiding clutter.
#    The frame still contains only the original four colored columns plus the inserted
#    grey pieces, so the final image remains easy to parse.
# g) Color inference
#    Colors behave exactly as in the base task: each slot column keeps its own color and
#    only changes vertical position relative to the inserted projection beneath it. No new
#    color mapping is introduced.


def _bbox(pts):
    return bbox_coords(pts if isinstance(pts, np.ndarray) else np.asarray(list(pts), dtype=int))


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _most_common(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _piece_projection_cols(pts: np.ndarray) -> tuple[int, list[int]]:
    """Return base row and absolute columns that truly project above the base."""
    base_row = max(int(r) for r, _ in pts)
    col_min_row: dict[int, int] = {}
    for r, c in pts:
        r = int(r)
        c = int(c)
        col_min_row[c] = min(col_min_row.get(c, r), r)
    proj_cols = sorted(c for c, rmin in col_min_row.items() if rmin < base_row)
    return base_row, proj_cols


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fit several grey mini-bars into one grey frame by consuming the colored columns in order.

    Input structure:
    - bg plus a grey structural color forming one large open frame on the right and 2-3 smaller grey bar pieces elsewhere.
    - Each small bar has 1 or 2 upward projections; together they account for all colored target columns inside the frame.
    - Colored pixels inside the frame still mark the destination columns, one color per column.

    Rule:
    - Remove every source bar from its original location and keep only the frame region.
    - Sort the source bars from top to bottom. The first bar aligns to the leftmost remaining colored columns in the frame, the next bar to the next columns, and so on.
    - Move each bar upward into the frame so its base sits one cell above the frame's interior floor, then stack each column's original colored pixels immediately above the corresponding inserted projection.

    Key insight:
    - The puzzle is no longer a single global alignment. Each disconnected source bar claims a consecutive block of destination columns, and the source bars must be assembled in visible top-to-bottom order to reconstruct the full arrangement inside the single frame.
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

    (_, slot_bbox, _), piece_comps = grey_comps[0], grey_comps[1:]
    if not piece_comps:
        return g.copy()
    sr0, sc0, sr1, sc1 = slot_bbox

    out = np.full_like(g, bg)
    out[sr0 : sr1 + 1, sc0 : sc1 + 1] = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region = out[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region[(slot_region != bg) & (slot_region != grey)] = bg

    dst_cols = sorted(
        {
            int(c)
            for r, c in np.argwhere((g != bg) & (g != grey))
            if sr0 <= int(r) <= sr1 and sc0 <= int(c) <= sc1
        }
    )
    if not dst_cols:
        return out

    slot_interior_bottom = sr1 - 1
    target_base_row = slot_interior_bottom - 1

    placed: list[tuple[int, int]] = []
    cursor = 0
    ordered_pieces = sorted(piece_comps, key=lambda t: (t[1][0], t[1][1]))
    for _, _bbox_piece, pts in ordered_pieces:
        piece_pts = np.asarray(pts, dtype=int)
        base_row, src_proj_cols = _piece_projection_cols(piece_pts)
        proj_count = len(src_proj_cols)
        if proj_count == 0 or cursor + proj_count > len(dst_cols):
            continue
        target_cols = dst_cols[cursor : cursor + proj_count]
        dx = target_cols[0] - src_proj_cols[0]
        dy = target_base_row - base_row

        for r, c in piece_pts:
            rr = int(r) + dy
            cc = int(c) + dx
            if sr0 <= rr <= sr1 and sc0 <= cc <= sc1:
                out[rr, cc] = grey
                placed.append((rr, cc))
        cursor += proj_count

    placed_by_col: dict[int, list[int]] = {}
    for rr, cc in placed:
        placed_by_col.setdefault(int(cc), []).append(int(rr))

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
        max_len = desired_bottom - interior_top + 1
        if max_len <= 0:
            continue
        if len(colors) > max_len:
            colors = colors[-max_len:]
        start_row = max(interior_top, desired_bottom - len(colors) + 1)

        for i, colv in enumerate(colors):
            rr = start_row + i
            if interior_top <= rr <= interior_bottom:
                out[rr, col] = colv

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
TARGET_BASE_ROW = 5

# Source staging slots: (base_row, left_col, max_height)
SOURCE_SLOTS = [
    (2, 0, 3),  # top slot; shortened so it never hits the upper boundary
    (5, 5, 4),  # middle slot
    (7, 0, 3),  # bottom slot; also shortened so source-piece ordering is unambiguous
]

PARTITIONS = [
    (2, 2),
    (2, 1, 1),
    (1, 2, 1),
    (1, 1, 2),
]

MAX_ATTEMPTS = 400


def _frame_patch() -> np.ndarray:
    patch = np.zeros((FRAME_H, FRAME_W), dtype=int)
    patch[[0, -1], :] = GREY
    patch[:, -1] = GREY
    patch[:, 0] = GREY
    patch[list(LEFT_WALL_OPEN_ROWS), 0] = BG
    return patch


def _piece_patch(heights: list[int]) -> np.ndarray:
    width = 2 * len(heights) - 1
    height = max(heights)
    patch = np.zeros((height, width), dtype=int)
    patch[-1, :] = GREY
    for i, proj_h in enumerate(heights):
        local_c = 2 * i
        patch[height - proj_h : height, local_c] = GREY
    return patch


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

    visible = min(count, TARGET_BASE_ROW - height)
    if visible > 0:
        row_end = TARGET_BASE_ROW - height
        grid[row_end - visible + 1 : row_end + 1, col] = color


def _draw_source_pieces(grid: np.ndarray, groups: list[list[int]], heights: list[int]) -> None:
    slot_ids = [0, 2] if len(groups) == 2 else [0, 1, 2]
    cursor = 0
    for slot_id, group in zip(slot_ids, groups):
        base_row, c0, _ = SOURCE_SLOTS[slot_id]
        piece_heights = heights[cursor : cursor + len(group)]
        patch = _piece_patch(piece_heights)
        r0 = base_row - patch.shape[0] + 1
        paste(grid, patch, r0, c0)
        cursor += len(group)


def _draw_output_pieces(grid: np.ndarray, groups: list[list[int]], heights: list[int]) -> None:
    cursor = 0
    for group in groups:
        piece_heights = heights[cursor : cursor + len(group)]
        patch = _piece_patch(piece_heights)
        c0 = SLOT_COLS[cursor]
        r0 = TARGET_BASE_ROW - patch.shape[0] + 1
        paste(grid, patch, r0, c0)
        cursor += len(group)


def _build_input(
    groups: list[list[int]],
    heights: list[int],
    colors: list[int],
    colored_counts: list[int],
) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    _draw_source_pieces(grid, groups, heights)
    for col, length, color, count in zip(SLOT_COLS, SLOT_LENS, colors, colored_counts):
        _paint_slot_column(grid, col, length, color, count, None)
    return grid


def _build_output(
    groups: list[list[int]],
    heights: list[int],
    colors: list[int],
    colored_counts: list[int],
) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    _draw_output_pieces(grid, groups, heights)
    for col, length, color, count, height in zip(SLOT_COLS, SLOT_LENS, colors, colored_counts, heights):
        _paint_slot_column(grid, col, length, color, count, height)
    return grid


def _partition_to_groups(partition: tuple[int, ...]) -> list[list[int]]:
    groups = []
    cursor = 0
    for size in partition:
        groups.append(list(range(cursor, cursor + size)))
        cursor += size
    return groups


def _sample_params(
    rng: np.random.Generator,
) -> tuple[list[list[int]], list[int], list[int], list[int]]:
    partition = PARTITIONS[int(rng.integers(0, len(PARTITIONS)))]
    groups = _partition_to_groups(partition)

    colors = [int(v) for v in rng.choice(PALETTE, size=len(SLOT_COLS), replace=False)]
    colored_counts = [int(rng.integers(1, L + 1)) for L in SLOT_LENS]

    slot_ids = [0, 2] if len(groups) == 2 else [0, 1, 2]
    heights: list[int] = []
    for group, slot_id in zip(groups, slot_ids):
        _base_row, _c0, max_h = SOURCE_SLOTS[slot_id]
        for _ in group:
            heights.append(int(rng.integers(2, max_h + 1)))
    return groups, heights, colors, colored_counts


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    groups, heights, colors, colored_counts = _sample_params(rng)
    grid = _build_input(groups, heights, colors, colored_counts)
    expected = _build_output(groups, heights, colors, colored_counts)

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
