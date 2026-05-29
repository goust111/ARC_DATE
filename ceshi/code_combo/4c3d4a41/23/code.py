# Combo variant of 4c3d4a41 (index 23)
# combo_spec: D2/2+D15/2
# Name: 多段杆+多杆对1框架

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg as _bg_color
from arc_utils import colors_present, get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    Seed task:
#    - one connected 7-wide source bar with four odd-column projections
#    - one destination frame containing four colored columns
#    - solve by one global horizontal shift that aligns all four projections at once
#
#    Combo task:
#    - the source is split into 2 or 3 disconnected fragments
#    - each fragment carries 1 or 2 projections, so the four destination columns are now
#      recovered piece-by-piece rather than by one monolithic bar
#    - the source fragments are staged in vertically separated slots on the left; their
#      top-to-bottom order is the assembly order into the single frame
#    - example: partition [2,1,1] means the top fragment claims slot cols [11,13], the
#      middle fragment claims [15], and the bottom fragment claims [17]
#
#    The important difference from the seed is that "find one dx for one object" no longer
#    applies. The solver must first detect multiple grey source components, then infer how
#    many destination columns each component owns, then place them cumulatively into one
#    shared frame.
#
# b) Complexity calibration
#    - Only four partition patterns are used: [2,2], [1,1,2], [1,2,1], [2,1,1].
#      This keeps the inference space small enough for ARC while still making the fragment
#      structure visually obvious.
#    - Each fragment is either width 1 (one projection) or width 3 (two projections).
#      That means the projection count is directly visible from shape, not hidden in color.
#    - Source order is encoded by vertical staging slots, not by color ids or metadata.
#
# c) Old assumptions that break
#    - "The second-largest grey component is the movable object" is false: there are now
#      2-3 movable components.
#    - "One global dx aligns source to destination" is false: each fragment gets its own dx.
#    - "Projection columns are read from one connected bar" is false: projection counting
#      must be done per fragment, then consumed in sequence.
#    - New assumption introduced by the combo: fragment order is read by vertical position.
#      The generator therefore uses separated source slots whose bboxes never overlap in a
#      way that would scramble this ordering.
#
# d) Placement legality
#    - The frame remains the largest grey component on the right.
#    - Source fragments are placed in fixed left-side staging slots with at least one blank
#      column before the frame, so they never merge with the frame.
#    - In output, fragment starts are determined by the fixed slot columns [11,13,15,17].
#      Because a 1-projection fragment has width 1 and a 2-projection fragment has width 3,
#      adjacent fragments remain disconnected inside the frame.
#
# e) Visual observability
#    - The input visibly contains multiple disconnected grey fragments, not one bar.
#    - The output visibly preserves that fragmentation inside one shared frame.
#    - The only ordering cue is geometric: top fragment enters first, next fragment enters
#      second, and so on. No invisible attribute is consulted.
#
# f) Visual cleanliness
#    - Only 2 or 3 fragments are used, with generous spacing between staging slots.
#    - The frame internals still use the original four fixed colored columns, so the puzzle
#      remains easy to parse despite the extra assembly step.
#
# g) Color inference
#    - Colors behave exactly like the seed task once a support column is placed: each slot
#      column keeps its original color stack, truncated only by the support height beneath it.
#      The combo changes structural assembly, not color mapping.
#
# h) Generator guarantees matched to solver assumptions
#    - Every source fragment has 1 or 2 true projections because all sampled heights are >= 2.
#    - The sum of fragment projection counts is always 4, matching the four destination cols.
#    - Vertical slot order is unambiguous because source bboxes occupy separated staging rows.
#    - The frame is always the largest grey component, so identifying the destination region
#      via the largest grey connected component is safe on generated samples.


def _bbox(pts):
    arr = pts if isinstance(pts, np.ndarray) else np.asarray(list(pts), dtype=int)
    return bbox_coords(arr)


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _most_common(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _piece_info(pts: np.ndarray) -> dict[str, object]:
    piece = np.asarray(pts, dtype=int)
    r0, c0, r1, c1 = _bbox(piece)
    base_row = int(piece[:, 0].max())

    col_min_row: dict[int, int] = {}
    for r, c in piece:
        rr = int(r)
        cc = int(c)
        col_min_row[cc] = min(col_min_row.get(cc, rr), rr)
    proj_cols = sorted(cc for cc, rmin in col_min_row.items() if rmin < base_row)

    return {
        "bbox": (r0, c0, r1, c1),
        "pts": piece,
        "base_row": base_row,
        "proj_cols": proj_cols,
        "nproj": len(proj_cols),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble several grey bar fragments into one grey frame, matching each fragment to the next destination columns it owns.

    Input structure:
    - The grey structural color forms one large open frame on the right plus 2-3 smaller source fragments on the left.
    - Each fragment has a short base and 1 or 2 upward projections; together the fragments account for the four colored target columns inside the frame.
    - The frame interior still contains one colored stack in each destination column, using the same fixed slot layout as the seed task.

    Rule:
    - Keep the frame, erase the source staging area, and process the source fragments in visible top-to-bottom order.
    - Move every fragment upward so its base sits one cell above the frame's interior floor, then align that fragment's 1 or 2 projection columns to the next 1 or 2 colored destination columns from left to right.
    - After all fragments are inserted, restack each destination column's original colors immediately above the grey support now occupying that column.

    Key insight:
    - This is a small assembly puzzle rather than one global translation: each disconnected fragment claims a consecutive block of destination columns, and the fragment's projection count tells you how large that block is.
    """
    g = np.asarray(grid, dtype=int)

    bg = _bg_color(g)
    other = g[g != bg]
    if other.size == 0:
        return g.copy()
    grey = _most_common(other)

    grey_comps = [(len(pts), _bbox(pts), pts) for pts in _cc4(g == grey)]
    grey_comps.sort(key=lambda t: (-t[0], t[1]))
    if len(grey_comps) < 3:
        return g.copy()

    (_, slot_bbox, _), piece_comps = grey_comps[0], grey_comps[1:]
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

    pieces = [_piece_info(pts) for _, _, pts in piece_comps]
    pieces = [info for info in pieces if int(info["nproj"]) > 0]
    pieces.sort(key=lambda info: (info["bbox"][0], info["bbox"][1]))
    if sum(int(info["nproj"]) for info in pieces) != len(dst_cols):
        return g.copy()

    target_base_row = sr1 - 2
    placed: list[tuple[int, int]] = []
    cursor = 0

    for info in pieces:
        proj_cols = list(info["proj_cols"])
        nproj = int(info["nproj"])
        target_cols = dst_cols[cursor : cursor + nproj]
        if len(target_cols) != nproj:
            return g.copy()
        dx = int(target_cols[0]) - int(proj_cols[0])
        dy = target_base_row - int(info["base_row"])

        for r, c in info["pts"]:
            rr = int(r) + dy
            cc = int(c) + dx
            if sr0 <= rr <= sr1 and sc0 <= cc <= sc1:
                out[rr, cc] = grey
                placed.append((rr, cc))
        cursor += nproj

    placed_by_col: dict[int, list[int]] = {}
    for rr, cc in placed:
        placed_by_col.setdefault(int(cc), []).append(int(rr))

    interior_top = sr0 + 1
    interior_bottom = sr1 - 1

    for col in dst_cols:
        if col not in placed_by_col:
            continue
        support_top = min(placed_by_col[col])
        desired_bottom = support_top - 1
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

# (base_row, left_col, max_height) for the source staging area.
SOURCE_SLOTS = [
    (2, 0, 3),
    (5, 5, 4),
    (7, 0, 3),
]

PARTITIONS = [
    (2, 2),
    (1, 1, 2),
    (1, 2, 1),
    (2, 1, 1),
]

MAX_ATTEMPTS = 400


def _frame_patch() -> np.ndarray:
    patch = np.zeros((FRAME_H, FRAME_W), dtype=int)
    patch[[0, -1], :] = GREY
    patch[:, -1] = GREY
    patch[:, 0] = GREY
    patch[list(LEFT_WALL_OPEN_ROWS), 0] = BG
    return patch


def _partition_to_groups(partition: tuple[int, ...]) -> list[list[int]]:
    groups: list[list[int]] = []
    cursor = 0
    for size in partition:
        groups.append(list(range(cursor, cursor + size)))
        cursor += size
    return groups


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


def _sample_params(
    rng: np.random.Generator,
) -> tuple[list[list[int]], list[int], list[int], list[int]]:
    partition = PARTITIONS[int(rng.integers(0, len(PARTITIONS)))]
    groups = _partition_to_groups(partition)

    colors = [int(v) for v in rng.choice(PALETTE, size=len(SLOT_COLS), replace=False)]
    colored_counts = [int(rng.integers(1, length + 1)) for length in SLOT_LENS]

    slot_ids = [0, 2] if len(groups) == 2 else [0, 1, 2]
    heights: list[int] = []
    for group, slot_id in zip(groups, slot_ids):
        _base_row, _left_col, max_height = SOURCE_SLOTS[slot_id]
        for _ in group:
            heights.append(int(rng.integers(2, max_height + 1)))

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
