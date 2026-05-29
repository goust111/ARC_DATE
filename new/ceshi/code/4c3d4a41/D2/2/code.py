# Variant of 4c3d4a41
# Dimension: D2 — 源对象：杆的结构 (src)
# Change: 灰色水平底座（7格宽）带4个等间距向上凸起（位于奇数列），凸起高度1-4各自独立... → 多段分离的杆片段

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg as _bg_color
from arc_utils import colors_present, get_bg, paste


# Design analysis for the D2 variant
# ----------------------------------
# a) Concrete behavior:
#    - The frame and the four colored slot columns stay exactly as in the seed task.
#    - The source bar is no longer one 7-wide connected object. Instead it is split into
#      2 or 3 disconnected fragments, ordered left-to-right, and those fragments together
#      still account for the same four projection columns.
#    - Allowed source partitions are contiguous groups of projection columns:
#        [2,2], [1,1,2], [1,2,1], [2,1,1]
#      where each number is how many projection columns belong to a fragment.
#    - A 1-projection fragment is a 1-column vertical stub; a 2-projection fragment is a
#      3-column mini-bar whose endpoints are the two projection columns.
#    - Output behavior changes from “translate one whole bar” to “translate each fragment
#      separately into the matching left-to-right region of the frame interior”.
#    - Example trace:
#        original logic: second-largest grey component = full bar -> one dx aligns all 4 keys.
#        new logic: grey components after the frame might be widths [1,3,1]; they map to
#        destination chunks [11], [13,15], [17] and get moved with separate dx values.
#
# b) Complexity calibration:
#    - The partition family is small (4 patterns), so the rule is still learnable from a
#      few examples, but the input is visibly different because the source now has multiple
#      disconnected grey objects that must be grouped and ordered.
#    - The mapping is unambiguous because fragments preserve left-to-right order and each
#      fragment size directly implies whether it owns 1 or 2 destination columns.
#
# c) Old assumptions that break:
#    - Seed solver assumed the bar is exactly the second-largest grey component.
#      This fails once the bar becomes multiple components.
#    - Seed solver computed one global horizontal shift; that no longer works because each
#      fragment may need a different dx.
#    - Seed solver inferred projection columns only from cells above the base row. That is
#      harmless for a single bar, but a 1-column fragment could be missed if treated the same
#      way. In this variant the fragment width (1 or 3) is the primary cue.
#
# d) Placement legality:
#    - Input fragments are placed in cols 0..6 with one empty column between neighbors and
#      at least two empty columns before the frame wall at col 9, so they never merge into
#      the frame component.
#    - Output fragments occupy disjoint frame-interior regions determined by the slot columns,
#      so they never overlap each other or the frame border.
#
# e) Visual observability:
#    - The variant is visible immediately: the input has 2-3 grey source fragments instead of
#      one continuous bar, and the output preserves that fragmentation inside the frame.
#    - No hidden color-id logic is used; only geometry and left-to-right order matter.
#
# f) Visual cleanliness:
#    - Fragments remain on a single base row like the seed task, with fixed one-cell gaps.
#      This keeps the left side readable and avoids clutter.
#
# g) Color reasoning:
#    - Colors are unchanged relative to the seed task: each slot column keeps its own color and
#      is stacked above the moved grey support in the same column. The variant adds structural
#      complexity, not a new color mapping problem.
#
# h) Generator guarantees for solver assumptions:
#    - The frame is the largest grey component.
#    - Every non-frame grey component is a valid fragment of width 1 or 3.
#    - Left-to-right fragment order matches the destination chunk order.
#    - The fragment sizes always sum to four total projection columns.


def _bbox(pts):
    return bbox_coords(pts if isinstance(pts, np.ndarray) else np.asarray(list(pts), dtype=int))


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _most_common(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _fragment_info(pts: np.ndarray) -> dict[str, object]:
    pts = np.asarray(pts, dtype=int)
    r0, c0, r1, c1 = _bbox(pts)
    base_row = int(pts[:, 0].max())
    cols = sorted({int(c) for _, c in pts})
    width = c1 - c0 + 1
    if width == 1:
        proj_cols = [c0]
    elif width == 3 and cols == [c0, c0 + 1, c1]:
        proj_cols = [c0, c1]
    else:
        col_min_row = {}
        for r, c in pts:
            col_min_row[int(c)] = min(col_min_row.get(int(c), int(r)), int(r))
        stems = sorted(c for c, rmin in col_min_row.items() if rmin < base_row)
        if len(stems) >= 2:
            proj_cols = [stems[0], stems[-1]]
        elif stems:
            proj_cols = stems
        else:
            proj_cols = [c0]
    return {
        "bbox": (r0, c0, r1, c1),
        "pts": pts,
        "base_row": base_row,
        "proj_cols": proj_cols,
        "nproj": len(proj_cols),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move several grey bar fragments into a grey frame, matching each fragment to its own colored-column region.

    Input structure:
    - The structural grey color forms one large frame on the right and 2-3 smaller source fragments on the left.
    - Each fragment contains 1 or 2 upward supports, and the frame interior contains four colored columns that keep their seed-task layout.

    Rule:
    - Treat the left-side grey components as an ordered sequence of fragments rather than one bar.
    - Move every fragment upward into the frame so its base sits one cell above the interior floor, and align the fragment's support columns with the next 1 or 2 colored target columns from left to right.
    - After the fragments are placed, restack each column's original colors immediately above the grey support now occupying that column.

    Key insight:
    - The destination still has four fixed support columns, but the source support shape is distributed across multiple disconnected pieces; solve by grouping fragments by geometry and preserving their left-to-right order.
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

    (_, slot_bbox, _), fragment_comps = grey_comps[0], grey_comps[1:]
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

    fragments = [_fragment_info(pts) for _, _, pts in fragment_comps]
    fragments.sort(key=lambda info: (info["bbox"][1], info["bbox"][0]))
    if sum(int(info["nproj"]) for info in fragments) != len(dst_cols):
        return g.copy()

    slot_interior_bottom = sr1 - 1
    target_base_row = slot_interior_bottom - 1

    placed = []
    cursor = 0
    for info in fragments:
        nproj = int(info["nproj"])
        chunk = dst_cols[cursor : cursor + nproj]
        if len(chunk) != nproj:
            return g.copy()
        dx = chunk[0] - int(info["proj_cols"][0])
        dy = target_base_row - int(info["base_row"])
        for r, c in info["pts"]:
            rr = int(r) + dy
            cc = int(c) + dx
            if sr0 <= rr <= sr1 and sc0 <= cc <= sc1:
                out[rr, cc] = grey
                placed.append((rr, cc))
        cursor += nproj

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

        max_len = desired_bottom - interior_top + 1
        if len(colors) > max_len:
            colors = colors[-max_len:]

        start_row = max(desired_bottom - len(colors) + 1, interior_top)
        for i, colv in enumerate(colors):
            rr = start_row + i
            if interior_top <= rr <= interior_bottom:
                out[rr, col] = colv

    return out


# --------------- generator (variant D2) ---------------

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

BAR_R0 = 2
BAR_H = 4
INPUT_START_C0 = 0
INPUT_FRAGMENT_GAP = 1

GROUP_SIZE_PATTERNS = [
    (2, 2),
    (1, 1, 2),
    (1, 2, 1),
    (2, 1, 1),
]

MAX_ATTEMPTS = 300


def _frame_patch() -> np.ndarray:
    patch = np.zeros((FRAME_H, FRAME_W), dtype=int)
    patch[[0, -1], :] = GREY
    patch[:, -1] = GREY
    patch[:, 0] = GREY
    patch[list(LEFT_WALL_OPEN_ROWS), 0] = BG
    return patch


def _groups_from_sizes(sizes: tuple[int, ...]) -> list[list[int]]:
    groups = []
    cursor = 0
    for size in sizes:
        groups.append(list(range(cursor, cursor + size)))
        cursor += size
    return groups


def _fragment_patch(group_heights: list[int]) -> np.ndarray:
    if len(group_heights) == 1:
        patch = np.zeros((BAR_H, 1), dtype=int)
        patch[-1, 0] = GREY
        h0 = int(group_heights[0])
        patch[BAR_H - h0 : BAR_H, 0] = GREY
        return patch
    patch = np.zeros((BAR_H, 3), dtype=int)
    patch[-1, :] = GREY
    h0, h1 = map(int, group_heights)
    patch[BAR_H - h0 : BAR_H, 0] = GREY
    patch[BAR_H - h1 : BAR_H, 2] = GREY
    return patch


def _input_fragment_starts(sizes: tuple[int, ...]) -> list[int]:
    widths = [1 if size == 1 else 3 for size in sizes]
    starts = []
    cursor = INPUT_START_C0
    for width in widths:
        starts.append(cursor)
        cursor += width + INPUT_FRAGMENT_GAP
    return starts


def _output_fragment_starts(groups: list[list[int]]) -> list[int]:
    return [SLOT_COLS[group[0]] for group in groups]


def _paint_slot_column(grid: np.ndarray, col: int, length: int, color: int,
                       count: int, height: int | None) -> None:
    cap = length - count
    grid[SLOT_TOP_ROW : SLOT_TOP_ROW + cap, col] = GREY
    if height is None:
        grid[SLOT_TOP_ROW + cap : SLOT_TOP_ROW + length, col] = color
        return
    visible = min(count, 5 - height)
    if visible > 0:
        row_end = 5 - height
        grid[row_end - visible + 1 : row_end + 1, col] = color


def _build_grid(heights: list[int], colors: list[int], colored_counts: list[int],
                sizes: tuple[int, ...], output_mode: bool) -> np.ndarray:
    groups = _groups_from_sizes(sizes)
    starts = _output_fragment_starts(groups) if output_mode else _input_fragment_starts(sizes)

    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)

    for group, start in zip(groups, starts):
        group_heights = [heights[i] for i in group]
        paste(grid, _fragment_patch(group_heights), BAR_R0, start)

    for col, length, color, count, height in zip(SLOT_COLS, SLOT_LENS, colors, colored_counts, heights):
        _paint_slot_column(grid, col, length, color, count, height if output_mode else None)
    return grid


def _sample_params(rng: np.random.Generator) -> tuple[tuple[int, ...], list[int], list[int], list[int]]:
    sizes = GROUP_SIZE_PATTERNS[int(rng.integers(len(GROUP_SIZE_PATTERNS)))]
    groups = _groups_from_sizes(sizes)
    colors = [int(v) for v in rng.choice(PALETTE, size=len(SLOT_COLS), replace=False)]
    colored_counts = [int(rng.integers(1, L + 1)) for L in SLOT_LENS]

    while True:
        heights = [int(rng.integers(1, 5)) for _ in SLOT_COLS]
        if all(max(heights[i] for i in group) > 1 for group in groups):
            return sizes, heights, colors, colored_counts


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    sizes, heights, colors, colored_counts = _sample_params(rng)
    grid = _build_grid(heights, colors, colored_counts, sizes, output_mode=False)
    expected = _build_grid(heights, colors, colored_counts, sizes, output_mode=True)

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
