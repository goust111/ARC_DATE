# Combo variant of 4c3d4a41 (index 8)
# combo_spec: D2/2+D3/3
# Name: 多段分离杆+框架带内部隔板

import numpy as np

from arc_utils import bbox_coords, colors_present, components_mask, get_bg as _bg_color
from arc_utils import get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
#    - Preserve the 8x20 canvas, the right-hand grey frame, and the idea that colored
#      columns inside the frame indicate where grey supports should end up.
#    - Fuse the two single variants by changing both sides at once:
#        1) source: the left object is 2 or 3 disconnected fragments rather than one bar;
#        2) destination: the frame interior is split by a horizontal divider into an
#           upper chamber and a lower chamber.
#    - To make the fragment-to-region correspondence visible and unambiguous, the target
#      columns are reassigned as contiguous chamber groups:
#        cols 11 and 13 belong to the upper chamber, cols 15 and 17 belong to the lower.
#    - The only allowed source partitions are [2,2], [1,1,2], [2,1,1]. This matters:
#      every fragment consumes the next 1 or 2 target columns, and no fragment is allowed
#      to straddle the chamber boundary between cols 13 and 15.
#    - Example trace:
#        seed task: one connected 4-support bar -> one dx -> all colors stack in one cavity.
#        combo task: fragments [1,1,2] -> first stub goes to col 11 (upper), second stub to
#        col 13 (upper), last mini-bar to cols 15/17 (lower); then each column keeps only the
#        part of its color stack that fits inside that column's own chamber above that support.
#
# b) Complexity calibration:
#    - The source fragmentation is clearly visible but controlled: 2 or 3 fragments only,
#      each fragment has 1 or 2 supports, and the legal partition family has 3 patterns.
#    - The destination partition is also controlled: one divider row, always the same, and
#      always a 2-column upper chamber followed by a 2-column lower chamber.
#    - Upper supports are fixed tall stems (height 4) so they visibly pierce the divider;
#      lower supports are sampled from {1,2} so they visibly stay below it.
#    - This creates repeated evidence across samples without exploding the state space.
#
# c) Hidden assumptions from the seed solver that break:
#    - "the bar is the second-largest grey component" fails because the source is now
#      several components; the solver must gather and order all non-frame grey fragments.
#    - "one horizontal shift aligns everything" fails because each fragment has its own dx.
#    - "all destination columns share one cavity top/bottom" fails because cols 11/13 and
#      cols 15/17 now belong to different row intervals.
#    - "extract colors from the whole column" fails because upper-chamber columns and
#      lower-chamber columns encode their colors in disjoint vertical shafts.
#
# d) Placement legality:
#    - Input fragments use starts derived from widths [3,3], [1,1,3], or [3,1,1], all on
#      rows 3..6, with one blank column between neighbors and at least two blank columns
#      before the frame wall at col 9, so no input fragment can merge with the frame.
#    - Output fragments map to starts [11,15], [11,13,15], or [11,15,17], so they remain
#      disjoint from each other except through the frame only where the tall upper supports
#      intentionally pass through divider holes.
#
# e) Visual observability:
#    - A human can see both fused changes directly: the source is split into 2-3 pieces,
#      and the destination frame has a divider with left-side holes only.
#    - The first two target columns always behave like an upper chamber, the last two like a
#      lower chamber, so 2-3 examples are enough to infer the chamber split.
#
# f) Visual cleanliness:
#    - Only one divider is added; no extra marker colors are introduced.
#    - The frame interior remains sparse: four isolated shafts, two above and two below.
#      Fragment gaps are fixed, so the left side stays easy to parse.
#
# g) Color reasoning:
#    - Each of the four target columns still has one unique color, exactly as in the seed.
#    - No color permutation or hidden numeric ordering is used. Colors only show how much
#      of each chamber-specific stack survives above the placed support.
#
# h) Fusion-specific non-degeneration guard:
#    - Because lower-chamber colors are restricted to rows 4..6 while upper-chamber colors
#      are restricted to rows 1..3, a solver that ignores the divider would place colors in
#      different rows from the intended output on most samples.
#    - Because the source is fragmented, a solver that expects one bar also fails.
#
# i) Generator guarantees for the new solver assumptions:
#    - The largest grey component is always the frame.
#    - Every remaining grey component is a legal width-1 or width-3 fragment.
#    - Sorted left-to-right fragment order matches the sorted left-to-right target-column
#      order, and the legal partitions ensure no fragment crosses the chamber boundary.
#    - Upper target columns always have non-grey cells on the divider row; lower columns
#      keep that row grey, so the solver can recover the chamber assignment from the image.


BG = 0
GREY = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)

H, W = 8, 20

FRAME_R0, FRAME_C0 = 0, 9
FRAME_H, FRAME_W = 8, 11
LEFT_WALL_OPEN_ROWS = (4, 5)
PARTITION_ROW = 3

SLOT_COLS = [11, 13, 15, 17]
SHAFT_ROWS = {
    11: [1, 2, 3],
    13: [1, 2, 3],
    15: [4, 5, 6],
    17: [4, 5, 6],
}

BAR_R0 = 3
BAR_H = 4
BAR_BASE_ROW = BAR_R0 + BAR_H - 1
INPUT_START_C0 = 0
INPUT_FRAGMENT_GAP = 1

VALID_GROUP_SIZE_PATTERNS = [
    (2, 2),
    (1, 1, 2),
    (2, 1, 1),
]
TWO_FRAGMENT_PATTERNS = [sizes for sizes in VALID_GROUP_SIZE_PATTERNS if len(sizes) == 2]
THREE_FRAGMENT_PATTERNS = [sizes for sizes in VALID_GROUP_SIZE_PATTERNS if len(sizes) == 3]

UPPER_HEIGHT = 4
LOWER_HEIGHTS = (1, 2)
MAX_ATTEMPTS = 300


def _bbox(pts):
    return bbox_coords(pts if isinstance(pts, np.ndarray) else np.asarray(list(pts), dtype=int))


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _most_common(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


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
        patch[BAR_H - int(group_heights[0]) : BAR_H, 0] = GREY
        return patch
    patch = np.zeros((BAR_H, 3), dtype=int)
    patch[-1, :] = GREY
    patch[BAR_H - int(group_heights[0]) : BAR_H, 0] = GREY
    patch[BAR_H - int(group_heights[1]) : BAR_H, 2] = GREY
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


def _frame_patch() -> np.ndarray:
    patch = np.zeros((FRAME_H, FRAME_W), dtype=int)
    patch[[0, -1], :] = GREY
    patch[:, 0] = GREY
    patch[:, -1] = GREY
    patch[list(LEFT_WALL_OPEN_ROWS), 0] = BG
    patch[1:PARTITION_ROW, 3] = GREY  # keep the frame connected between the two upper shafts
    patch[PARTITION_ROW, :] = GREY
    patch[PARTITION_ROW, 2] = BG   # abs col 11
    patch[PARTITION_ROW, 4] = BG   # abs col 13
    return patch


def _paint_input_column(grid: np.ndarray, col: int, color: int, count: int) -> None:
    rows = SHAFT_ROWS[col]
    cap = len(rows) - count
    for i, r in enumerate(rows):
        grid[r, col] = GREY if i < cap else color


def _paint_output_column(grid: np.ndarray, col: int, color: int, count: int, height: int) -> None:
    rows = SHAFT_ROWS[col]
    cap = len(rows) - count
    for i in range(cap):
        grid[rows[i], col] = GREY

    top_row = BAR_BASE_ROW - height + 1
    allowed_rows = [r for r in rows if r < top_row]
    visible = min(count, len(allowed_rows))
    if visible <= 0:
        return
    for r in allowed_rows[-visible:]:
        grid[r, col] = color


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
        proj_cols = sorted(c for c, rmin in col_min_row.items() if rmin < base_row)
        if not proj_cols:
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
    Move several grey fragments into a divided frame, sending each fragment's supports to its chamber-specific target columns.

    Input structure:
    - Grey forms one large frame on the right and 2 or 3 smaller source fragments on the left.
    - Inside the frame, a horizontal grey divider splits the interior into an upper chamber and a lower chamber.
    - Four colored columns mark the targets: the first two columns use the upper chamber, while the last two use the lower chamber.

    Rule:
    - Sort the left-side grey fragments from left to right and map each fragment to the next 1 or 2 target columns from left to right.
    - Translate every fragment so its base sits on the frame floor and its support columns align with those target columns.
    - For each target column, keep only the original colored cells from that column's own chamber, stacking the surviving cells immediately above the placed support within that same chamber.

    Key insight:
    - This is not just "multiple bars" or just "a divided frame"; the fragment grouping and the chamber split must be solved together, because each fragment only owns a local run of target columns and each of those columns is constrained to one vertical region of the frame.
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

    target_base_row = sr1 - 1
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

    interior_cols = list(range(sc0 + 1, sc1))
    divider_row = max(
        range(sr0 + 1, sr1),
        key=lambda r: sum(g[r, c] == grey for c in interior_cols),
    )

    for col in dst_cols:
        if col not in placed_by_col:
            continue

        shaft_rows = (
            list(range(sr0 + 1, divider_row + 1))
            if g[divider_row, col] != grey
            else list(range(divider_row + 1, sr1))
        )
        if not shaft_rows:
            continue

        bar_rows_here = sorted(r for r in placed_by_col[col] if r in shaft_rows)
        if not bar_rows_here:
            continue

        bar_top = bar_rows_here[0]
        allowed_rows = [r for r in shaft_rows if r < bar_top]
        if not allowed_rows:
            continue

        colors = [int(g[r, col]) for r in shaft_rows if g[r, col] != bg and g[r, col] != grey]
        if not colors:
            continue
        if len(colors) > len(allowed_rows):
            colors = colors[-len(allowed_rows) :]

        dest_rows = allowed_rows[-len(colors) :]
        for rr, colv in zip(dest_rows, colors):
            out[rr, col] = colv

    return out


def _build_grid(
    heights: list[int],
    colors: list[int],
    counts: list[int],
    sizes: tuple[int, ...],
    output_mode: bool,
) -> np.ndarray:
    groups = _groups_from_sizes(sizes)
    fragment_starts = _output_fragment_starts(groups) if output_mode else _input_fragment_starts(sizes)

    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)

    for col, color, count, height in zip(SLOT_COLS, colors, counts, heights):
        if output_mode:
            _paint_output_column(grid, col, color, count, height)
        else:
            _paint_input_column(grid, col, color, count)

    for group, start in zip(groups, fragment_starts):
        group_heights = [heights[i] for i in group]
        paste(grid, _fragment_patch(group_heights), BAR_R0, start, transparent=BG)

    return grid


def _sample_params(rng: np.random.Generator) -> tuple[tuple[int, ...], list[int], list[int], list[int]]:
    # Balance branch coverage between the 2-fragment and 3-fragment regimes.
    if int(rng.integers(2)) == 0:
        sizes = TWO_FRAGMENT_PATTERNS[0]
    else:
        sizes = THREE_FRAGMENT_PATTERNS[int(rng.integers(len(THREE_FRAGMENT_PATTERNS)))]
    colors = [int(v) for v in rng.choice(PALETTE, size=len(SLOT_COLS), replace=False)]
    counts = [int(rng.integers(1, 4)) for _ in SLOT_COLS]
    lower = [int(rng.choice(LOWER_HEIGHTS)), int(rng.choice(LOWER_HEIGHTS))]
    if lower == [1, 1]:
        lower[int(rng.integers(2))] = 2
    heights = [
        UPPER_HEIGHT,
        UPPER_HEIGHT,
        lower[0],
        lower[1],
    ]
    forcing_idx = 2 if heights[2] == 2 else 3
    if counts[forcing_idx] == 1:
        counts[forcing_idx] = int(rng.integers(2, 4))
    return sizes, heights, colors, counts


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    sizes, heights, colors, counts = _sample_params(rng)
    grid = _build_grid(heights, colors, counts, sizes, output_mode=False)
    expected = _build_grid(heights, colors, counts, sizes, output_mode=True)

    if get_bg(grid) != BG:
        return None
    if sorted(colors_present(grid, exclude=[BG, GREY])) != sorted(colors):
        return None
    if sorted(colors_present(expected, exclude=[BG, GREY])) != sorted(colors):
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
