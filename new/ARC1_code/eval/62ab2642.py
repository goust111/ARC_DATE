import numpy as np

from arc_utils import Color, components_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the smallest black region orange and the largest black region teal.

    Input structure:
    - The grid contains only black background regions and grey one-cell-wide walls.
    - The grey walls partition the black pixels into multiple 4-connected regions.
    - The smallest and largest black regions are each unique.

    Rule:
    - Find all 4-connected black regions.
    - Recolor every pixel of the unique smallest black region orange.
    - Recolor every pixel of the unique largest black region teal.
    - Preserve all grey pixels and all other black regions.

    Key insight:
    - The colors mark extreme region size among the black cells; the region location
      and shape do not otherwise determine the output.
    """
    g = np.asarray(grid, dtype=int)
    colors = set(map(int, g.ravel()))
    assert Color.BLACK in colors
    assert Color.GREY in colors
    assert colors <= {Color.BLACK, Color.GREY}

    regions = components_mask(g == Color.BLACK, connectivity=4)
    assert len(regions) >= 2

    areas = np.array([int(region.shape[0]) for region in regions], dtype=int)
    small_indices = np.flatnonzero(areas == areas.min())
    large_indices = np.flatnonzero(areas == areas.max())
    assert small_indices.size == 1
    assert large_indices.size == 1
    assert int(small_indices[0]) != int(large_indices[0])

    out = g.copy()
    small_region = regions[int(small_indices[0])]
    large_region = regions[int(large_indices[0])]
    out[small_region[:, 0], small_region[:, 1]] = Color.ORANGE
    out[large_region[:, 0], large_region[:, 1]] = Color.TEAL
    return out

# --------------- generator (o5x) ---------------

# Canvas bounds and admissible total area.
CANVAS_ROW_CHOICES = range(5, 16)
CANVAS_COL_CHOICES = range(5, 13)
CANVAS_AREA_RANGE = (60, 150)

# Required black-region and area structure.
REGION_COUNT_CHOICES = (7, 8)
SMALLEST_REGION_AREA_RANGE, LARGEST_REGION_AREA_RANGE = (1, 4), (10, 23)
MIN_LARGE_SMALL_RATIO = 5

# Grey wall layout and validation limits.
GREY_FRACTION_RANGE = (0.35, 0.44)
SIDE_WIDTH_LIMIT, MAX_TRUNK_SHIFTS, TRUNK_SHIFT_CLEARANCE = 4, 4, 1
BRANCH_STRAIGHT_PROB, BRANCH_MIN_BEND_ROOM = 0.35, 3

# Retry budget for rejection sampling.
GENERATION_RETRIES = 300

CANVAS_DIMS = tuple(
    (h, w)
    for h in CANVAS_ROW_CHOICES
    for w in CANVAS_COL_CHOICES
    if CANVAS_AREA_RANGE[0] <= h * w <= CANVAS_AREA_RANGE[1]
)


def _gen_bounded_composition(total, parts, max_each):
    if parts <= 0 or max_each < 1 or total < parts or total > parts * max_each:
        return None

    vals = []
    remaining = total
    for i in range(parts):
        left = parts - i - 1
        lo = max(1, remaining - left * max_each)
        hi = min(max_each, remaining - left)
        if lo > hi:
            return None
        val = int(np.random.randint(lo, hi + 1))
        vals.append(val)
        remaining -= val

    return [int(v) for v in np.random.permutation(vals)]


def _gen_branch_rows(heights):
    rows = []
    r = 0
    for h in heights[:-1]:
        r += int(h)
        rows.append(r)
        r += 1
    return rows


def _gen_all_black_regions_touch_border(grid, regions):
    h, w = grid.shape
    for region in regions:
        rr = region[:, 0]
        cc = region[:, 1]
        if not (
            np.any(rr == 0)
            or np.any(rr == h - 1)
            or np.any(cc == 0)
            or np.any(cc == w - 1)
        ):
            return False
    return True


def _gen_trunk_spans(length, width, start_col, blocked_rows):
    blocked_rows = set(blocked_rows)
    candidates = [
        r
        for r in range(TRUNK_SHIFT_CLEARANCE, length - 2)
        if all(abs(r - blocked) > TRUNK_SHIFT_CLEARANCE for blocked in blocked_rows)
    ]
    if candidates:
        candidates = [int(r) for r in np.random.permutation(candidates)]

    used = []
    max_shifts = min(MAX_TRUNK_SHIFTS, len(candidates))
    shift_goal = int(np.random.randint(0, max_shifts + 1))
    for r in candidates:
        if len(used) >= shift_goal:
            break
        if all(abs(r - prev) > TRUNK_SHIFT_CLEARANCE for prev in used):
            used.append(r)

    shift_rows = set(used)
    spans = []
    c = int(start_col)
    for r in range(length):
        if r in shift_rows:
            directions = []
            if c > 1:
                directions.append(-1)
            if c < width - 2:
                directions.append(1)
            if directions:
                nc = c + int(directions[int(np.random.randint(0, len(directions)))])
                spans.append((min(c, nc), max(c, nc)))
                c = nc
                continue
        spans.append((c, c))
    return spans


def _gen_branch_row_offset(row, length):
    offsets = [0]
    if row > 0:
        offsets.append(-1)
    if row < length - 1:
        offsets.append(1)
    return int(offsets[int(np.random.randint(0, len(offsets)))])


def _gen_paint_straight_branch(grid, row, anchor, left_side):
    if left_side:
        grid[row, : anchor + 1] = Color.GREY
    else:
        grid[row, anchor:] = Color.GREY


def _gen_paint_branch(grid, spans, row, side):
    length, width = grid.shape
    lo, hi = spans[row]
    left_side = side < 0
    anchor = lo if left_side else hi
    room = anchor if left_side else width - anchor - 1

    if room < BRANCH_MIN_BEND_ROOM or np.random.random() < BRANCH_STRAIGHT_PROB:
        _gen_paint_straight_branch(grid, row, anchor, left_side)
        return

    if left_side:
        turn = int(np.random.randint(1, anchor - 1))
    else:
        turn = int(np.random.randint(anchor + 2, width - 1))

    end_row = row + _gen_branch_row_offset(row, length)
    if end_row == row:
        _gen_paint_straight_branch(grid, row, anchor, left_side)
        return

    if left_side:
        grid[row, turn : anchor + 1] = Color.GREY
        grid[min(row, end_row) : max(row, end_row) + 1, turn] = Color.GREY
        grid[end_row, : turn + 1] = Color.GREY
        return

    grid[row, anchor : turn + 1] = Color.GREY
    grid[min(row, end_row) : max(row, end_row) + 1, turn] = Color.GREY
    grid[end_row, turn:] = Color.GREY


def _gen_oriented_frame():
    h, w = CANVAS_DIMS[int(np.random.randint(0, len(CANVAS_DIMS)))]
    target_regions = int(np.random.choice(REGION_COUNT_CHOICES))
    orientations = []
    if target_regions <= 2 * ((h + 1) // 2):
        orientations.append(False)
    if target_regions <= 2 * ((w + 1) // 2):
        orientations.append(True)
    if not orientations:
        return None

    transpose = bool(orientations[int(np.random.randint(0, len(orientations)))])
    length, width = (w, h) if transpose else (h, w)
    return target_regions, transpose, length, width


def _gen_layout_plan(length, width, target_regions):
    max_segments_per_side = (length + 1) // 2
    segment_pairs = [
        (left, target_regions - left)
        for left in range(1, target_regions)
        if left <= max_segments_per_side and target_regions - left <= max_segments_per_side
    ]
    if not segment_pairs:
        return None

    left_segments, right_segments = segment_pairs[
        int(np.random.randint(0, len(segment_pairs)))
    ]
    trunk_choices = []
    for trunk_col in range(1, width - 1):
        left_width = trunk_col
        right_width = width - trunk_col - 1
        if min(left_width, right_width) > SIDE_WIDTH_LIMIT:
            continue

        left_total = length - (left_segments - 1)
        right_total = length - (right_segments - 1)
        if (
            left_total <= left_segments * (LARGEST_REGION_AREA_RANGE[1] // left_width)
            and right_total
            <= right_segments * (LARGEST_REGION_AREA_RANGE[1] // right_width)
        ):
            trunk_choices.append((trunk_col, left_width, right_width))

    if not trunk_choices:
        return None

    trunk_col, left_width, right_width = trunk_choices[
        int(np.random.randint(0, len(trunk_choices)))
    ]

    left_heights = _gen_bounded_composition(
        length - (left_segments - 1),
        left_segments,
        LARGEST_REGION_AREA_RANGE[1] // left_width,
    )
    right_heights = _gen_bounded_composition(
        length - (right_segments - 1),
        right_segments,
        LARGEST_REGION_AREA_RANGE[1] // right_width,
    )
    if left_heights is None or right_heights is None:
        return None

    left_rows = _gen_branch_rows(left_heights)
    right_rows = _gen_branch_rows(right_heights)
    return trunk_col, left_rows, right_rows


def _gen_render_wall_grid(length, width, trunk_col, left_rows, right_rows):
    spans = _gen_trunk_spans(length, width, trunk_col, left_rows + right_rows)

    grid = np.full((length, width), Color.BLACK, dtype=int)
    for r, (lo, hi) in enumerate(spans):
        grid[r, lo : hi + 1] = Color.GREY
    for r in left_rows:
        _gen_paint_branch(grid, spans, r, -1)
    for r in right_rows:
        _gen_paint_branch(grid, spans, r, 1)

    return grid


def _gen_validated_regions(grid, target_regions):
    grey = grid == Color.GREY
    has_2x2_wall = np.any(grey[:-1, :-1] & grey[1:, :-1] & grey[:-1, 1:] & grey[1:, 1:])
    if has_2x2_wall:
        return None

    grey_regions = components_mask(grey, connectivity=4)
    if len(grey_regions) != 1:
        return None

    black_regions = components_mask(grid == Color.BLACK, connectivity=4)
    if len(black_regions) != target_regions:
        return None

    if not _gen_all_black_regions_touch_border(grid, black_regions):
        return None

    grey_fraction = np.count_nonzero(grey) / grid.size
    if not (GREY_FRACTION_RANGE[0] <= grey_fraction <= GREY_FRACTION_RANGE[1]):
        return None

    areas = np.array([int(region.shape[0]) for region in black_regions], dtype=int)
    min_area = int(areas.min())
    max_area = int(areas.max())
    if not (
        SMALLEST_REGION_AREA_RANGE[0] <= min_area <= SMALLEST_REGION_AREA_RANGE[1]
        and LARGEST_REGION_AREA_RANGE[0] <= max_area <= LARGEST_REGION_AREA_RANGE[1]
        and np.count_nonzero(areas == min_area) == 1
        and np.count_nonzero(areas == max_area) == 1
        and min_area != max_area
        and max_area >= MIN_LARGE_SMALL_RATIO * min_area
    ):
        return None

    return black_regions, areas, min_area, max_area


def _gen_attempt() -> np.ndarray | None:
    oriented = _gen_oriented_frame()
    if oriented is None:
        return None

    target_regions, transpose, length, width = oriented
    layout = _gen_layout_plan(length, width, target_regions)
    if layout is None:
        return None

    trunk_col, left_rows, right_rows = layout
    grid = _gen_render_wall_grid(length, width, trunk_col, left_rows, right_rows)
    if transpose:
        grid = grid.T

    region_data = _gen_validated_regions(grid, target_regions)
    if region_data is None:
        return None

    black_regions, areas, min_area, max_area = region_data
    expected = grid.copy()
    small_region = black_regions[int(np.flatnonzero(areas == min_area)[0])]
    large_region = black_regions[int(np.flatnonzero(areas == max_area)[0])]
    expected[small_region[:, 0], small_region[:, 1]] = Color.ORANGE
    expected[large_region[:, 0], large_region[:, 1]] = Color.TEAL

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 62ab2642 input")
