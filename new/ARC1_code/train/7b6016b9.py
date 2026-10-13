import numpy as np

from arc_utils import Color, colors_present, components_mask, find_components, flood_fill, get_bg
import random


def _paint_exterior_background(out: np.ndarray, bg: int) -> None:
    height, width = out.shape

    for r in range(height):
        for c in (0, width - 1):
            if int(out[r, c]) == bg:
                flood_fill(out, r, c, Color.GREEN, connectivity=4)

    for c in range(width):
        for r in (0, height - 1):
            if int(out[r, c]) == bg:
                flood_fill(out, r, c, Color.GREEN, connectivity=4)


def main(grid: np.ndarray) -> np.ndarray:
    """Color exterior background green and enclosed background regions red around the line object.

    Input structure
    - The grid has black background and one non-background color.
    - The non-background pixels form one 4-connected orthogonal line object.

    Rule
    - Preserve the colored line object.
    - Treat the line object as a wall: background connected to the grid border is exterior and becomes green.
    - Background not 4-connected to the border is enclosed by the line object and becomes red.

    Key insight
    - This is a wall flood-fill task: flood from the boundary to identify exterior background, then the remaining background cells are holes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    assert bg == Color.BLACK
    assert len(colors_present(g, exclude={bg})) == 1
    assert len(find_components(g, bg=bg, connectivity=4, monochromatic=False)) == 1

    out = g.copy()
    _paint_exterior_background(out, bg)

    enclosed = out == bg
    assert bool(enclosed.any())
    out[enclosed] = Color.RED
    return out

# --------------- generator (o5x) ---------------

# Canvas and palette.
GEN_HEIGHT_RANGE, GEN_WIDTH_RANGE = (21, 25), (21, 25)
GEN_LINE_COLORS = [Color.BLUE, Color.YELLOW, Color.GREY, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]

# Line counts and spacing.
GEN_H_COUNT_CHOICES, GEN_ENCLOSED_COUNT_CHOICES = [4, 5], [2, 3]
GEN_TOTAL_SEGMENT_RANGE, GEN_H_LEN_RANGE, GEN_V_LEN_RANGE = (7, 9), (5, 25), (7, 22)
GEN_GAP_RANGE = (3, 7)
GEN_ROW_MIN_MARGIN, GEN_ROW_MAX_SPAN, GEN_COL_MIN_MARGIN = 2, 18, 5

# Topology and region validation.
GEN_FG_RATIO_RANGE, GEN_INTERSECTION_RANGE, GEN_ENDPOINT_RANGE = (0.18, 0.22), (8, 11), (13, 18)
GEN_BORDER_ENDPOINT_RANGE, GEN_TOUCHED_BORDER_RANGE = (1, 2), (1, 2)
GEN_MAX_DEGREE3 = 1
GEN_ENCLOSED_SIZE_RANGE, GEN_MIN_LARGE_ENCLOSED = (6, 66), 15

# Endpoint and branch construction.
GEN_T_ROW_PROBABILITY = 0.5
GEN_BORDER_CONTACT_COUNT_RANGE = (1, 2)
GEN_ENDPOINT_NAMES = [("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right")]
GEN_SIDE_EXTENSION_RANGE, GEN_T_EXTENSION_RANGE = (3, 9), (4, 9)
GEN_EDGE_EXTENSION_MIN, GEN_VERTICAL_ENDPOINT_MARGIN = 1, 1

# Retry budget.
GEN_RETRY_LIMIT = 300


def _gen_between(value, bounds):
    return bounds[0] <= value <= bounds[1]


def _gen_gap_positions(size, count, min_margin, max_span=None):
    gap_count = count - 1
    low = GEN_GAP_RANGE[0] * gap_count
    high = min(GEN_GAP_RANGE[1] * gap_count, size - 1 - 2 * min_margin)
    if max_span is not None:
        high = min(high, max_span)
    if high < low:
        return None

    total = random.randint(low, high)
    gaps = []
    remaining = total
    for i in range(gap_count):
        left = gap_count - i - 1
        lo = max(GEN_GAP_RANGE[0], remaining - GEN_GAP_RANGE[1] * left)
        hi = min(GEN_GAP_RANGE[1], remaining - GEN_GAP_RANGE[0] * left)
        gap = random.randint(lo, hi)
        gaps.append(gap)
        remaining -= gap

    start_hi = size - 1 - min_margin - total
    if start_hi < min_margin:
        return None
    pos = [random.randint(min_margin, start_hi)]
    for gap in gaps:
        pos.append(pos[-1] + gap)
    return pos


def _gen_vertical_span(height, rows):
    r_min = min(rows)
    r_max = max(rows)
    base_len = r_max - r_min + 1
    up_cap = r_min - GEN_VERTICAL_ENDPOINT_MARGIN
    down_cap = height - 1 - GEN_VERTICAL_ENDPOINT_MARGIN - r_max
    min_ext = max(2 * GEN_VERTICAL_ENDPOINT_MARGIN, GEN_V_LEN_RANGE[0] - base_len)
    max_ext = min(GEN_V_LEN_RANGE[1] - base_len, up_cap + down_cap)
    if up_cap < GEN_VERTICAL_ENDPOINT_MARGIN or down_cap < GEN_VERTICAL_ENDPOINT_MARGIN or max_ext < min_ext:
        return None

    total_ext = random.randint(min_ext, max_ext)
    up_lo = max(GEN_VERTICAL_ENDPOINT_MARGIN, total_ext - down_cap)
    up_hi = min(up_cap, total_ext - GEN_VERTICAL_ENDPOINT_MARGIN)
    if up_lo > up_hi:
        return None
    up_ext = random.randint(up_lo, up_hi)
    return r_min - up_ext, r_max + (total_ext - up_ext)


def _gen_draw_segments(height, width, color, h_segments, v_segments):
    grid = np.zeros((height, width), dtype=int)
    for row, c0, c1 in h_segments:
        grid[row, c0 : c1 + 1] = color
    for col, r0, r1 in v_segments:
        grid[r0 : r1 + 1, col] = color
    return grid


def _gen_degree_counts(grid, color):
    endpoints = 0
    border_endpoints = 0
    degree3 = 0
    degree4 = 0
    touched_borders = set()
    height, width = grid.shape
    for r, c in np.argwhere(grid == color):
        degree = 0
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < height and 0 <= cc < width and grid[rr, cc] == color:
                degree += 1
        if degree == 1:
            endpoints += 1
            if r == 0 or r == height - 1 or c == 0 or c == width - 1:
                border_endpoints += 1
                if r == 0:
                    touched_borders.add("top")
                if r == height - 1:
                    touched_borders.add("bottom")
                if c == 0:
                    touched_borders.add("left")
                if c == width - 1:
                    touched_borders.add("right")
        elif degree == 3:
            degree3 += 1
        elif degree == 4:
            degree4 += 1
    return endpoints, border_endpoints, len(touched_borders), degree3, degree4


def _gen_exterior_map(grid):
    exterior = grid.copy()
    _paint_exterior_background(exterior, Color.BLACK)
    return exterior


def _gen_enclosed_regions(grid):
    exterior = _gen_exterior_map(grid)
    return [len(coords) for coords in components_mask(exterior == Color.BLACK, connectivity=4)]


def _gen_segment_intersections(h_segments, v_segments):
    count = 0
    for row, c0, c1 in h_segments:
        for col, r0, r1 in v_segments:
            if c0 <= col <= c1 and r0 <= row <= r1:
                count += 1
    return count


def _gen_validate_segment_specs(h_segments, v_segments):
    h_lens = [c1 - c0 + 1 for _, c0, c1 in h_segments]
    v_lens = [r1 - r0 + 1 for _, r0, r1 in v_segments]
    return (
        len(h_segments) in GEN_H_COUNT_CHOICES
        and 3 <= len(v_segments) <= 5
        and _gen_between(len(h_segments) + len(v_segments), GEN_TOTAL_SEGMENT_RANGE)
        and not any(not _gen_between(length, GEN_H_LEN_RANGE) for length in h_lens)
        and not any(not _gen_between(length, GEN_V_LEN_RANGE) for length in v_lens)
    )


def _gen_validate_grid_geometry(grid, color):
    fg = grid == color
    ratio = float(fg.sum()) / float(grid.size)
    if not _gen_between(ratio, GEN_FG_RATIO_RANGE):
        return False
    if np.any(
        fg[:-1, :-1].astype(int)
        + fg[1:, :-1].astype(int)
        + fg[:-1, 1:].astype(int)
        + fg[1:, 1:].astype(int)
        == 4
    ):
        return False

    comps = find_components(grid, bg=Color.BLACK, connectivity=4, monochromatic=False)
    return len(comps) == 1


def _gen_validate_graph_topology(grid, color, h_segments, v_segments):
    intersections = _gen_segment_intersections(h_segments, v_segments)
    endpoints, border_endpoints, border_count, degree3, degree4 = _gen_degree_counts(grid, color)
    if not (
        _gen_between(intersections, GEN_INTERSECTION_RANGE)
        and _gen_between(endpoints, GEN_ENDPOINT_RANGE)
        and _gen_between(border_endpoints, GEN_BORDER_ENDPOINT_RANGE)
        and _gen_between(border_count, GEN_TOUCHED_BORDER_RANGE)
    ):
        return False
    if degree3 > GEN_MAX_DEGREE3 or degree3 + degree4 != intersections:
        return False

    enclosed = _gen_enclosed_regions(grid)
    return (
        len(enclosed) in GEN_ENCLOSED_COUNT_CHOICES
        and not any(not _gen_between(size, GEN_ENCLOSED_SIZE_RANGE) for size in enclosed)
        and max(enclosed) >= GEN_MIN_LARGE_ENCLOSED
    )


def _gen_matches_solver(grid):
    try:
        out = main(grid)
    except AssertionError:
        return False
    expected = grid.copy()
    exterior = _gen_exterior_map(grid)
    expected[exterior == Color.BLACK] = Color.RED
    expected[exterior == Color.GREEN] = Color.GREEN
    return np.array_equal(out, expected)


def _gen_validate(grid, color, h_segments, v_segments):
    return (
        _gen_validate_segment_specs(h_segments, v_segments)
        and _gen_validate_grid_geometry(grid, color)
        and _gen_validate_graph_topology(grid, color, h_segments, v_segments)
        and _gen_matches_solver(grid)
    )


def _gen_select_enclosing_rows(rows, cols, enclosed_target):
    eligible_base = []
    for i in range(len(rows) - 1):
        for j in range(i + 1, len(rows)):
            row_gap = rows[j] - rows[i] - 1
            areas = [row_gap * (cols[k + 1] - cols[k] - 1) for k in range(enclosed_target)]
            if all(_gen_between(area, GEN_ENCLOSED_SIZE_RANGE) for area in areas) and max(areas) >= GEN_MIN_LARGE_ENCLOSED:
                eligible_base.append((i, j, max(areas) ** 2))
    if not eligible_base:
        return None

    base_i, base_j, _ = random.choices(
        eligible_base, weights=[item[2] for item in eligible_base], k=1
    )[0]
    top = rows[base_i]
    bottom = rows[base_j]
    extra_rows = [row for row in rows if row not in (top, bottom)]
    return top, bottom, extra_rows


def _gen_outer_h_segments(width, top, bottom, cols):
    contact_count = random.randint(*GEN_BORDER_CONTACT_COUNT_RANGE)
    contacts = set(random.sample(GEN_ENDPOINT_NAMES, contact_count))
    row_name = {top: "top", bottom: "bottom"}
    h_segments = []
    left_col = cols[0]
    right_col = cols[-1]
    for row in (top, bottom):
        name = row_name[row]
        if (name, "left") in contacts:
            c0 = 0
        else:
            c0 = left_col - random.randint(GEN_EDGE_EXTENSION_MIN, left_col - 1)
        if (name, "right") in contacts:
            c1 = width - 1
        else:
            c1 = right_col + random.randint(GEN_EDGE_EXTENSION_MIN, width - 2 - right_col)
        h_segments.append((row, c0, c1))
    return h_segments


def _gen_add_side_rows(width, cols, top, bottom, extra_rows, t_row, h_segments):
    v_count = len(cols)
    assigned_rows = {i: [top, bottom] for i in range(v_count)}
    for row in extra_rows:
        side_i = random.choice([0, v_count - 1])
        col = cols[side_i]
        assigned_rows[side_i].append(row)
        make_t = row == t_row
        if side_i == 0:
            cap = col - 1
            if make_t:
                out = random.randint(GEN_T_EXTENSION_RANGE[0], min(GEN_T_EXTENSION_RANGE[1], cap))
                h_segments.append((row, col - out, col))
            else:
                out = random.randint(GEN_SIDE_EXTENSION_RANGE[0], min(GEN_SIDE_EXTENSION_RANGE[1], cap))
                h_segments.append((row, col - out, col + 1))
        else:
            cap = width - 2 - col
            if make_t:
                out = random.randint(GEN_T_EXTENSION_RANGE[0], min(GEN_T_EXTENSION_RANGE[1], cap))
                h_segments.append((row, col, col + out))
            else:
                out = random.randint(GEN_SIDE_EXTENSION_RANGE[0], min(GEN_SIDE_EXTENSION_RANGE[1], cap))
                h_segments.append((row, col - 1, col + out))
    return assigned_rows


def _gen_vertical_segments(height, cols, assigned_rows):
    v_segments = []
    for i, col in enumerate(cols):
        span = _gen_vertical_span(height, assigned_rows[i])
        if span is None:
            return None
        r0, r1 = span
        v_segments.append((col, r0, r1))
    return v_segments


def _gen_attempt():
    height = random.randint(*GEN_HEIGHT_RANGE)
    width = random.randint(*GEN_WIDTH_RANGE)
    color = random.choice(GEN_LINE_COLORS)
    enclosed_target = random.choice(GEN_ENCLOSED_COUNT_CHOICES)
    h_count = random.choice(GEN_H_COUNT_CHOICES)
    v_count = enclosed_target + 1

    rows = _gen_gap_positions(height, h_count, min_margin=GEN_ROW_MIN_MARGIN, max_span=GEN_ROW_MAX_SPAN)
    cols = _gen_gap_positions(width, v_count, min_margin=GEN_COL_MIN_MARGIN)
    if rows is None or cols is None:
        return None

    enclosing_rows = _gen_select_enclosing_rows(rows, cols, enclosed_target)
    if enclosing_rows is None:
        return None
    top, bottom, extra_rows = enclosing_rows

    t_row = random.choice(extra_rows) if random.random() < GEN_T_ROW_PROBABILITY else None
    h_segments = _gen_outer_h_segments(width, top, bottom, cols)
    assigned_rows = _gen_add_side_rows(width, cols, top, bottom, extra_rows, t_row, h_segments)
    v_segments = _gen_vertical_segments(height, cols, assigned_rows)
    if v_segments is None:
        return None

    grid = _gen_draw_segments(height, width, color, h_segments, v_segments)
    if not _gen_validate(grid, color, h_segments, v_segments):
        return None
    return grid


def generate_input():
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
