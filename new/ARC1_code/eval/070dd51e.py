import numpy as np

from arc_utils import colors_present, draw_line, get_bg
import itertools
import random


def _segment_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    if r0 == r1:
        c_start, c_end = sorted((c0, c1))
        return {(r0, c) for c in range(c_start, c_end + 1)}
    r_start, r_end = sorted((r0, r1))
    return {(r, c0) for r in range(r_start, r_end + 1)}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color pair of marker pixels with a straight segment, drawing vertical segments over horizontal ones.

    Input structure
    - The background is the dominant color.
    - Each foreground color appears in exactly two marker pixels.
    - The two markers of each color share exactly one row or exactly one column.

    Rule
    - Draw the same-color horizontal or vertical segment between each marker pair, inclusive.
    - Paint horizontal segments first, then vertical segments so crossings keep the vertical segment color.

    Key insight
    - The colored pixels are not isolated objects; same-color pairs are endpoints of axis-aligned segments, and segment orientation determines the crossing layer.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    horizontal_segments = []
    vertical_segments = []
    occupied_by_orientation = {"horizontal": set(), "vertical": set()}
    for color in colors_present(g, exclude={bg}):
        coords = np.argwhere(g == color)
        assert coords.shape == (2, 2)
        (r0, c0), (r1, c1) = coords.tolist()
        assert (r0 == r1) ^ (c0 == c1)

        segment = (int(color), int(r0), int(c0), int(r1), int(c1))
        if r0 == r1:
            cells = _segment_cells(r0, c0, r1, c1)
            assert occupied_by_orientation["horizontal"].isdisjoint(cells)
            occupied_by_orientation["horizontal"].update(cells)
            horizontal_segments.append(segment)
        else:
            cells = _segment_cells(r0, c0, r1, c1)
            assert occupied_by_orientation["vertical"].isdisjoint(cells)
            occupied_by_orientation["vertical"].update(cells)
            vertical_segments.append(segment)

    for color, r0, c0, r1, c1 in horizontal_segments:
        draw_line(out, r0, c0, end_r=r1, end_c=c1, color=color)
    for color, r0, c0, r1, c1 in vertical_segments:
        draw_line(out, r0, c0, end_r=r1, end_c=c1, color=color)

    return out

# --------------- generator (o5x) ---------------

# Canvas, palette, and retry budget.
GRID_HEIGHT_RANGE, GRID_WIDTH_RANGE = (20, 30), (10, 20)
COLOR_POOL = range(1, 10)
GENERATION_RETRIES = 300

# Segment structure.
PAIR_COUNT = 5
HORIZONTAL_PAIR_RANGE, TARGET_CROSSING_RANGE = (2, 3), (2, 3)
MIN_SPAN_LEN, MAX_SPAN_LEN = 4, 18
POSITION_MARGIN = 1


def _gen_spaced_positions(size: int, count: int) -> list[int]:
    choices = list(range(POSITION_MARGIN, size - POSITION_MARGIN))
    buckets = np.array_split(choices, count)
    vals = [int(random.choice(bucket.tolist())) for bucket in buckets if len(bucket)]
    random.shuffle(vals)
    return vals


def _gen_span_options(limit: int, include: list[int], exclude: list[int]) -> dict[int, list[tuple[int, int]]]:
    options: dict[int, list[tuple[int, int]]] = {}
    upper = min(int(limit), MAX_SPAN_LEN)
    for length in range(MIN_SPAN_LEN, upper + 1):
        spans = []
        for start in range(0, int(limit) - length + 1):
            end = start + length - 1
            if all(start <= x <= end for x in include) and all(
                not (start <= x <= end) for x in exclude
            ):
                spans.append((start, end))
        if spans:
            options[length] = spans
    return options


def _gen_choose_span(options: dict[int, list[tuple[int, int]]]) -> tuple[int, int]:
    length = random.choice(list(options))
    return random.choice(options[length])


def _axis_span_options(primary_count, secondary_count, secondary_positions, target, limit, *, flipped=False):
    options = []
    for primary_i in range(primary_count):
        include = []
        exclude = []
        for secondary_i in range(secondary_count):
            pair = (secondary_i, primary_i) if flipped else (primary_i, secondary_i)
            (include if pair in target else exclude).append(secondary_positions[secondary_i])
        opts = _gen_span_options(limit, include, exclude)
        if not opts:
            return None
        options.append(opts)
    return options


def _choose_crossing_plan(height, width, h_rows, v_cols, h_count, v_count, target_count):
    all_pairs = [(h_i, v_i) for h_i in range(h_count) for v_i in range(v_count)]
    valid_targets = []
    for combo in itertools.combinations(all_pairs, target_count):
        target = set(combo)
        h_options = _axis_span_options(h_count, v_count, v_cols, target, width)
        if h_options is None:
            continue
        v_options = _axis_span_options(v_count, h_count, h_rows, target, height, flipped=True)
        if v_options is not None:
            valid_targets.append((target, h_options, v_options))
    if not valid_targets:
        return None
    return random.choice(valid_targets)


def _build_segments(colors, h_count, h_rows, v_cols, h_options, v_options):
    horizontal = []
    vertical = []
    for h_i, opts in enumerate(h_options):
        c0, c1 = _gen_choose_span(opts)
        horizontal.append((colors[h_i], h_rows[h_i], c0, c1))
    for v_i, opts in enumerate(v_options):
        r0, r1 = _gen_choose_span(opts)
        vertical.append((colors[h_count + v_i], v_cols[v_i], r0, r1))
    return horizontal, vertical


def _crossing_set(horizontal, vertical):
    return {
        (h_i, v_i)
        for h_i, (_, row, c0, c1) in enumerate(horizontal)
        for v_i, (_, col, r0, r1) in enumerate(vertical)
        if c0 <= col <= c1 and r0 <= row <= r1
    }


def _draw_solution(height, width, horizontal, vertical):
    out = np.zeros((height, width), dtype=int)
    for color, row, c0, c1 in horizontal:
        draw_line(out, row, c0, end_r=row, end_c=c1, color=color)
    for color, col, r0, r1 in vertical:
        draw_line(out, r0, col, end_r=r1, end_c=col, color=color)
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    colors = random.sample(COLOR_POOL, PAIR_COUNT)
    h_count = random.randint(*HORIZONTAL_PAIR_RANGE)
    v_count = PAIR_COUNT - h_count

    h_rows = _gen_spaced_positions(height, h_count)
    v_cols = _gen_spaced_positions(width, v_count)
    target_count = random.randint(*TARGET_CROSSING_RANGE)

    plan = _choose_crossing_plan(height, width, h_rows, v_cols, h_count, v_count, target_count)
    if plan is None:
        return None
    target, h_options, v_options = plan

    horizontal, vertical = _build_segments(colors, h_count, h_rows, v_cols, h_options, v_options)

    if _crossing_set(horizontal, vertical) != target:
        return None
    endpoints = []
    for color, row, c0, c1 in horizontal:
        endpoints.extend([(row, c0, color), (row, c1, color)])
    for color, col, r0, r1 in vertical:
        endpoints.extend([(r0, col, color), (r1, col, color)])
    if len({(r, c) for r, c, _ in endpoints}) != len(endpoints):
        return None

    out = _draw_solution(height, width, horizontal, vertical)
    inp = np.zeros((height, width), dtype=int)
    for r, c, color in endpoints:
        inp[r, c] = color

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 070dd51e input")
