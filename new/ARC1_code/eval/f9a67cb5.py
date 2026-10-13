import numpy as np

from arc_utils import colors_present, get_bg, in_bounds
import random
from itertools import combinations


Coord = tuple[int, int]


def _edge_direction(pos: Coord, shape: tuple[int, int]) -> Coord:
    r, c = pos
    h, w = shape
    directions: list[Coord] = []
    if r == 0:
        directions.append((1, 0))
    if r == h - 1:
        directions.append((-1, 0))
    if c == 0:
        directions.append((0, 1))
    if c == w - 1:
        directions.append((0, -1))
    assert len(directions) == 1
    return directions[0]


def _roles(g: np.ndarray) -> tuple[int, int]:
    bg = get_bg(g)
    foreground = colors_present(g, exclude={bg})
    marker_colors = [color for color in foreground if np.count_nonzero(g == color) == 1]
    assert len(marker_colors) == 1
    marker = marker_colors[0]
    wall_colors = [color for color in foreground if color != marker]
    assert len(wall_colors) == 1
    return marker, wall_colors[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Trace the edge marker inward through broken wall segments, branching at each segment opening.

    Input structure
    A mostly background grid contains one single-cell marker on exactly one edge
    and one other foreground color forming parallel, broken wall segments.

    Rule
    Move the marker inward from its edge. Paint straight through open cells; when
    the next cell is a wall, paint the near side of that contiguous wall segment
    plus its adjacent openings, then continue inward from every in-bounds opening.

    Key insight
    The output is not a flood fill: each obstacle is handled as the single wall
    segment directly hit by the current ray, whose endpoints determine the next
    branch positions.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    marker, wall = _roles(g)

    marker_cells = np.argwhere(g == marker)
    assert len(marker_cells) == 1
    start = tuple(map(int, marker_cells[0]))
    dr, dc = _edge_direction(start, g.shape)

    out = g.copy()
    stack: list[Coord] = [start]
    seen_states: set[Coord] = set()

    while stack:
        r, c = stack.pop()
        if (r, c) in seen_states:
            continue
        seen_states.add((r, c))
        assert in_bounds(g.shape, r, c)
        assert g[r, c] != wall
        out[r, c] = marker

        cr, cc = r, c
        while True:
            nr, nc = cr + dr, cc + dc
            if not in_bounds(g.shape, nr, nc):
                break
            if g[nr, nc] != wall:
                out[nr, nc] = marker
                cr, cc = nr, nc
                continue

            if dr:
                wall_r = nr
                lo = hi = nc
                while lo - 1 >= 0 and g[wall_r, lo - 1] == wall:
                    lo -= 1
                while hi + 1 < w and g[wall_r, hi + 1] == wall:
                    hi += 1

                for x in range(max(0, lo - 1), min(w - 1, hi + 1) + 1):
                    out[cr, x] = marker
                for x in (lo - 1, hi + 1):
                    if in_bounds(g.shape, wall_r, x):
                        assert g[wall_r, x] != wall
                        out[wall_r, x] = marker
                        stack.append((wall_r, x))
            else:
                wall_c = nc
                lo = hi = nr
                while lo - 1 >= 0 and g[lo - 1, wall_c] == wall:
                    lo -= 1
                while hi + 1 < h and g[hi + 1, wall_c] == wall:
                    hi += 1

                for y in range(max(0, lo - 1), min(h - 1, hi + 1) + 1):
                    out[y, cc] = marker
                for y in (lo - 1, hi + 1):
                    if in_bounds(g.shape, y, wall_c):
                        assert g[y, wall_c] != wall
                        out[y, wall_c] = marker
                        stack.append((y, wall_c))
            break

    return out

# --------------- generator (o5x) ---------------

# Canvas, palette, and edge choices.
BG_COLOR, MARKER_COLOR, WALL_COLOR = 0, 2, 8
ROW_BOUNDS, COL_BOUNDS = (10, 15), (12, 17)
EDGES = ("top", "bottom", "left", "right")
ROW_LINE_EDGES = ("top", "bottom")

# Wall layout ranges.
LINE_COUNTS = (3, 4)
FIRST_LINE_DISTS, LINE_SPACINGS, FAR_LINE_DISTS = range(2, 4), range(2, 5), range(2, 5)
GAP_COUNT_BOUNDS = (1, 3)
MARKER_BAND = (0.30, 0.50)

# Acceptance and retry limits.
HIT_COUNT_BOUNDS, DEPTH_BOUNDS = (3, 9), (2, 4)
MIN_SEEN_STATES, MAX_GENERATION_ATTEMPTS = 3, 300


def _gen_axis_plans(axis_len, line_count):
    plans = []
    for first_dist in FIRST_LINE_DISTS:
        spacings = [[]]
        for _ in range(line_count - 1):
            spacings = [prev + [gap] for prev in spacings for gap in LINE_SPACINGS]
        for line_spacings in spacings:
            for far_dist in FAR_LINE_DISTS:
                if 1 + first_dist + sum(line_spacings) + far_dist == axis_len:
                    dists = [first_dist]
                    for line_gap in line_spacings:
                        dists.append(dists[-1] + line_gap)
                    plans.append(dists)
    return plans


def _gen_trace_metrics(g):
    h, w = g.shape
    marker_cells = np.argwhere(g == MARKER_COLOR)
    if len(marker_cells) != 1:
        return None
    r, c = map(int, marker_cells[0])
    directions = []
    if r == 0:
        directions.append((1, 0))
    if r == h - 1:
        directions.append((-1, 0))
    if c == 0:
        directions.append((0, 1))
    if c == w - 1:
        directions.append((0, -1))
    if len(directions) != 1:
        return None
    dr, dc = directions[0]

    stack = [((r, c), 0)]
    seen = set()
    hit_count = 0
    max_depth = 0
    while stack:
        (sr, sc), depth = stack.pop()
        if (sr, sc) in seen:
            continue
        seen.add((sr, sc))
        cr, cc = sr, sc
        while True:
            nr, nc = cr + dr, cc + dc
            if not (0 <= nr < h and 0 <= nc < w):
                max_depth = max(max_depth, depth)
                break
            if g[nr, nc] != WALL_COLOR:
                cr, cc = nr, nc
                continue

            next_depth = depth + 1
            hit_count += 1
            max_depth = max(max_depth, next_depth)
            if dr:
                lo = hi = nc
                while lo - 1 >= 0 and g[nr, lo - 1] == WALL_COLOR:
                    lo -= 1
                while hi + 1 < w and g[nr, hi + 1] == WALL_COLOR:
                    hi += 1
                for x in (lo - 1, hi + 1):
                    if 0 <= x < w and g[nr, x] != WALL_COLOR:
                        stack.append(((nr, x), next_depth))
            else:
                lo = hi = nr
                while lo - 1 >= 0 and g[lo - 1, nc] == WALL_COLOR:
                    lo -= 1
                while hi + 1 < h and g[hi + 1, nc] == WALL_COLOR:
                    hi += 1
                for y in (lo - 1, hi + 1):
                    if 0 <= y < h and g[y, nc] != WALL_COLOR:
                        stack.append(((y, nc), next_depth))
            break
    return hit_count, max_depth, len(seen)


def _gen_observed_band_positions(span_len):
    lo = max(1, int(np.floor((span_len - 1) * MARKER_BAND[0])))
    hi = min(span_len - 2, int(np.ceil((span_len - 1) * MARKER_BAND[1])))
    if lo <= hi:
        return list(range(lo, hi + 1))
    return list(range(1, span_len - 1))


def _gen_gap_positions(span_len, gap_count):
    choices = [
        combo
        for combo in combinations(range(1, span_len - 1), gap_count)
        if all(b - a > 1 for a, b in zip(combo, combo[1:]))
    ]
    if not choices:
        return None
    return list(random.choice(choices))


def _gen_choose_distances(axis_len):
    valid_counts = [count for count in LINE_COUNTS if _gen_axis_plans(axis_len, count)]
    if not valid_counts:
        return None
    line_count = random.choice(valid_counts)
    return random.choice(_gen_axis_plans(axis_len, line_count))


def _gen_draw_walls(g, edge, line_positions, span_len):
    first_gaps = None
    for i, line_pos in enumerate(line_positions):
        gap_count = random.randint(*GAP_COUNT_BOUNDS)
        gaps = _gen_gap_positions(span_len, gap_count)
        if gaps is None:
            return None
        if edge in ROW_LINE_EDGES:
            g[line_pos, :] = WALL_COLOR
            g[line_pos, gaps] = BG_COLOR
        else:
            g[:, line_pos] = WALL_COLOR
            g[gaps, line_pos] = BG_COLOR
        if i == 0:
            first_gaps = set(gaps)
    return first_gaps


def _gen_place_marker(g, edge, span_len, first_gaps):
    choices = [p for p in _gen_observed_band_positions(span_len) if p not in first_gaps]
    if not choices:
        choices = [p for p in range(1, span_len - 1) if p not in first_gaps]
    if not choices:
        return False

    h, w = g.shape
    offset = random.choice(choices)
    if edge == "top":
        g[0, offset] = MARKER_COLOR
    elif edge == "bottom":
        g[h - 1, offset] = MARKER_COLOR
    elif edge == "left":
        g[offset, 0] = MARKER_COLOR
    else:
        g[offset, w - 1] = MARKER_COLOR
    return True


def _gen_is_valid(g):
    metrics = _gen_trace_metrics(g)
    if metrics is None:
        return False
    hit_count, max_depth, seen_count = metrics
    if not (
        HIT_COUNT_BOUNDS[0] <= hit_count <= HIT_COUNT_BOUNDS[1]
        and DEPTH_BOUNDS[0] <= max_depth <= DEPTH_BOUNDS[1]
        and seen_count >= MIN_SEEN_STATES
    ):
        return False

    try:
        main(g)
    except Exception:
        return False
    return True


def _gen_attempt():
    h = random.randint(*ROW_BOUNDS)
    w = random.randint(*COL_BOUNDS)
    edge = random.choice(EDGES)
    axis_len = h if edge in ROW_LINE_EDGES else w
    span_len = w if edge in ROW_LINE_EDGES else h
    distances = _gen_choose_distances(axis_len)
    if distances is None:
        return None

    g = np.zeros((h, w), dtype=int)
    if edge in ("top", "left"):
        line_positions = distances
    else:
        line_positions = [axis_len - 1 - d for d in distances]
    first_gaps = _gen_draw_walls(g, edge, line_positions, span_len)
    if first_gaps is None or not _gen_place_marker(g, edge, span_len, first_gaps):
        return None
    if not _gen_is_valid(g):
        return None
    return g


def generate_input():
    for _ in range(MAX_GENERATION_ATTEMPTS):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("failed to generate valid f9a67cb5 input")
