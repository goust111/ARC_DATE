import numpy as np

from arc_utils import Color, colors_present, get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each horizontal span enclosed by consecutive outline pixels with red.

    Input structure
    - The grid has one dominant background color.
    - All non-background pixels share one outline color and form diagonal boundary strokes.
    - Rows contain one or more outline pixels; consecutive outline pixels bound horizontal gaps.

    Rule
    - Preserve the original background and outline pixels.
    - In each row, find consecutive outline pixels from left to right.
    - Recolor every background pixel strictly between each consecutive pair to red.

    Key insight
    - The diagonal outline becomes a filled shape one scanline at a time: each row's outline pixels mark the span endpoints.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    outline_colors = colors_present(g, exclude={bg})
    assert len(outline_colors) == 1, "expected exactly one outline color"
    outline = outline_colors[0]

    out = g.copy()
    filled_any = False
    for r in range(g.shape[0]):
        cols = np.flatnonzero(g[r] == outline)
        for left, right in zip(cols, cols[1:]):
            if right - left > 1:
                assert np.all(g[r, left + 1 : right] == bg), "span interior must be background"
                out[r, left + 1 : right] = Color.RED
                filled_any = True

    assert filled_any, "expected at least one bounded horizontal span"
    return out

# --------------- generator (o5x) ---------------

# Canvas, palette, and fill colors.
ROW_RANGE, COL_RANGE, RIGHT_PAD_RANGE = (3, 5), (8, 12), (0, 2)
BACKGROUND_COLOR, FILL_COLOR = 0, 2
OUTLINE_COLORS = [1, 3, 4, 5, 6, 7, 8, 9]

# Segment topology constraints.
SEGMENT_COUNT_RANGE, SEGMENT_LEN_RANGE = (2, 4), (2, 5)
MIN_MULTI_ROWS, MIN_ROW_GAP = 2, 2
START_COUNT_RANGE, START_ROWS, REQUIRED_START_ROW = (1, 2), {0, 1}, 0
DIAGONAL_DIRECTIONS = (-1, 1)

# Layout geometry and retry budget.
FORCE_BOTTOM_PROB = 0.5
PARALLEL_SEP_RANGE = (2, 6)
STAGGERED_MIN_ROWS, FOUR_ARM_MAX_ROWS = 4, 4
FOUR_ARM_SEP, FOUR_ARM_INNER_MAX = 4, 2
GENERATE_RETRIES = 300


def _gen_line(r0, c0, dc, steps):
    return [(r0 + i, c0 + dc * i) for i in range(steps + 1)]


def _force_one_arm_to_bottom(steps_a, steps_b, bottom_steps):
    if max(steps_a, steps_b) < bottom_steps:
        if random.random() < FORCE_BOTTOM_PROB:
            steps_a = bottom_steps
        else:
            steps_b = bottom_steps
    return steps_a, steps_b


def _gen_single_apex(R):
    left_steps = random.randint(SEGMENT_LEN_RANGE[0], R - 1)
    right_steps = random.randint(SEGMENT_LEN_RANGE[0], R - 1)
    left_steps, right_steps = _force_one_arm_to_bottom(left_steps, right_steps, R - 1)
    return [
        _gen_line(0, 0, -1, left_steps),
        _gen_line(0, 0, 1, right_steps),
    ]


def _gen_inward_merge(R):
    merge_r = random.randint(SEGMENT_LEN_RANGE[0], R - 1)
    segments = [
        _gen_line(0, 0, 1, merge_r),
        _gen_line(0, 2 * merge_r, -1, merge_r),
    ]
    if merge_r < R - 1:
        segments.append(_gen_line(merge_r, merge_r, random.choice(DIAGONAL_DIRECTIONS), R - 1 - merge_r))
    return segments


def _gen_parallel_pair(R):
    sep = random.randint(*PARALLEL_SEP_RANGE)
    d1, d2 = random.choice([(-1, -1), (1, 1), (-1, 1)])
    steps1 = random.randint(1, R - 1)
    steps2 = random.randint(1, R - 1)
    steps1, steps2 = _force_one_arm_to_bottom(steps1, steps2, R - 1)
    return [
        _gen_line(0, 0, d1, steps1),
        _gen_line(0, sep, d2, steps2),
    ]


def _gen_staggered_merge(R):
    if R < STAGGERED_MIN_ROWS:
        return None
    return [
        _gen_line(0, 0, 1, R - 1),
        _gen_line(1, 2 * R - 3, -1, R - 2),
    ]


def _gen_four_arm(R):
    if R > FOUR_ARM_MAX_ROWS:
        return None
    inner_max = min(FOUR_ARM_INNER_MAX, R - 1)
    return [
        _gen_line(0, 0, -1, R - 1),
        _gen_line(0, FOUR_ARM_SEP, 1, R - 1),
        _gen_line(0, 0, 1, random.randint(1, inner_max)),
        _gen_line(0, FOUR_ARM_SEP, -1, random.randint(1, inner_max)),
    ]


LAYOUT_BUILDERS = (_gen_single_apex, _gen_inward_merge, _gen_parallel_pair, _gen_staggered_merge, _gen_four_arm)


def _gen_segments(R):
    return random.choice(LAYOUT_BUILDERS)(R)


def _gen_graph_segments(coords):
    children = {}
    parents = {}
    for r, c in coords:
        children[(r, c)] = [(r + 1, c + dc) for dc in DIAGONAL_DIRECTIONS if (r + 1, c + dc) in coords]
        parents[(r, c)] = [(r - 1, c + dc) for dc in DIAGONAL_DIRECTIONS if (r - 1, c + dc) in coords]

    graph_segments = []
    seen_edges = set()
    for p in sorted(coords):
        for ch in sorted(children[p]):
            edge = (p, ch)
            if edge in seen_edges:
                continue
            direction = ch[1] - p[1]
            start = len(parents[p]) != 1 or len(children[p]) != 1
            if not start and parents[p]:
                start = p[1] - parents[p][0][1] != direction
            if not start:
                continue

            seg = [p, ch]
            seen_edges.add(edge)
            cur = ch
            while len(parents[cur]) == 1 and len(children[cur]) == 1:
                nxt = children[cur][0]
                if nxt[1] - cur[1] != direction:
                    break
                seen_edges.add((cur, nxt))
                seg.append(nxt)
                cur = nxt
            graph_segments.append(seg)
    return graph_segments


def _collect_coord_uses(segments, R):
    uses = {}
    for si, seg in enumerate(segments):
        if len(set(seg)) != len(seg):
            return None
        for idx, coord in enumerate(seg):
            r, _ = coord
            if not (0 <= r < R):
                return None
            uses.setdefault(coord, []).append((si, idx))
    return uses


def _shared_coords_are_endpoints(segments, uses):
    for coord_uses in uses.values():
        if len(coord_uses) <= 1:
            continue
        for si, idx in coord_uses:
            if idx not in (0, len(segments[si]) - 1):
                return False
    return True


def _row_constraints_hold(coords, R):
    multi_rows = 0
    for r in range(R):
        cols = sorted(c for rr, c in coords if rr == r)
        if len(cols) >= 2:
            multi_rows += 1
        if any(b - a < MIN_ROW_GAP for a, b in zip(cols, cols[1:])):
            return False
    return multi_rows >= MIN_MULTI_ROWS


def _all_points_have_diagonal_neighbor(coords, R):
    for r, c in coords:
        if not any(
            (r + dr, c + dc) in coords
            for dr in DIAGONAL_DIRECTIONS
            for dc in DIAGONAL_DIRECTIONS
            if 0 <= r + dr < R
        ):
            return False
    return True


def _start_points(coords):
    return [
        (r, c)
        for r, c in coords
        if not any((r - 1, c + dc) in coords for dc in DIAGONAL_DIRECTIONS)
    ]


def _gen_segments_are_valid(segments, R):
    if segments is None or not (SEGMENT_COUNT_RANGE[0] <= len(segments) <= SEGMENT_COUNT_RANGE[1]):
        return False
    if any(not (SEGMENT_LEN_RANGE[0] <= len(seg) <= SEGMENT_LEN_RANGE[1]) for seg in segments):
        return False

    uses = _collect_coord_uses(segments, R)
    if uses is None or not _shared_coords_are_endpoints(segments, uses):
        return False
    coords = set(uses)
    if {r for r, _ in coords} != set(range(R)):
        return False
    if not _row_constraints_hold(coords, R):
        return False
    if not _all_points_have_diagonal_neighbor(coords, R):
        return False
    starts = _start_points(coords)
    if not (START_COUNT_RANGE[0] <= len(starts) <= START_COUNT_RANGE[1]):
        return False
    if not any(r == REQUIRED_START_ROW for r, _ in starts):
        return False
    if any(r not in START_ROWS for r, _ in starts):
        return False

    graph_segments = _gen_graph_segments(coords)
    if not (SEGMENT_COUNT_RANGE[0] <= len(graph_segments) <= SEGMENT_COUNT_RANGE[1]):
        return False
    if any(not (SEGMENT_LEN_RANGE[0] <= len(seg) <= SEGMENT_LEN_RANGE[1]) for seg in graph_segments):
        return False
    return True


def _gen_expected_output(grid):
    out = grid.copy()
    outline_colors = [int(c) for c in np.unique(grid) if c != BACKGROUND_COLOR]
    if len(outline_colors) != 1:
        return None
    outline = outline_colors[0]
    filled_any = False
    for r in range(grid.shape[0]):
        cols = np.flatnonzero(grid[r] == outline)
        for left, right in zip(cols, cols[1:]):
            if right - left > 1:
                out[r, left + 1 : right] = FILL_COLOR
                filled_any = True
    return out if filled_any else None


def _normalize_coords(segments):
    coords = {coord for seg in segments for coord in seg}
    min_c = min(c for _, c in coords)
    shifted = {(r, c - min_c) for r, c in coords}
    span = max(c for _, c in shifted)
    return shifted, span


def _choose_canvas_width(span):
    valid_widths = []
    for W in range(COL_RANGE[0], COL_RANGE[1] + 1):
        pads = [pad for pad in range(RIGHT_PAD_RANGE[0], RIGHT_PAD_RANGE[1] + 1) if span + 1 + pad <= W]
        if pads:
            valid_widths.append((W, pads))
    if not valid_widths:
        return None

    W, pads = random.choice(valid_widths)
    right_pad = random.choice(pads)
    left_pad = W - right_pad - span - 1
    return W, left_pad


def _render_outline(R, W, shifted, left_pad, color):
    grid = np.zeros((R, W), dtype=int)
    coords = np.array(list(shifted), dtype=int)
    grid[coords[:, 0], coords[:, 1] + left_pad] = color
    return grid


def _gen_attempt():
    R = random.randint(*ROW_RANGE)
    segments = _gen_segments(R)
    if not _gen_segments_are_valid(segments, R):
        return None

    shifted, span = _normalize_coords(segments)
    canvas = _choose_canvas_width(span)
    if canvas is None:
        return None

    W, left_pad = canvas
    color = random.choice(OUTLINE_COLORS)
    return _render_outline(R, W, shifted, left_pad, color)


def generate_input():
    for _ in range(GENERATE_RETRIES):
        inp = _gen_attempt()
        if inp is None:
            continue
        expected = _gen_expected_output(inp)
        if expected is None:
            continue
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate valid aa18de87 input")
