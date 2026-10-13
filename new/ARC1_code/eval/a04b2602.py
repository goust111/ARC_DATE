import numpy as np

from arc_utils import Color, find_components, in_bounds, label_mask, neighbors8, paste
import random


def _bbox_contains(bbox: tuple[int, int, int, int], coord: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    r, c = coord
    return r0 <= r <= r1 and c0 <= c <= c1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint a blue 3x3 hollow neighborhood around each red marker inside a green object's bounding rectangle.

    Input structure
    - The input contains a black background, green 4-connected objects, and red marker pixels.
    - Each green object forms an almost-rectangular region; its bounding box identifies the region it controls.
    - The active red markers are exactly the red pixels lying inside one of those green-object bounding boxes.
    Rule
    - Preserve the input grid, including every active red marker itself.
    - For each active red marker, recolor all in-bounds 8-neighbor cells to blue.
    - If two active markers touch, their marker centers remain red rather than being overwritten by the other's blue stamp.
    Key insight
    - Red pixels outside the green rectangles are distractors; containment in a green component's bbox selects the markers to stamp.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= {Color.BLACK, Color.RED, Color.GREEN}

    green_objects = find_components(
        g,
        bg=Color.BLACK,
        connectivity=4,
        monochromatic=True,
        colors=[Color.GREEN],
        return_format="dict",
    )
    assert green_objects, "expected at least one green object"
    bboxes = [obj["bbox"] for obj in green_objects]

    red_pixels = [tuple(map(int, coord)) for coord in np.argwhere(g == Color.RED)]
    assert red_pixels, "expected red marker pixels"

    active_markers: set[tuple[int, int]] = set()
    for coord in red_pixels:
        containing = [bbox for bbox in bboxes if _bbox_contains(bbox, coord)]
        assert len(containing) <= 1, "green object bounding boxes should not overlap at a red marker"
        if containing:
            active_markers.add(coord)

    assert active_markers, "expected at least one red marker inside a green-object bbox"
    for bbox in bboxes:
        assert any(_bbox_contains(bbox, marker) for marker in active_markers)

    out = g.copy()
    for r, c in active_markers:
        for rr, cc in neighbors8(r, c):
            if in_bounds(g.shape, rr, cc) and (rr, cc) not in active_markers:
                out[rr, cc] = Color.BLUE

    return out

# --------------- generator (o5x) ---------------

# Colors used by generated inputs.
_GEN_BLACK = Color.BLACK
_GEN_RED = Color.RED
_GEN_GREEN = Color.GREEN

# Canvas and rectangle bounds.
_GEN_HEIGHT_RANGE = (10, 20)
_GEN_WIDTH_RANGE = (17, 27)
_GEN_RECT_COUNT_CHOICES = (2, 3)
_GEN_RECT_HEIGHT_RANGE = (4, 14)
_GEN_RECT_WIDTH_RANGE = (4, 12)
_GEN_MIN_AXIS_GAP = 1

# Marker and distractor count / placement bounds.
_GEN_ACTIVE_TOTAL_RANGE = (4, 18)
_GEN_ACTIVE_PER_RECT_RANGE = (1, 7)
_GEN_DISTRACTOR_COUNT_RANGE = (6, 13)
_GEN_DISTRACTOR_DISTANCE_RANGE = (1, 5)

# Layout structure constants.
_GEN_PAIR_COUNT = 2
_GEN_L_RECT_COUNT = 3

# Retry budget for rejection sampling.
_GEN_MAX_ATTEMPTS = 300


def _gen_size_tuples(total, n, size_range):
    lo, hi = size_range
    hi = min(hi, total)
    out = []

    def rec(prefix):
        if len(prefix) == n:
            if sum(prefix) + _GEN_MIN_AXIS_GAP * (n - 1) <= total:
                out.append(tuple(prefix))
            return
        remaining_slots = n - len(prefix) - 1
        for size in range(lo, hi + 1):
            min_remaining = remaining_slots * lo
            min_gaps = _GEN_MIN_AXIS_GAP * (n - 1)
            if sum(prefix) + size + min_remaining + min_gaps <= total:
                rec(prefix + [size])

    rec([])
    return out


def _gen_composition(total, n):
    parts = [0] * n
    for _ in range(total):
        parts[random.randrange(n)] += 1
    return parts


def _gen_split_axis(total, n, size_range):
    choices = _gen_size_tuples(total, n, size_range)
    if not choices:
        return None
    sizes = list(random.choice(choices))
    leftover = total - sum(sizes) - _GEN_MIN_AXIS_GAP * (n - 1)
    extras = _gen_composition(leftover, n + 1)
    spans = []
    pos = extras[0]
    for i, size in enumerate(sizes):
        spans.append((pos, pos + size - 1))
        pos += size
        if i < n - 1:
            pos += _GEN_MIN_AXIS_GAP + extras[i + 1]
    return spans


def _gen_has_split(total, n, size_range):
    return bool(_gen_size_tuples(total, n, size_range))


def _gen_random_span(total, size_range):
    lo, hi = size_range
    hi = min(hi, total)
    size = random.randint(lo, hi)
    start = random.randint(0, total - size)
    return start, start + size - 1


def _gen_bbox(row_span, col_span):
    r0, r1 = row_span
    c0, c1 = col_span
    return r0, c0, r1, c1


def _gen_horizontal_bboxes(h, w, n):
    col_spans = _gen_split_axis(w, n, _GEN_RECT_WIDTH_RANGE)
    if col_spans is None:
        return None
    return [_gen_bbox(_gen_random_span(h, _GEN_RECT_HEIGHT_RANGE), col_span) for col_span in col_spans]


def _gen_vertical_bboxes(h, w, n):
    row_spans = _gen_split_axis(h, n, _GEN_RECT_HEIGHT_RANGE)
    if row_spans is None:
        return None
    return [_gen_bbox(row_span, _gen_random_span(w, _GEN_RECT_WIDTH_RANGE)) for row_span in row_spans]


def _gen_diagonal_bboxes(h, w):
    row_spans = _gen_split_axis(h, _GEN_PAIR_COUNT, _GEN_RECT_HEIGHT_RANGE)
    col_spans = _gen_split_axis(w, _GEN_PAIR_COUNT, _GEN_RECT_WIDTH_RANGE)
    if row_spans is None or col_spans is None:
        return None
    if random.randrange(_GEN_PAIR_COUNT):
        pairs = [(row_spans[0], col_spans[0]), (row_spans[1], col_spans[1])]
    else:
        pairs = [(row_spans[0], col_spans[1]), (row_spans[1], col_spans[0])]
    return [_gen_bbox(row_span, col_span) for row_span, col_span in pairs]


def _gen_column_l_bboxes(h, w):
    col_spans = _gen_split_axis(w, _GEN_PAIR_COUNT, _GEN_RECT_WIDTH_RANGE)
    row_spans = _gen_split_axis(h, _GEN_PAIR_COUNT, _GEN_RECT_HEIGHT_RANGE)
    if col_spans is None or row_spans is None:
        return None
    stack_col = random.randrange(_GEN_PAIR_COUNT)
    single_col = 1 - stack_col
    bboxes = [_gen_bbox(row_span, col_spans[stack_col]) for row_span in row_spans]
    bboxes.append(_gen_bbox(_gen_random_span(h, _GEN_RECT_HEIGHT_RANGE), col_spans[single_col]))
    random.shuffle(bboxes)
    return bboxes


def _gen_row_l_bboxes(h, w):
    row_spans = _gen_split_axis(h, _GEN_PAIR_COUNT, _GEN_RECT_HEIGHT_RANGE)
    col_spans = _gen_split_axis(w, _GEN_PAIR_COUNT, _GEN_RECT_WIDTH_RANGE)
    if row_spans is None or col_spans is None:
        return None
    stack_row = random.randrange(_GEN_PAIR_COUNT)
    single_row = 1 - stack_row
    bboxes = [_gen_bbox(row_spans[stack_row], col_span) for col_span in col_spans]
    bboxes.append(_gen_bbox(row_spans[single_row], _gen_random_span(w, _GEN_RECT_WIDTH_RANGE)))
    random.shuffle(bboxes)
    return bboxes


def _gen_make_bboxes(h, w, n):
    layouts = []
    has_pair_split = (
        _gen_has_split(h, _GEN_PAIR_COUNT, _GEN_RECT_HEIGHT_RANGE)
        and _gen_has_split(w, _GEN_PAIR_COUNT, _GEN_RECT_WIDTH_RANGE)
    )
    if _gen_has_split(w, n, _GEN_RECT_WIDTH_RANGE):
        layouts.append(_gen_horizontal_bboxes)
    if _gen_has_split(h, n, _GEN_RECT_HEIGHT_RANGE):
        layouts.append(_gen_vertical_bboxes)
    if n == _GEN_PAIR_COUNT and has_pair_split:
        layouts.append(lambda hh, ww, nn: _gen_diagonal_bboxes(hh, ww))
    if n == _GEN_L_RECT_COUNT and has_pair_split:
        layouts.extend([
            lambda hh, ww, nn: _gen_column_l_bboxes(hh, ww),
            lambda hh, ww, nn: _gen_row_l_bboxes(hh, ww),
        ])
    if not layouts:
        return None
    return random.choice(layouts)(h, w, n)


def _gen_bounded_counts(n, total, count_range):
    lo, hi = count_range
    choices = []

    def rec(prefix, remaining):
        if len(prefix) == n:
            if remaining == 0:
                choices.append(tuple(prefix))
            return
        slots = n - len(prefix) - 1
        for count in range(lo, hi + 1):
            if slots * lo <= remaining - count <= hi * slots:
                rec(prefix + [count], remaining - count)

    rec([], total)
    return list(random.choice(choices)) if choices else None


def _gen_green_component_ok(grid, bbox):
    r0, c0, r1, c1 = bbox
    mask = grid[r0 : r1 + 1, c0 : c1 + 1] == _GEN_GREEN
    if not (mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
        return False
    _, num = label_mask(mask, connectivity=4)
    return num == 1


def _gen_place_active_markers(grid, bboxes, counts):
    for bbox, count in zip(bboxes, counts):
        r0, c0, r1, c1 = bbox
        cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
        for r, c in random.sample(cells, count):
            grid[r, c] = _GEN_RED
        if not _gen_green_component_ok(grid, bbox):
            return False
    return True


def _gen_bbox_distance(r, c, bbox):
    r0, c0, r1, c1 = bbox
    return max(r0 - r, 0, r - r1) + max(c0 - c, 0, c - c1)


def _gen_outside_bboxes(r, c, bboxes):
    return all(not (r0 <= r <= r1 and c0 <= c <= c1) for r0, c0, r1, c1 in bboxes)


def _gen_distractor_candidates(grid, bboxes):
    min_dist, max_dist = _GEN_DISTRACTOR_DISTANCE_RANGE
    choices = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] != _GEN_BLACK or not _gen_outside_bboxes(r, c, bboxes):
                continue
            dist = min(_gen_bbox_distance(r, c, bbox) for bbox in bboxes)
            if min_dist <= dist <= max_dist:
                choices.append((r, c))
    return choices


def _gen_place_distractors(grid, bboxes):
    choices = _gen_distractor_candidates(grid, bboxes)
    count = random.randint(*_GEN_DISTRACTOR_COUNT_RANGE)
    if len(choices) < count:
        return False
    for r, c in random.sample(choices, count):
        grid[r, c] = _GEN_RED
    return True


def _gen_sample_plan():
    h = random.randint(*_GEN_HEIGHT_RANGE)
    w = random.randint(*_GEN_WIDTH_RANGE)
    total_active = random.randint(*_GEN_ACTIVE_TOTAL_RANGE)
    per_rect_min, per_rect_max = _GEN_ACTIVE_PER_RECT_RANGE
    rect_counts = [
        n for n in _GEN_RECT_COUNT_CHOICES if per_rect_min * n <= total_active <= per_rect_max * n
    ]
    n_rects = random.choice(rect_counts)
    marker_counts = _gen_bounded_counts(n_rects, total_active, _GEN_ACTIVE_PER_RECT_RANGE)
    if marker_counts is None:
        return None
    return h, w, n_rects, marker_counts


def _gen_paint_rectangles(grid, bboxes):
    for r0, c0, r1, c1 in bboxes:
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), _GEN_GREEN, dtype=int)
        paste(grid, patch, r0, c0)


def _gen_attempt():
    plan = _gen_sample_plan()
    if plan is None:
        return None
    h, w, n_rects, marker_counts = plan

    bboxes = _gen_make_bboxes(h, w, n_rects)
    if bboxes is None:
        return None

    grid = np.full((h, w), _GEN_BLACK, dtype=int)
    _gen_paint_rectangles(grid, bboxes)
    if not _gen_place_active_markers(grid, bboxes, marker_counts):
        return None
    if not _gen_place_distractors(grid, bboxes):
        return None
    return grid


def generate_input():
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is None:
            continue
        try:
            main(grid)
        except Exception:
            continue
        return grid
    raise RuntimeError("failed to generate a valid a04b2602 input")
