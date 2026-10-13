import numpy as np

from arc_utils import Color, bbox, bbox as _gen_bbox, colors_present, get_bg, paste as _gen_paste
import math
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint the blank separator row or column inside the foreground bounding box green across the whole grid.

    Input structure
    A background grid contains one colored foreground mask whose bounding box has exactly one completely empty interior row or column.

    Rule
    Find the foreground bounding box and locate the all-background row or column strictly inside it; recolor that entire grid row or column green.

    Key insight
    The blank interior line is the separation axis between the colored subregions, and the output marks this axis over the full canvas.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    assert bg != Color.GREEN

    foreground_colors = colors_present(g, exclude={bg})
    assert len(foreground_colors) == 1
    assert Color.GREEN not in foreground_colors

    foreground_box = bbox(g != bg)
    assert foreground_box is not None
    r0, c0, r1, c1 = foreground_box

    empty_rows = [r for r in range(r0 + 1, r1) if np.all(g[r, :] == bg)]
    empty_cols = [c for c in range(c0 + 1, c1) if np.all(g[:, c] == bg)]
    separators = [("row", r) for r in empty_rows] + [("col", c) for c in empty_cols]
    assert len(separators) == 1

    axis, index = separators[0]
    out = g.copy()
    if axis == "row":
        out[index, :] = Color.GREEN
    else:
        out[:, index] = Color.GREEN
    return out

# --------------- generator (o5x) ---------------

# Palette and canvas.
_GEN_BG = 0
_GEN_GREEN = 3
_GEN_COLORS = [1, 2, 4, 5, 6, 7, 8, 9]
_GEN_GRID_SIZE = 10
_GEN_EDGE_MARGIN = 1
_GEN_SEPARATOR_THICKNESS = 1

# Fill modes and validation.
_GEN_FILL_RATIO_MIN = 0.4
_GEN_FILL_RATIO_MAX = 0.8
_GEN_FILL_MODES = ["low", "mid", "high"]
_GEN_FILL_WEIGHTS = [4, 4, 2]
_GEN_FILL_BANDS = {"low": (0.4, 0.5), "mid": (0.5, 0.6), "high": (0.6, 0.7)}
_GEN_ROW_AXIS_PROB = 0.5
_GEN_MAX_ATTEMPTS = 300

# Interval selection.
_GEN_INTERVAL_MIN_LEN = 2
_GEN_INTERVAL_MAX_LEN = 6
_GEN_INTERVAL_POWER_DEFAULT = 2
_GEN_INTERVAL_POWER_LOW = 1
_GEN_INTERVAL_POWER_MID = 4

# Row-separator layout choices.
_GEN_ROW_HEIGHT_PAIRS = {
    "low": [(1, 2), (2, 1), (1, 3), (2, 2), (3, 1), (1, 4), (2, 3), (3, 2), (4, 1)],
    "mid": [(1, 3), (2, 2), (3, 1), (1, 4), (2, 3), (3, 2), (4, 1)],
    "high": [(1, 4), (2, 3), (3, 2), (4, 1)],
}
_GEN_ROW_WIDTH_RANGE = (4, 7)
_GEN_ROW_WIDTH_RANGE_HIGH = (4, 6)

# Column-separator layout choices.
_GEN_COL_WIDTH_PAIRS = {
    "low": [(1, 2), (2, 1), (1, 3), (2, 2), (3, 1), (1, 4), (2, 3), (3, 2), (4, 1), (2, 4), (3, 3), (4, 2)],
    "mid": [(1, 3), (2, 2), (3, 1), (1, 4), (2, 3), (3, 2), (4, 1), (2, 4), (3, 3), (4, 2)],
    "high": [(2, 4), (3, 3), (4, 2)],
}
_GEN_COL_HEIGHT_RANGE = (4, 6)


def _gen_interval_pair(total, min_len=_GEN_INTERVAL_MIN_LEN, max_len=_GEN_INTERVAL_MAX_LEN, power=_GEN_INTERVAL_POWER_DEFAULT):
    intervals = []
    full_cover = set(range(total))
    for a0 in range(total):
        for a1 in range(a0 + min_len - 1, min(total, a0 + max_len)):
            cover_a = set(range(a0, a1 + 1))
            for b0 in range(total):
                for b1 in range(b0 + min_len - 1, min(total, b0 + max_len)):
                    cover_b = set(range(b0, b1 + 1))
                    if cover_a | cover_b == full_cover:
                        intervals.append((a0, a1, b0, b1))
    weights = [((a1 - a0 + 1) + (b1 - b0 + 1)) ** power for a0, a1, b0, b1 in intervals]
    return random.choices(intervals, weights=weights, k=1)[0]


def _gen_region(rows, cols, required, target):
    required = set(required)
    if len(required) > target:
        return None

    cells = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in required]
    chosen = set(required)
    chosen.update(random.sample(cells, target - len(chosen)))

    patch = np.zeros((rows, cols), dtype=int)
    if chosen:
        rr, cc = zip(*chosen)
        patch[rr, cc] = 1
    return patch


def _gen_region_bounds(rows, cols, required):
    area = rows * cols
    return max(math.ceil(area * _GEN_FILL_RATIO_MIN), len(required)), math.floor(area * _GEN_FILL_RATIO_MAX)


def _gen_region_pair(rows_a, cols_a, req_a, rows_b, cols_b, req_b, total_area, fill_mode):
    """Build two half-regions whose combined foreground count matches a fill band."""
    lo_a, hi_a = _gen_region_bounds(rows_a, cols_a, req_a)
    lo_b, hi_b = _gen_region_bounds(rows_b, cols_b, req_b)
    total_lo = max(math.ceil(total_area * _GEN_FILL_RATIO_MIN), lo_a + lo_b)
    total_hi = min(math.floor(total_area * _GEN_FILL_RATIO_MAX), hi_a + hi_b)
    if lo_a > hi_a or lo_b > hi_b or total_lo > total_hi:
        return None, None

    fill_lo, fill_hi = _GEN_FILL_BANDS[fill_mode]
    target_options = [n for n in range(total_lo, total_hi + 1) if fill_lo <= n / total_area < fill_hi]
    if not target_options:
        return None, None
    target_total = random.choice(target_options)
    count_a_lo = max(lo_a, target_total - hi_b)
    count_a_hi = min(hi_a, target_total - lo_b)
    count_a = random.randint(count_a_lo, count_a_hi)
    count_b = target_total - count_a
    return (
        _gen_region(rows_a, cols_a, req_a, count_a),
        _gen_region(rows_b, cols_b, req_b, count_b),
    )


def _gen_interval_spans(total, fill_mode):
    if fill_mode == "high":
        return 0, total - 1, 0, total - 1
    power = _GEN_INTERVAL_POWER_MID if fill_mode == "mid" else _GEN_INTERVAL_POWER_LOW
    return _gen_interval_pair(total, power=power)


def _gen_require_each_line(required, line_count, span_count, line_axis):
    for i in range(line_count):
        if line_axis == "row":
            required.add((i, random.randint(0, span_count - 1)))
        else:
            required.add((random.randint(0, span_count - 1), i))


def _gen_require_span_edges(required, span_len, cross_len, line_axis):
    for edge in (0, span_len - 1):
        if line_axis == "col":
            required.add((random.randrange(cross_len), edge))
        else:
            required.add((edge, random.randrange(cross_len)))


def _gen_cover_gap_axis(req_a, req_b, total, span_a, span_b, depth_a, depth_b, line_axis):
    a0, a1 = span_a
    b0, b1 = span_b
    for i in range(total):
        owners = []
        if a0 <= i <= a1:
            owners.append("a")
        if b0 <= i <= b1:
            owners.append("b")
        if random.choice(owners) == "a":
            if line_axis == "col":
                req_a.add((random.randrange(depth_a), i - a0))
            else:
                req_a.add((i - a0, random.randrange(depth_a)))
        elif line_axis == "col":
            req_b.add((random.randrange(depth_b), i - b0))
        else:
            req_b.add((i - b0, random.randrange(depth_b)))


def _gen_build_row_separator(fill_mode):
    """Construct a local bbox containing exactly one blank separator row."""
    top_h, bottom_h = random.choice(_GEN_ROW_HEIGHT_PAIRS[fill_mode])
    height = top_h + _GEN_SEPARATOR_THICKNESS + bottom_h
    width_lo, width_hi = _GEN_ROW_WIDTH_RANGE_HIGH if fill_mode == "high" else _GEN_ROW_WIDTH_RANGE
    width = random.randint(width_lo, width_hi)
    top_c0, top_c1, bot_c0, bot_c1 = _gen_interval_spans(width, fill_mode)

    top_req = set()
    bot_req = set()
    top_w = top_c1 - top_c0 + 1
    bottom_w = bot_c1 - bot_c0 + 1
    _gen_require_each_line(top_req, top_h, top_w, "row")
    _gen_require_each_line(bot_req, bottom_h, bottom_w, "row")
    _gen_require_span_edges(top_req, top_w, top_h, "col")
    _gen_require_span_edges(bot_req, bottom_w, bottom_h, "col")
    _gen_cover_gap_axis(top_req, bot_req, width, (top_c0, top_c1), (bot_c0, bot_c1), top_h, bottom_h, "col")

    top, bottom = _gen_region_pair(top_h, top_w, top_req, bottom_h, bottom_w, bot_req, height * width, fill_mode)
    if top is None or bottom is None:
        return None

    local = np.zeros((height, width), dtype=int)
    _gen_paste(local, top, 0, top_c0, transparent=_GEN_BG)
    _gen_paste(local, bottom, top_h + _GEN_SEPARATOR_THICKNESS, bot_c0, transparent=_GEN_BG)
    return local, "row", top_h


def _gen_build_col_separator(fill_mode):
    """Construct a local bbox containing exactly one blank separator column."""
    left_w, right_w = random.choice(_GEN_COL_WIDTH_PAIRS[fill_mode])
    height_lo, height_hi = _GEN_COL_HEIGHT_RANGE
    height = random.randint(height_lo, height_hi)
    width = left_w + _GEN_SEPARATOR_THICKNESS + right_w
    left_r0, left_r1, right_r0, right_r1 = _gen_interval_spans(height, fill_mode)

    left_req = set()
    right_req = set()
    left_h = left_r1 - left_r0 + 1
    right_h = right_r1 - right_r0 + 1
    _gen_require_each_line(left_req, left_w, left_h, "col")
    _gen_require_each_line(right_req, right_w, right_h, "col")
    _gen_require_span_edges(left_req, left_h, left_w, "row")
    _gen_require_span_edges(right_req, right_h, right_w, "row")
    _gen_cover_gap_axis(left_req, right_req, height, (left_r0, left_r1), (right_r0, right_r1), left_w, right_w, "row")

    left, right = _gen_region_pair(left_h, left_w, left_req, right_h, right_w, right_req, height * width, fill_mode)
    if left is None or right is None:
        return None

    local = np.zeros((height, width), dtype=int)
    _gen_paste(local, left, left_r0, 0, transparent=_GEN_BG)
    _gen_paste(local, right, right_r0, left_w + _GEN_SEPARATOR_THICKNESS, transparent=_GEN_BG)
    return local, "col", left_w


def _gen_build_separator(fill_mode):
    if random.random() < _GEN_ROW_AXIS_PROB:
        return _gen_build_row_separator(fill_mode)
    return _gen_build_col_separator(fill_mode)


def _gen_empty_lines(local):
    box = _gen_bbox(local != _GEN_BG)
    if box is None:
        return None, [], []
    r0, c0, r1, c1 = box
    empty_rows = [r for r in range(r0 + 1, r1) if np.all(local[r, :] == _GEN_BG)]
    empty_cols = [c for c in range(c0 + 1, c1) if np.all(local[:, c] == _GEN_BG)]
    return box, empty_rows, empty_cols


def _gen_local_is_valid(local, axis, sep):
    box, empty_rows, empty_cols = _gen_empty_lines(local)
    if box is None:
        return False
    if empty_rows != ([sep] if axis == "row" else []):
        return False
    if empty_cols != ([sep] if axis == "col" else []):
        return False
    if box != (0, 0, local.shape[0] - 1, local.shape[1] - 1):
        return False

    fill = int(np.count_nonzero(local)) / local.size
    return _GEN_FILL_RATIO_MIN <= fill <= _GEN_FILL_RATIO_MAX


def _gen_render_on_canvas(local, axis, sep):
    color = random.choice(_GEN_COLORS)
    grid = np.zeros((_GEN_GRID_SIZE, _GEN_GRID_SIZE), dtype=int)
    patch = local * color
    r0 = random.randint(_GEN_EDGE_MARGIN, _GEN_GRID_SIZE - local.shape[0] - _GEN_EDGE_MARGIN)
    c0 = random.randint(_GEN_EDGE_MARGIN, _GEN_GRID_SIZE - local.shape[1] - _GEN_EDGE_MARGIN)
    _gen_paste(grid, patch, r0, c0, transparent=_GEN_BG)

    out = grid.copy()
    if axis == "row":
        out[r0 + sep, :] = _GEN_GREEN
    else:
        out[:, c0 + sep] = _GEN_GREEN
    return grid, out


def _gen_attempt():
    fill_mode = random.choices(_GEN_FILL_MODES, weights=_GEN_FILL_WEIGHTS, k=1)[0]
    built = _gen_build_separator(fill_mode)
    if built is None:
        return None
    local, axis, sep = built

    if not _gen_local_is_valid(local, axis, sep):
        return None

    grid, out = _gen_render_on_canvas(local, axis, sep)
    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input():
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid da2b0fe3 input")
