from __future__ import annotations

from collections import defaultdict

import numpy as np

from arc_utils import Color, bbox_coords, colors_present, components_mask, get_bg
import itertools
import random


def _marker_centers(g: np.ndarray, bg: int, marker_color: int) -> list[tuple[int, int]]:
    centers = []
    for coords in components_mask(g == marker_color, connectivity=8):
        r0, c0, r1, c1 = bbox_coords(coords)
        assert coords.shape[0] == 4, "each marker must have four pixels"
        assert (r1 - r0, c1 - c0) == (2, 2), "each marker must span a 3x3 box"

        cr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        expected = {(cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)}
        assert set(map(tuple, coords)) == expected, "marker must be a hollow plus"
        assert int(g[cr, cc]) == bg, "marker center must be background"
        centers.append((int(cr), int(cc)))

    assert centers, "expected at least one marker"
    return centers


def _paint_between(out: np.ndarray, g: np.ndarray, bg: int, cells: tuple[slice, int] | tuple[int, slice]) -> int:
    target = g[cells]
    assert target.size > 0, "aligned markers must leave a gap to fill"
    assert np.all(target == bg), "connector path must be empty background"
    out[cells] = Color.BLUE
    return int(target.size)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect adjacent aligned red hollow markers by painting blue straight segments through the empty gaps between them.

    Input structure
    - The background is the dominant color, and the input foreground has exactly one color.
    - Each foreground object is an 8-connected four-pixel hollow plus: a 3x3 marker with only the north, south, east, and west arm pixels present.
    - Some marker centers share a row or column, with background cells between the opposing outer arms of adjacent markers.

    Rule
    - Compute the empty center of each marker.
    - For each row, sort the marker centers left to right and paint blue over the background cells between each adjacent pair's right and left arms.
    - For each column, sort the marker centers top to bottom and paint blue over the background cells between each adjacent pair's bottom and top arms.

    Key insight
    - The missing centers are anchors for a visibility graph: only neighboring anchors on the same row or column are connected, so a marker between two others blocks any longer-range connection.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)

    foreground_colors = colors_present(g, exclude={bg})
    assert len(foreground_colors) == 1, "input must contain one marker color"
    marker_color = foreground_colors[0]

    centers = _marker_centers(g, bg, marker_color)
    assert len(centers) >= 2, "expected multiple markers"

    rows: dict[int, list[int]] = defaultdict(list)
    cols: dict[int, list[int]] = defaultdict(list)
    for r, c in centers:
        rows[r].append(c)
        cols[c].append(r)

    out = g.copy()
    painted = 0
    for r, group in rows.items():
        ordered = sorted(group)
        for c0, c1 in zip(ordered, ordered[1:]):
            painted += _paint_between(out, g, bg, (r, slice(c0 + 2, c1 - 1)))

    for c, group in cols.items():
        ordered = sorted(group)
        for r0, r1 in zip(ordered, ordered[1:]):
            painted += _paint_between(out, g, bg, (slice(r0 + 2, r1 - 1), c))

    assert painted > 0, "expected at least one connector"
    return out

# --------------- generator (o5x) ---------------

# Colors used by the synthetic input/output pair.
_GEN_BG = Color.BLACK
_GEN_CONNECTOR = Color.BLUE
_GEN_MARKER = Color.RED

# Canvas bounds and hollow-plus marker geometry.
_GEN_HEIGHT_MIN = 9
_GEN_HEIGHT_MAX = 14
_GEN_WIDTH_MIN = 15
_GEN_WIDTH_MAX = 16
_GEN_MARKER_COUNT_MIN = 3
_GEN_MARKER_COUNT_MAX = 5
_GEN_CENTER_MARGIN = 1
_GEN_MARKER_SPAN = 3
_GEN_MARKER_ARM_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Center-to-center spacing; connector gaps are spacing minus marker span.
_GEN_MIN_STEP = 4
_GEN_MAX_STEP = 8
_GEN_STEP_VALUES = tuple(range(_GEN_MIN_STEP, _GEN_MAX_STEP + 1))
_GEN_MIN_GAP = 1
_GEN_MAX_GAP = 5
_GEN_GAP_START_OFFSET = 2
_GEN_GAP_STOP_OFFSET = 1

# Structural counts and motif choices.
_GEN_EDGE_COUNT_RANGES = {
    3: (2, 2),
    4: (2, 4),
    5: (3, 4),
}
_GEN_MOTIF_OPTIONS = {
    (3, 2): (("elbow", 0), ("straight3", 0)),
    (4, 2): (("elbow", 1), ("straight3", 1)),
    (4, 3): (("tee", 0), ("path4", 0), ("straight4", 0)),
    (4, 4): (("rectangle", 0),),
    (5, 3): (("tee", 1), ("path4", 1), ("straight4", 1)),
    (5, 4): (("rectangle", 1), ("star5", 0), ("branch_chain5", 0)),
}

# Retry budget for rejection sampling.
_GEN_ATTEMPT_LIMIT = 300


def _gen_step_tuple(count, max_sum=None):
    choices = [
        steps
        for steps in itertools.product(_GEN_STEP_VALUES, repeat=count)
        if max_sum is None or sum(steps) <= max_sum
    ]
    if not choices:
        return None
    return random.choice(choices)


def _gen_inner_spans(height, width):
    return height - _GEN_MARKER_SPAN, width - _GEN_MARKER_SPAN


def _gen_span_fits(row_span, col_span, height, width):
    h_span, w_span = _gen_inner_spans(height, width)
    return (row_span <= h_span and col_span <= w_span) or (
        col_span <= h_span and row_span <= w_span
    )


def _gen_center_bounds(height, width):
    return (
        _GEN_CENTER_MARGIN,
        height - _GEN_CENTER_MARGIN - 1,
        _GEN_CENTER_MARGIN,
        width - _GEN_CENTER_MARGIN - 1,
    )


def _gen_point_bounds(points):
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), max(rows), min(cols), max(cols)


def _gen_gap_length(start, end):
    return end - start - _GEN_MARKER_SPAN


def _gen_gap_slice(start, end):
    return slice(start + _GEN_GAP_START_OFFSET, end - _GEN_GAP_STOP_OFFSET)


def _gen_row_chain(steps):
    centers = [(0, 0)]
    c = 0
    for step in steps:
        c += step
        centers.append((0, c))
    return centers


def _gen_marker_cells(r, c):
    return {(r + dr, c + dc) for dr, dc in _GEN_MARKER_ARM_OFFSETS}


def _gen_transforms(points):
    return [
        [(r, c) for r, c in points],
        [(r, -c) for r, c in points],
        [(-r, c) for r, c in points],
        [(-r, -c) for r, c in points],
        [(c, r) for r, c in points],
        [(c, -r) for r, c in points],
        [(-c, r) for r, c in points],
        [(-c, -r) for r, c in points],
    ]


def _gen_fit_centers(raw_centers, height, width):
    valid = []
    h_span, w_span = _gen_inner_spans(height, width)
    for points in _gen_transforms(raw_centers):
        row_min, row_max, col_min, col_max = _gen_point_bounds(points)
        if row_max - row_min <= h_span and col_max - col_min <= w_span:
            valid.append(points)
    if not valid:
        return None

    points = random.choice(valid)
    row_min, row_max, col_min, col_max = _gen_point_bounds(points)
    row_low, row_high, col_low, col_high = _gen_center_bounds(height, width)
    row_shift = random.randint(row_low - row_min, row_high - row_max)
    col_shift = random.randint(col_low - col_min, col_high - col_max)
    centers = [(r + row_shift, c + col_shift) for r, c in points]
    random.shuffle(centers)
    return centers


def _gen_edges(centers):
    by_row = {}
    by_col = {}
    for r, c in centers:
        by_row.setdefault(r, []).append(c)
        by_col.setdefault(c, []).append(r)

    edges = []
    for r, cols in by_row.items():
        ordered = sorted(cols)
        for c0, c1 in zip(ordered, ordered[1:]):
            edges.append(("H", r, c0, c1, _gen_gap_length(c0, c1)))
    for c, rows in by_col.items():
        ordered = sorted(rows)
        for r0, r1 in zip(ordered, ordered[1:]):
            edges.append(("V", c, r0, r1, _gen_gap_length(r0, r1)))
    return edges


def _gen_path_cells(edges):
    cells = set()
    for edge in edges:
        if edge[0] == "H":
            _, r, c0, c1, _ = edge
            gap = _gen_gap_slice(c0, c1)
            cells.update((r, c) for c in range(gap.start, gap.stop))
        else:
            _, c, r0, r1, _ = edge
            gap = _gen_gap_slice(r0, r1)
            cells.update((r, c) for r in range(gap.start, gap.stop))
    return cells


def _gen_centers_valid(centers, height, width, marker_count, edge_count):
    if len(centers) != marker_count or len(set(centers)) != marker_count:
        return False
    row_low, row_high, col_low, col_high = _gen_center_bounds(height, width)
    for r, c in centers:
        if not (row_low <= r <= row_high and col_low <= c <= col_high):
            return False
    for i, (r0, c0) in enumerate(centers):
        for r1, c1 in centers[i + 1 :]:
            if max(abs(r0 - r1), abs(c0 - c1)) < _GEN_MARKER_SPAN:
                return False

    edges = _gen_edges(centers)
    if len(edges) != edge_count:
        return False
    if any(gap < _GEN_MIN_GAP or gap > _GEN_MAX_GAP for *_, gap in edges):
        return False

    path = _gen_path_cells(edges)
    marker_cells = set()
    for r, c in centers:
        cells = _gen_marker_cells(r, c)
        if any(rr < 0 or rr >= height or cc < 0 or cc >= width for rr, cc in cells):
            return False
        if marker_cells & cells:
            return False
        marker_cells.update(cells)
    return not (marker_cells & path)


def _gen_isolated_marker_choices(centers, height, width):
    occupied_rows = {r for r, _ in centers}
    occupied_cols = {c for _, c in centers}
    occupied_cells = set()
    for r, c in centers:
        occupied_cells.update(_gen_marker_cells(r, c))
    path = _gen_path_cells(_gen_edges(centers))

    choices = []
    for r in range(_GEN_CENTER_MARGIN, height - _GEN_CENTER_MARGIN):
        if r in occupied_rows:
            continue
        for c in range(_GEN_CENTER_MARGIN, width - _GEN_CENTER_MARGIN):
            if c in occupied_cols:
                continue
            if any(max(abs(r - rr), abs(c - cc)) < _GEN_MARKER_SPAN for rr, cc in centers):
                continue
            cells = _gen_marker_cells(r, c)
            if cells & occupied_cells or cells & path:
                continue
            choices.append((r, c))
    return choices


def _gen_add_isolated_marker(centers, height, width):
    choices = _gen_isolated_marker_choices(centers, height, width)
    if not choices:
        return None

    return centers + [random.choice(choices)]


def _gen_raw_elbow():
    d0 = random.randint(_GEN_MIN_STEP, _GEN_MAX_STEP)
    d1 = random.randint(_GEN_MIN_STEP, _GEN_MAX_STEP)
    return [(0, 0), (0, d0), (d1, d0)]


def _gen_raw_straight(height, width, count):
    steps = _gen_step_tuple(count - 1, max(_gen_inner_spans(height, width)))
    if steps is None:
        return None
    return _gen_row_chain(steps)


def _gen_raw_tee(height, width):
    h_span, w_span = _gen_inner_spans(height, width)
    span_wide = max(h_span, w_span)
    span_narrow = min(h_span, w_span)
    arms = _gen_step_tuple(2, span_wide)
    if arms is None or span_narrow < _GEN_MIN_STEP:
        return None
    left, right = arms
    leg = random.randint(_GEN_MIN_STEP, min(_GEN_MAX_STEP, span_narrow))
    return [(0, 0), (0, -left), (0, right), (leg, 0)]


def _gen_raw_rectangle(height, width):
    choices = []
    for dr in _GEN_STEP_VALUES:
        for dc in _GEN_STEP_VALUES:
            if _gen_span_fits(dr, dc, height, width):
                choices.append((dr, dc))
    if not choices:
        return None
    dr, dc = random.choice(choices)
    return [(0, 0), (0, dc), (dr, 0), (dr, dc)]


def _gen_raw_path4(height, width):
    choices = []
    for d0, d1, d2 in itertools.product(_GEN_STEP_VALUES, repeat=3):
        row_span = d1
        col_span = d0 + d2
        if _gen_span_fits(row_span, col_span, height, width):
            choices.append((d0, d1, d2))
    if not choices:
        return None
    d0, d1, d2 = random.choice(choices)
    return [(0, 0), (0, d0), (d1, d0), (d1, d0 + d2)]


def _gen_raw_star5(height, width):
    choices = []
    for up, down, left, right in itertools.product(_GEN_STEP_VALUES, repeat=4):
        row_span = up + down
        col_span = left + right
        if _gen_span_fits(row_span, col_span, height, width):
            choices.append((up, down, left, right))
    if not choices:
        return None
    up, down, left, right = random.choice(choices)
    return [(0, 0), (-up, 0), (down, 0), (0, -left), (0, right)]


def _gen_raw_branch_chain5(height, width):
    choices = []
    for steps in itertools.product(_GEN_STEP_VALUES, repeat=3):
        base_span = sum(steps)
        for branch in _GEN_STEP_VALUES:
            if _gen_span_fits(branch, base_span, height, width):
                for branch_index in range(len(steps) + 1):
                    choices.append((steps, branch, branch_index))
    if not choices:
        return None

    steps, branch, branch_index = random.choice(choices)
    centers = _gen_row_chain(steps)
    centers.append((branch, centers[branch_index][1]))
    return centers


def _gen_raw_centers(motif, height, width):
    if motif == "elbow":
        return _gen_raw_elbow()
    if motif == "straight3":
        return _gen_raw_straight(height, width, 3)
    if motif == "straight4":
        return _gen_raw_straight(height, width, 4)
    if motif == "tee":
        return _gen_raw_tee(height, width)
    if motif == "rectangle":
        return _gen_raw_rectangle(height, width)
    if motif == "path4":
        return _gen_raw_path4(height, width)
    if motif == "star5":
        return _gen_raw_star5(height, width)
    if motif == "branch_chain5":
        return _gen_raw_branch_chain5(height, width)
    raise ValueError(f"unknown motif {motif}")


def _gen_draw_marker(grid, r, c):
    for rr, cc in _gen_marker_cells(r, c):
        grid[rr, cc] = _GEN_MARKER


def _gen_draw_edges(grid, centers):
    for edge in _gen_edges(centers):
        if edge[0] == "H":
            _, r, c0, c1, _ = edge
            grid[r, _gen_gap_slice(c0, c1)] = _GEN_CONNECTOR
        else:
            _, c, r0, r1, _ = edge
            grid[_gen_gap_slice(r0, r1), c] = _GEN_CONNECTOR


def _gen_edge_count(marker_count):
    low, high = _GEN_EDGE_COUNT_RANGES[marker_count]
    if low == high:
        return low
    return random.randint(low, high)


def _gen_layout_centers(height, width, marker_count, edge_count):
    options = _GEN_MOTIF_OPTIONS.get((marker_count, edge_count), ())
    if not options:
        return None
    motif, isolated_count = random.choice(options)
    raw_centers = _gen_raw_centers(motif, height, width)
    if raw_centers is None:
        return None
    centers = _gen_fit_centers(raw_centers, height, width)
    if centers is None:
        return None
    for _ in range(isolated_count):
        centers = _gen_add_isolated_marker(centers, height, width)
        if centers is None:
            return None
    return centers


def _gen_render_output(height, width, centers):
    output = np.full((height, width), _GEN_BG, dtype=int)
    for r, c in centers:
        _gen_draw_marker(output, r, c)
    _gen_draw_edges(output, centers)
    return output


def _gen_matches_solver(generated, output):
    try:
        actual = main(generated)
    except AssertionError:
        return False
    return np.array_equal(actual, output)


def _gen_attempt():
    height = random.randint(_GEN_HEIGHT_MIN, _GEN_HEIGHT_MAX)
    width = random.randint(_GEN_WIDTH_MIN, _GEN_WIDTH_MAX)
    marker_count = random.randint(_GEN_MARKER_COUNT_MIN, _GEN_MARKER_COUNT_MAX)
    edge_count = _gen_edge_count(marker_count)
    centers = _gen_layout_centers(height, width, marker_count, edge_count)
    if centers is None:
        return None
    if not _gen_centers_valid(centers, height, width, marker_count, edge_count):
        return None

    output = _gen_render_output(height, width, centers)
    generated = output.copy()
    generated[generated == _GEN_CONNECTOR] = _GEN_BG
    if not _gen_matches_solver(generated, output):
        return None
    return generated


def generate_input():
    for _ in range(_GEN_ATTEMPT_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 60a26a3e input")
