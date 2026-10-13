import numpy as np

from arc_utils import colors_present, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete straight horizontal and vertical connectors between same-color marker pixels that already share a row or column.

    Input structure
    - A mostly background grid containing isolated foreground marker pixels of one color.

    Rule
    - For every row containing at least two original foreground pixels, color the full span from the leftmost to rightmost such pixel.
    - For every column containing at least two original foreground pixels, color the full span from the topmost to bottommost such pixel.
    - Leave all other cells unchanged.

    Key insight
    - The markers define all orthogonal line segments by row/column alignment; spans are computed from the input markers, not from newly drawn pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 1

    color = fg_colors[0]
    markers = g == color
    assert np.any(markers)

    out = g.copy()
    h, w = g.shape

    for r in range(h):
        cols = np.flatnonzero(markers[r])
        if len(cols) >= 2:
            out[r, cols[0] : cols[-1] + 1] = color

    for c in range(w):
        rows = np.flatnonzero(markers[:, c])
        if len(rows) >= 2:
            out[rows[0] : rows[-1] + 1, c] = color

    return out

# --------------- generator (o5x) ---------------

# Grid and palette.
GEN_SHAPE = (10, 10)
GEN_BG = 0
GEN_COLOR = 8
GEN_COORDS = tuple(range(1, 10))

# Line and marker constraints.
GEN_MIN_SPAN = 4
GEN_MAX_SPAN = 7
GEN_MIN_LINES_PER_AXIS = 1
GEN_MAX_LINES_PER_AXIS = 2
GEN_MAX_MARKERS = 6
GEN_MAX_PER_ROW_OR_COL = 2
GEN_ISOLATED_MARKERS = 1

# Retry budget.
GEN_RETRY_LIMIT = 300


def _gen_span_ok(a, b):
    return GEN_MIN_SPAN <= abs(int(a) - int(b)) + 1 <= GEN_MAX_SPAN


def _gen_shuffle(seq):
    vals = list(seq)
    np.random.shuffle(vals)
    return vals


def _gen_pick_indices(count, require_pair_span):
    vals = list(GEN_COORDS)
    if count == 1:
        return [int(np.random.choice(vals))]

    if require_pair_span:
        pairs = [(a, b) for a in vals for b in vals if a < b and _gen_span_ok(a, b)]
    else:
        pairs = [(a, b) for a in vals for b in vals if a < b]
    if not pairs:
        return None
    return list(pairs[int(np.random.randint(len(pairs)))])


def _gen_pick_shared(h_count, v_count):
    cells = [(i, j) for i in range(h_count) for j in range(v_count)]
    min_shared = max(
        0,
        GEN_MAX_PER_ROW_OR_COL * (h_count + v_count) + GEN_ISOLATED_MARKERS - GEN_MAX_MARKERS,
    )
    max_shared = min(len(cells), GEN_MAX_PER_ROW_OR_COL * h_count, GEN_MAX_PER_ROW_OR_COL * v_count)
    shared_count = int(np.random.randint(min_shared, max_shared + 1))
    return set(_gen_shuffle(cells)[:shared_count])


def _gen_pair_span_requirements(shared, h_count, v_count):
    h_rows_need_span = any(
        sum((i, j) in shared for i in range(h_count)) == GEN_MAX_PER_ROW_OR_COL
        for j in range(v_count)
    )
    v_cols_need_span = any(
        sum((i, j) in shared for j in range(v_count)) == GEN_MAX_PER_ROW_OR_COL
        for i in range(h_count)
    )
    return h_rows_need_span, v_cols_need_span


def _gen_used_positions(endpoints, coord_index):
    return {cell[coord_index] for cell in endpoints}


def _gen_no_extra_crossing(span_positions, crossing_positions, allowed_crossings):
    lo, hi = sorted(span_positions)
    return all(p in allowed_crossings or not (lo < p < hi) for p in crossing_positions)


def _gen_choose_free_positions(
    existing_positions,
    blocked_positions,
    used_positions,
    crossing_positions,
    allowed_crossings,
    needed,
):
    available = [
        p for p in GEN_COORDS
        if p not in blocked_positions and p not in used_positions
    ]
    choices = []
    if needed == 1:
        for p in available:
            positions = list(existing_positions) + [p]
            if _gen_span_ok(positions[0], positions[1]) and _gen_no_extra_crossing(
                positions, crossing_positions, allowed_crossings
            ):
                choices.append([p])
    else:
        for i, p0 in enumerate(available):
            for p1 in available[i + 1:]:
                positions = [p0, p1]
                if _gen_span_ok(p0, p1) and _gen_no_extra_crossing(
                    positions, crossing_positions, allowed_crossings
                ):
                    choices.append(positions)
    if not choices:
        return None
    return choices[int(np.random.randint(len(choices)))]


def _gen_seed_shared_endpoints(h_rows, v_cols, shared):
    row_cols = {r: set() for r in h_rows}
    col_rows = {c: set() for c in v_cols}
    endpoints = set()
    for i, r in enumerate(h_rows):
        for j, c in enumerate(v_cols):
            if (i, j) in shared:
                row_cols[r].add(c)
                col_rows[c].add(r)
                endpoints.add((r, c))
    return row_cols, col_rows, endpoints


def _gen_complete_axis_endpoints(point_map, line_positions, crossing_positions, endpoints, line_is_row):
    used_coord_index = 1 if line_is_row else 0
    for line in _gen_shuffle(line_positions):
        existing = sorted(point_map[line])
        if len(existing) == GEN_MAX_PER_ROW_OR_COL:
            if not _gen_span_ok(existing[0], existing[1]):
                return False
            continue
        free_positions = _gen_choose_free_positions(
            existing,
            set(crossing_positions),
            _gen_used_positions(endpoints, used_coord_index),
            set(crossing_positions),
            set(existing),
            GEN_MAX_PER_ROW_OR_COL - len(existing),
        )
        if free_positions is None:
            return False
        for pos in free_positions:
            point_map[line].add(pos)
            endpoints.add((line, pos) if line_is_row else (pos, line))
    return True


def _gen_pick_isolated_marker(endpoints):
    used_rows = _gen_used_positions(endpoints, 0)
    used_cols = _gen_used_positions(endpoints, 1)
    choices = [
        (r, c) for r in GEN_COORDS for c in GEN_COORDS
        if r not in used_rows and c not in used_cols
    ]
    if not choices:
        return None
    return choices[int(np.random.randint(len(choices)))]


def _gen_render_grids(h_rows, v_cols, endpoints, isolated):
    inp = np.full(GEN_SHAPE, GEN_BG, dtype=int)
    out = np.full(GEN_SHAPE, GEN_BG, dtype=int)

    for r, c in endpoints:
        inp[r, c] = GEN_COLOR
    inp[isolated] = GEN_COLOR
    out[isolated] = GEN_COLOR

    for r in h_rows:
        cols = sorted(c for rr, c in endpoints if rr == r)
        if len(cols) != GEN_MAX_PER_ROW_OR_COL or not _gen_span_ok(cols[0], cols[1]):
            return None, None
        out[r, cols[0]:cols[1] + 1] = GEN_COLOR

    for c in v_cols:
        rows = sorted(r for r, cc in endpoints if cc == c)
        if len(rows) != GEN_MAX_PER_ROW_OR_COL or not _gen_span_ok(rows[0], rows[1]):
            return None, None
        out[rows[0]:rows[1] + 1, c] = GEN_COLOR

    return inp, out


def _gen_valid_output(inp, out):
    markers = inp == GEN_COLOR
    marker_rows = np.count_nonzero(markers, axis=1)
    marker_cols = np.count_nonzero(markers, axis=0)
    if marker_rows.max() > GEN_MAX_PER_ROW_OR_COL or marker_cols.max() > GEN_MAX_PER_ROW_OR_COL:
        return False

    line_members = set()
    for r, n in enumerate(marker_rows):
        if n == GEN_MAX_PER_ROW_OR_COL:
            line_members.update((r, int(c)) for c in np.flatnonzero(markers[r]))
    for c, n in enumerate(marker_cols):
        if n == GEN_MAX_PER_ROW_OR_COL:
            line_members.update((int(r), c) for r in np.flatnonzero(markers[:, c]))
    isolated_markers = [
        (int(r), int(c)) for r, c in np.argwhere(markers)
        if (int(r), int(c)) not in line_members
    ]
    if len(isolated_markers) != GEN_ISOLATED_MARKERS:
        return False

    return np.array_equal(main(inp), out)


def _gen_attempt():
    h_count = int(np.random.randint(GEN_MIN_LINES_PER_AXIS, GEN_MAX_LINES_PER_AXIS + 1))
    v_count = int(np.random.randint(GEN_MIN_LINES_PER_AXIS, GEN_MAX_LINES_PER_AXIS + 1))
    shared = _gen_pick_shared(h_count, v_count)

    h_rows_need_span, v_cols_need_span = _gen_pair_span_requirements(shared, h_count, v_count)
    h_rows = _gen_pick_indices(h_count, h_rows_need_span)
    v_cols = _gen_pick_indices(v_count, v_cols_need_span)
    if h_rows is None or v_cols is None:
        return None

    row_cols, col_rows, endpoints = _gen_seed_shared_endpoints(h_rows, v_cols, shared)
    if not _gen_complete_axis_endpoints(row_cols, h_rows, v_cols, endpoints, True):
        return None
    if not _gen_complete_axis_endpoints(col_rows, v_cols, h_rows, endpoints, False):
        return None

    isolated = _gen_pick_isolated_marker(endpoints)
    if isolated is None:
        return None

    inp, out = _gen_render_grids(h_rows, v_cols, endpoints, isolated)
    if inp is None or not _gen_valid_output(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate valid ded97339 input")
