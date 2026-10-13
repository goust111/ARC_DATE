import numpy as np

from arc_utils import Color, bbox, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """Move the grey divider to the far edge of the left-hand mirror slot and recolor matching red markers blue.

    Input structure
    - The grid has a black background and one full-height grey vertical divider.
    - Blue marker pixels are to the left of the divider; red marker pixels are to the right.

    Rule
    - Let the divider be at column k; erase it and draw it at column 2*k+1.
    - Reflect the horizontal span of the left blue markers across the original divider.
    - If red markers between the old and new divider occupy exactly that reflected column span, recolor those red pixels blue.

    Key insight
    - The new divider creates a mirror slot with the same width as the left partition; only the red mask whose column span matches the reflected blue span is converted.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    assert bg == Color.BLACK

    grey_cols = np.flatnonzero(np.all(g == Color.GREY, axis=0))
    assert len(grey_cols) == 1
    sep = int(grey_cols[0])
    assert np.count_nonzero(g == Color.GREY) == h

    new_sep = 2 * sep + 1
    assert new_sep < w

    blue_box = bbox(g[:, :sep] == Color.BLUE)
    assert blue_box is not None
    _, min_blue_col, _, max_blue_col = blue_box

    span_left = 2 * sep - max_blue_col
    span_right = 2 * sep - min_blue_col
    assert sep < span_left <= span_right < new_sep

    out = g.copy()
    out[:, sep] = bg
    out[:, new_sep] = Color.GREY

    red_slot = g[:, sep + 1 : new_sep] == Color.RED
    rel_red_cols = np.flatnonzero(np.any(red_slot, axis=0))
    if rel_red_cols.size:
        red_cols = rel_red_cols + sep + 1
        assert np.array_equal(red_cols, np.arange(red_cols[0], red_cols[-1] + 1))
        if int(red_cols[0]) == span_left and int(red_cols[-1]) == span_right:
            cols = np.arange(w)[None, :]
            out[(g == Color.RED) & (span_left <= cols) & (cols <= span_right)] = Color.BLUE

    return out

# --------------- generator (o5x) ---------------

# Canvas and divider geometry.
GRID_SHAPE = (7, 15)
DIVIDER_COL_RANGE = (1, 6)
# Blue marker sampling.
BLUE_COUNT_RANGE = (1, 6)
BLUE_WIDE_SPAN_PROB = 0.85
MAX_MARKERS_PER_ROW = 2
# Red mirror-slot sampling.
MATCHING_MIRROR_PROB = 0.6
MIRROR_MAX_TOTAL_RED = 7
NONMATCH_MIRROR_MAX_WIDTH = 2
MIRROR_SMALL_COL_CAP, MIRROR_LARGE_COL_CAP = 2, 3
MIRROR_SMALL_COL_LIMIT = 2
# Red markers beyond the moved divider.
BEYOND_RED_RANGE = (2, 13)
# Placement scoring and retry limits.
ORTHOGONAL_NEIGHBOR_PENALTY, DIAGONAL_NEIGHBOR_PENALTY = 4, 1
SPACING_SCORE_SCALE = 10
GEN_MAX_ATTEMPTS = 300

_gen_rng = np.random.default_rng()


def _gen_choice(items):
    return items[int(_gen_rng.integers(0, len(items)))]


def _gen_add_cell(cells, used, row_counts, r, c):
    r, c = int(r), int(c)
    if (r, c) in used or row_counts[r] >= MAX_MARKERS_PER_ROW:
        return False
    cells.append((r, c))
    used.add((r, c))
    row_counts[r] += 1
    return True


def _gen_blue_cells(k, n_blue):
    if n_blue == 1 or k == 1:
        max_blue_col = 0
    elif float(_gen_rng.random()) < BLUE_WIDE_SPAN_PROB:
        max_blue_col = int(_gen_rng.integers(1, k))
    else:
        max_blue_col = 0
    required_cols = [0] if max_blue_col == 0 else [0, max_blue_col]
    cells = []
    used = set()
    row_counts = [0] * GRID_SHAPE[0]

    for col in required_cols:
        rows = [int(r) for r in _gen_rng.permutation(GRID_SHAPE[0])]
        candidates = [r for r in rows if row_counts[r] < MAX_MARKERS_PER_ROW and (r, col) not in used]
        unused_rows = [r for r in candidates if row_counts[r] == 0]
        pool = [(r, col) for r in (unused_rows or candidates)]
        best = min(_gen_neighbor_score(cell, used) for cell in pool)
        r, c = _gen_choice([cell for cell in pool if _gen_neighbor_score(cell, used) == best])
        if not _gen_add_cell(cells, used, row_counts, r, c):
            return None

    while len(cells) < n_blue:
        available = [
            (r, c)
            for c in range(max_blue_col + 1)
            for r in range(GRID_SHAPE[0])
            if (r, c) not in used and row_counts[r] < MAX_MARKERS_PER_ROW
        ]
        if not available:
            return None
        scores = [
            _gen_neighbor_score((r, c), used) * SPACING_SCORE_SCALE + row_counts[r]
            for r, c in available
        ]
        best = min(scores)
        pool = [cell for cell, score in zip(available, scores) if score == best]
        r, c = _gen_choice(pool)
        if not _gen_add_cell(cells, used, row_counts, r, c):
            return None

    return cells


def _gen_paint_red_columns(grid, cols, max_total=7):
    cols = [int(c) for c in cols]
    if not cols:
        return
    cells = set()
    col_counts = {c: 0 for c in cols}
    for col in cols:
        available = [(r, col) for r in range(GRID_SHAPE[0])]
        scores = [_gen_neighbor_score(cell, cells) for cell in available]
        best = min(scores)
        cell = _gen_choice([cell for cell, score in zip(available, scores) if score == best])
        cells.add(cell)
        col_counts[col] += 1

    per_col_cap = MIRROR_SMALL_COL_CAP if len(cols) <= MIRROR_SMALL_COL_LIMIT else MIRROR_LARGE_COL_CAP
    extra_capacity = min(max_total, per_col_cap * len(cols)) - len(cells)
    if extra_capacity > 0:
        extra_count = int(_gen_rng.integers(0, extra_capacity + 1))
        for _ in range(extra_count):
            available = [
                (r, c)
                for c in cols
                for r in range(GRID_SHAPE[0])
                if (r, c) not in cells and col_counts[c] < per_col_cap
            ]
            if not available:
                break
            scores = [
                _gen_neighbor_score(cell, cells) * SPACING_SCORE_SCALE + col_counts[cell[1]]
                for cell in available
            ]
            best = min(scores)
            cell = _gen_choice([cell for cell, score in zip(available, scores) if score == best])
            cells.add(cell)
            col_counts[cell[1]] += 1

    for r, c in cells:
        grid[r, c] = Color.RED


def _gen_nonmatching_mirror_cols(k, span_left, span_right):
    options = [()]
    for width in range(1, min(NONMATCH_MIRROR_MAX_WIDTH, k) + 1):
        for start in range(k + 1, 2 * k - width + 2):
            end = start + width - 1
            if start == span_left and end == span_right:
                continue
            options.append(tuple(range(start, end + 1)))
    return _gen_choice(options)


def _gen_neighbor_score(cell, used):
    r, c = cell
    score = 0
    for rr, cc in used:
        dr, dc = abs(r - rr), abs(c - cc)
        if dr + dc == 1:
            score += ORTHOGONAL_NEIGHBOR_PENALTY
        elif max(dr, dc) == 1:
            score += DIAGONAL_NEIGHBOR_PENALTY
    return score


def _gen_paint_spread_red(grid, cols, n_red):
    cols = [int(c) for c in cols]
    used = set()
    for _ in range(n_red):
        available = [(r, c) for c in cols for r in range(GRID_SHAPE[0]) if (r, c) not in used]
        if not available:
            return False
        scores = [_gen_neighbor_score(cell, used) for cell in available]
        best = min(scores)
        pool = [cell for cell, score in zip(available, scores) if score == best]
        r, c = _gen_choice(pool)
        used.add((r, c))
        grid[r, c] = Color.RED
    return True


def _gen_place_blue_markers(grid, k):
    """Paint blue markers and return their reflected mirror span."""
    n_blue = int(_gen_rng.integers(*BLUE_COUNT_RANGE))
    blue_cells = _gen_blue_cells(k, n_blue)
    if blue_cells is None:
        return None
    for r, c in blue_cells:
        grid[r, c] = Color.BLUE

    blue_cols = [c for _, c in blue_cells]
    span_left = 2 * k - max(blue_cols)
    span_right = 2 * k - min(blue_cols)
    return span_left, span_right


def _gen_place_mirror_red(grid, k, span_left, span_right):
    """Paint matching or intentionally nonmatching red columns in the mirror slot."""
    if float(_gen_rng.random()) < MATCHING_MIRROR_PROB:
        mirror_cols = range(span_left, span_right + 1)
    else:
        mirror_cols = _gen_nonmatching_mirror_cols(k, span_left, span_right)
    _gen_paint_red_columns(grid, mirror_cols, max_total=MIRROR_MAX_TOTAL_RED)


def _gen_place_beyond_red(grid, k):
    """Paint the distractor red markers to the right of the new divider."""
    beyond_cols = range(2 * k + 2, GRID_SHAPE[1])
    n_beyond_red = int(_gen_rng.integers(*BEYOND_RED_RANGE))
    return _gen_paint_spread_red(grid, beyond_cols, n_beyond_red)


def _gen_attempt():
    k = int(_gen_rng.integers(*DIVIDER_COL_RANGE))
    grid = np.zeros(GRID_SHAPE, dtype=int)
    grid[:, k] = Color.GREY
    mirror_span = _gen_place_blue_markers(grid, k)
    if mirror_span is None:
        return None
    _gen_place_mirror_red(grid, k, *mirror_span)
    if not _gen_place_beyond_red(grid, k):
        return None

    return grid


def generate_input():
    for _ in range(GEN_MAX_ATTEMPTS):
        inp = _gen_attempt()
        if inp is None:
            continue
        try:
            main(inp)
        except Exception:
            continue
        return inp
    raise RuntimeError("failed to generate valid input for dd2401ed")
