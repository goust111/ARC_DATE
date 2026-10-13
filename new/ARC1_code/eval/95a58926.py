from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg


def _lattice_color(g: np.ndarray, foreground: list[int]) -> int:
    scores: dict[int, int] = {}
    for color in foreground:
        mask = g == color
        scores[int(color)] = max(int(mask.sum(axis=1).max()), int(mask.sum(axis=0).max()))

    best_score = max(scores.values())
    candidates = [color for color, score in scores.items() if score == best_score]
    assert len(candidates) == 1, "expected a unique line-like lattice color"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the clean lattice and put the marker color at every separator-row/column intersection.

    Input structure
    - The background is the dominant empty color.
    - Exactly two foreground colors appear: a lattice color and a marker/noise color.
    - True separator rows and columns contain no background pixels, although some
      lattice cells may be overwritten by the marker color.

    Rule
    - Clear all noisy marker pixels that are not at separator intersections.
    - Repaint every separator row and column with the lattice color.
    - Paint every separator-row/column intersection with the marker color.

    Key insight
    - The invariant separator rows and columns are the complete non-background
      rows/columns; scattered marker pixels are noise except where these lines cross.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    height, width = map(int, g.shape)

    bg = get_bg(g)
    foreground = colors_present(g, exclude={bg})
    assert len(foreground) == 2, "expected one lattice color and one marker color"

    line_color = _lattice_color(g, foreground)
    marker_colors = [color for color in foreground if color != line_color]
    assert len(marker_colors) == 1, "expected a unique marker color"
    marker_color = marker_colors[0]

    rows = np.flatnonzero(np.all(g != bg, axis=1))
    cols = np.flatnonzero(np.all(g != bg, axis=0))
    assert rows.size > 0 and cols.size > 0, "expected separator rows and columns"
    assert all(np.count_nonzero(g[r, :] == line_color) > width // 2 for r in rows)
    assert all(np.count_nonzero(g[:, c] == line_color) > height // 2 for c in cols)

    out = np.full_like(g, bg)
    out[rows, :] = line_color
    out[:, cols] = line_color
    out[np.ix_(rows, cols)] = marker_color
    return out

# --------------- generator (o5x) ---------------

# Canvas and lattice geometry.
GEN_GRID_SIZE_RANGE = (13, 27)
GEN_CELL_SIZE_RANGE = (3, 5)
GEN_SEP_ROW_COUNT_RANGE = (2, 5)
GEN_SEP_COL_COUNT_RANGE = (2, 6)

# Color roles.
GEN_BG_COLOR, GEN_LINE_COLOR = 0, 5
GEN_MARKER_COLORS = (1, 2, 3, 4)

# Marker-noise quotas.
GEN_TOTAL_MARKERS_RANGE = (18, 57)
GEN_INTERIOR_MARKER_FRAC = (0.05, 0.18)
GEN_LINE_MARKER_FRAC = (0.03, 0.16)
GEN_INTERSECTION_MARKER_CAP_DIVISOR = 4

# Retry limit for rejection sampling.
GEN_MAX_ATTEMPTS = 300


def _gen_pick(seq):
    return seq[int(np.random.randint(0, len(seq)))]


def _gen_axis_options(cell_size: int, count_range: tuple[int, int]):
    grid_min, grid_max = GEN_GRID_SIZE_RANGE
    step = cell_size + 1
    options = []
    for count in range(count_range[0], count_range[1] + 1):
        gaps = [gap for gap in range(cell_size + 1) if grid_min <= count * step + gap <= grid_max]
        if gaps:
            options.append((count, gaps))
    return options


def _gen_fraction_bounds(total: int, lo: float, hi: float) -> tuple[int, int]:
    return int(np.ceil(total * lo - 1e-9)), int(np.floor(total * hi + 1e-9))


def _gen_random_allocation(total: int, caps: list[int]) -> list[int] | None:
    if total < 0 or total > sum(caps):
        return None
    alloc = np.zeros(len(caps), dtype=int)
    remaining_caps = np.array(caps, dtype=int)
    for _ in range(total):
        available = np.flatnonzero(remaining_caps > 0)
        if available.size == 0:
            return None
        weights = remaining_caps[available].astype(float)
        weights /= weights.sum()
        choice = int(np.random.choice(available, p=weights))
        alloc[choice] += 1
        remaining_caps[choice] -= 1
    return alloc.tolist()


def _gen_lattice():
    """Choose lattice dimensions and separator coordinates."""
    cell_min, cell_max = GEN_CELL_SIZE_RANGE
    cell_size = int(np.random.randint(cell_min, cell_max + 1))
    row_count, row_gaps = _gen_pick(_gen_axis_options(cell_size, GEN_SEP_ROW_COUNT_RANGE))
    col_count, col_gaps = _gen_pick(_gen_axis_options(cell_size, GEN_SEP_COL_COUNT_RANGE))
    row_gap = _gen_pick(row_gaps)
    col_gap = _gen_pick(col_gaps)

    step = cell_size + 1
    height = row_count * step + row_gap
    width = col_count * step + col_gap
    sep_rows = cell_size + np.arange(row_count) * step
    sep_cols = cell_size + np.arange(col_count) * step
    return height, width, sep_rows, sep_cols


def _gen_render_output(height: int, width: int, sep_rows: np.ndarray, sep_cols: np.ndarray, marker: int):
    """Render the clean target lattice for the sampled geometry."""
    out = np.full((height, width), GEN_BG_COLOR, dtype=int)
    out[sep_rows, :] = GEN_LINE_COLOR
    out[:, sep_cols] = GEN_LINE_COLOR
    out[np.ix_(sep_rows, sep_cols)] = marker
    return out


def _gen_noise_domain_counts(
    height: int, width: int, sep_rows: np.ndarray, sep_cols: np.ndarray
) -> tuple[int, int, int]:
    """Return counts for interior cells, line cells, and intersections."""
    interior_mask = np.ones((height, width), dtype=bool)
    interior_mask[sep_rows, :] = False
    interior_mask[:, sep_cols] = False
    intersection_count = len(sep_rows) * len(sep_cols)
    line_cell_count = int(height * width - interior_mask.sum() - intersection_count)
    return int(interior_mask.sum()), line_cell_count, intersection_count


def _gen_choose_marker_counts(interior_cells: int, line_cells: int, intersections: int):
    interior_lo, interior_hi = _gen_fraction_bounds(interior_cells, *GEN_INTERIOR_MARKER_FRAC)
    line_lo, line_hi = _gen_fraction_bounds(line_cells, *GEN_LINE_MARKER_FRAC)
    intersection_hi = min(intersections, max(1, intersections // GEN_INTERSECTION_MARKER_CAP_DIVISOR))

    valid_intersections = []
    for intersection_count in range(intersection_hi + 1):
        total_lo = max(GEN_TOTAL_MARKERS_RANGE[0], interior_lo + line_lo + intersection_count)
        total_hi = min(GEN_TOTAL_MARKERS_RANGE[1], interior_hi + line_hi + intersection_count)
        if total_lo <= total_hi:
            valid_intersections.append((intersection_count, total_lo, total_hi))
    if not valid_intersections:
        return None

    intersection_count, total_lo, total_hi = _gen_pick(valid_intersections)
    total_markers = int(np.random.randint(total_lo, total_hi + 1))
    interior_min = max(interior_lo, total_markers - intersection_count - line_hi)
    interior_max = min(interior_hi, total_markers - intersection_count - line_lo)
    if interior_min > interior_max:
        return None
    interior_count = int(np.random.randint(interior_min, interior_max + 1))
    line_count = total_markers - intersection_count - interior_count
    if not (line_lo <= line_count <= line_hi):
        return None
    return interior_count, line_count, intersection_count


def _gen_place_intersections(inp: np.ndarray, sep_rows: np.ndarray, sep_cols: np.ndarray, marker: int, count: int):
    inp[np.ix_(sep_rows, sep_cols)] = GEN_LINE_COLOR
    if count == 0:
        return np.zeros((len(sep_rows), len(sep_cols)), dtype=bool)

    marker_mask = np.zeros((len(sep_rows), len(sep_cols)), dtype=bool)
    flat = np.random.choice(marker_mask.size, size=count, replace=False)
    row_ix, col_ix = np.unravel_index(flat, marker_mask.shape)
    marker_mask[row_ix, col_ix] = True
    inp[sep_rows[row_ix], sep_cols[col_ix]] = marker
    return marker_mask


def _gen_place_interior_noise(
    inp: np.ndarray,
    sep_rows: np.ndarray,
    sep_cols: np.ndarray,
    marker: int,
    count: int,
) -> bool:
    if count == 0:
        return True

    height, width = inp.shape
    sep_row_set = set(map(int, sep_rows))
    sep_col_set = set(map(int, sep_cols))
    interior_rows = np.array([r for r in range(height) if r not in sep_row_set])
    interior_cols = np.array([c for c in range(width) if c not in sep_col_set])
    row_caps = [max(0, len(interior_cols) - 1)] * len(interior_rows)
    row_alloc = _gen_random_allocation(count, row_caps)
    if row_alloc is None:
        return False

    col_caps = np.full(len(interior_cols), max(0, len(interior_rows) - 1), dtype=int)
    row_order = np.random.permutation(len(interior_rows))
    for row_i in row_order:
        take = row_alloc[int(row_i)]
        if take == 0:
            continue
        available = np.flatnonzero(col_caps > 0)
        if available.size < take:
            return False
        weights = col_caps[available].astype(float)
        weights /= weights.sum()
        chosen = np.random.choice(available, size=take, replace=False, p=weights)
        inp[int(interior_rows[row_i]), interior_cols[chosen]] = marker
        col_caps[chosen] -= 1
    return True


def _gen_place_line_noise(
    inp: np.ndarray,
    sep_rows: np.ndarray,
    sep_cols: np.ndarray,
    intersection_markers: np.ndarray,
    marker: int,
    count: int,
) -> bool:
    if count == 0:
        return True

    height, width = inp.shape
    sep_row_set = set(map(int, sep_rows))
    sep_col_set = set(map(int, sep_cols))
    non_sep_rows = np.array([r for r in range(height) if r not in sep_row_set])
    non_sep_cols = np.array([c for c in range(width) if c not in sep_col_set])

    row_marker_counts = intersection_markers.sum(axis=1).astype(int)
    col_marker_counts = intersection_markers.sum(axis=0).astype(int)
    row_caps = [max(0, width - (width // 2 + 1) - int(x)) for x in row_marker_counts]
    col_caps = [max(0, height - (height // 2 + 1) - int(x)) for x in col_marker_counts]
    row_caps = [min(cap, len(non_sep_cols)) for cap in row_caps]
    col_caps = [min(cap, len(non_sep_rows)) for cap in col_caps]

    horizontal_lo = max(0, count - sum(col_caps))
    horizontal_hi = min(count, sum(row_caps))
    if horizontal_lo > horizontal_hi:
        return False
    horizontal_count = int(np.random.randint(horizontal_lo, horizontal_hi + 1))
    vertical_count = count - horizontal_count

    row_alloc = _gen_random_allocation(horizontal_count, row_caps)
    col_alloc = _gen_random_allocation(vertical_count, col_caps)
    if row_alloc is None or col_alloc is None:
        return False

    for r, take in zip(sep_rows, row_alloc):
        if take:
            cols = np.random.choice(non_sep_cols, size=take, replace=False)
            inp[int(r), cols] = marker
    for c, take in zip(sep_cols, col_alloc):
        if take:
            rows = np.random.choice(non_sep_rows, size=take, replace=False)
            inp[rows, int(c)] = marker
    return True


def _gen_attempt() -> np.ndarray | None:
    height, width, sep_rows, sep_cols = _gen_lattice()
    marker = int(np.random.choice(GEN_MARKER_COLORS))
    out = _gen_render_output(height, width, sep_rows, sep_cols, marker)

    counts = _gen_choose_marker_counts(*_gen_noise_domain_counts(height, width, sep_rows, sep_cols))
    if counts is None:
        return None
    interior_noise_count, line_noise_count, input_intersection_count = counts

    inp = out.copy()
    intersection_markers = _gen_place_intersections(
        inp, sep_rows, sep_cols, marker, input_intersection_count
    )
    if not _gen_place_interior_noise(inp, sep_rows, sep_cols, marker, interior_noise_count):
        return None
    if not _gen_place_line_noise(inp, sep_rows, sep_cols, intersection_markers, marker, line_noise_count):
        return None

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 95a58926 input")
