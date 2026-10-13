import numpy as np

from arc_utils import colors_present, get_bg, in_bounds
import random


def _find_cross_axes(g: np.ndarray, bg: int) -> tuple[int, int, int]:
    candidates: list[tuple[int, np.ndarray, np.ndarray]] = []
    for color in colors_present(g, exclude={bg}):
        full_rows = np.where(np.all(g == color, axis=1))[0]
        full_cols = np.where(np.all(g == color, axis=0))[0]
        if len(full_rows) > 0 and len(full_cols) > 0:
            candidates.append((int(color), full_rows, full_cols))

    assert len(candidates) == 1, "expected one full-row/full-column cross color"
    axis_color, rows, cols = candidates[0]
    assert len(rows) == 1 and len(cols) == 1, "cross must have one row axis and one column axis"
    return axis_color, int(rows[0]), int(cols[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the sparse colored marker pattern by reflecting it across a full-row/full-column cross.

    Input structure
    The grid has a dominant background and one non-background cross color that
    fills exactly one complete row and one complete column. All other foreground
    pixels are sparse markers away from that cross.

    Rule
    Keep the input unchanged, and for each marker pixel copy its color to the
    positions obtained by reflecting it across the cross row, across the cross
    column, and across both axes. Reflected positions outside the grid are
    clipped.

    Key insight
    The full cross is not a copied object; it is the pair of mirror axes for
    completing the quadrant pattern.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    axis_color, axis_r, axis_c = _find_cross_axes(g, bg)

    marker_coords = np.argwhere((g != bg) & (g != axis_color))
    assert marker_coords.size > 0, "expected sparse markers off the cross"

    out = g.copy()
    for r0, c0 in marker_coords:
        r = int(r0)
        c = int(c0)
        color = int(g[r, c])
        assert r != axis_r and c != axis_c, "marker pixels must not lie on the cross axes"

        for rr in (r, 2 * axis_r - r):
            for cc in (c, 2 * axis_c - c):
                if not in_bounds(g.shape, rr, cc):
                    continue
                existing = int(out[rr, cc])
                assert existing in (bg, color), "reflected marker pixels must not conflict"
                out[rr, cc] = color

    return out

# --------------- generator (att2) ---------------

BG = 7
PALETTE = tuple(range(10))
# Canvas and cross-axis bounds.
GRID_SIZE_RANGE = (8, 17)
AXIS_ROW_RANGE = (4, 6)
AXIS_COL_RANGE = (3, 8)
AXIS_END_MARGIN = 4
# Marker placement constraints.
QUADRANT_COUNT = 4
MARKER_COUNT_RANGE = (7, 12)
MARKER_DENSITY_RANGE = (0.25, 0.45)
# Marker color constraints.
MARKER_COLOR_RANGE = (5, 7)
MAX_MARKERS_PER_COLOR = 3
# Rejection sampling limit.
MAX_GENERATION_ATTEMPTS = 300


def _quadrant_cells(h: int, w: int, axis_r: int, axis_c: int, quadrant: int) -> list[tuple[int, int]]:
    if quadrant < 2:
        rows = range(0, axis_r)
    else:
        rows = range(axis_r + 1, h)
    if quadrant % 2 == 0:
        cols = range(0, axis_c)
    else:
        cols = range(axis_c + 1, w)
    return [(r, c) for r in rows for c in cols]


def _valid_counts(area: int) -> list[int]:
    min_count, max_count = MARKER_COUNT_RANGE
    min_density, max_density = MARKER_DENSITY_RANGE
    return [n for n in range(min_count, max_count + 1) if min_density <= n / area <= max_density]


def _color_counts(marker_count: int, color_count: int) -> list[int]:
    counts = [1] * color_count
    for _ in range(marker_count - color_count):
        expandable = [i for i, count in enumerate(counts) if count < MAX_MARKERS_PER_COLOR]
        counts[random.choice(expandable)] += 1
    random.shuffle(counts)
    return counts


def _sample_layout():
    """Sample canvas, cross axes, source quadrant cells, and marker/color counts."""
    h = random.randint(*GRID_SIZE_RANGE)
    w = random.randint(*GRID_SIZE_RANGE)
    row_lo = AXIS_ROW_RANGE[0]
    row_hi = min(AXIS_ROW_RANGE[1], h - AXIS_END_MARGIN)
    col_lo = AXIS_COL_RANGE[0]
    col_hi = min(AXIS_COL_RANGE[1], w - AXIS_END_MARGIN)
    if row_lo > row_hi or col_lo > col_hi:
        return None
    axis_r = random.randint(row_lo, row_hi)
    axis_c = random.randint(col_lo, col_hi)
    options = []
    for quadrant in range(QUADRANT_COUNT):
        cells = _quadrant_cells(h, w, axis_r, axis_c, quadrant)
        counts = _valid_counts(len(cells))
        if counts:
            options.append((quadrant, cells, counts))
    if not options:
        return None
    _, cells, possible_counts = random.choice(options)
    marker_count = random.choice(possible_counts)
    color_count = random.randint(MARKER_COLOR_RANGE[0], min(MARKER_COLOR_RANGE[1], marker_count))
    return h, w, axis_r, axis_c, cells, marker_count, color_count


def _sample_markers(cells: list[tuple[int, int]], marker_count: int, color_count: int):
    """Sample cross color, source cells, and marker color assignments."""
    cross_color = random.choice([color for color in PALETTE if color != BG])
    marker_colors = random.sample([color for color in PALETTE if color not in (BG, cross_color)], color_count)
    random.shuffle(cells)
    marker_cells = cells[:marker_count]
    assignments: list[int] = []
    for color, count in zip(marker_colors, _color_counts(marker_count, color_count)):
        assignments.extend([color] * count)
    random.shuffle(assignments)
    return cross_color, marker_cells, assignments


def _render_pair(h: int, w: int, axis_r: int, axis_c: int, cross_color: int, marker_cells, assignments):
    """Render input and oracle output, or return None when reflected colors conflict."""
    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    inp[axis_r, :] = out[axis_r, :] = cross_color
    inp[:, axis_c] = out[:, axis_c] = cross_color
    for (r, c), color in zip(marker_cells, assignments):
        inp[r, c] = color
        reflected = ((r, c), (2 * axis_r - r, c), (r, 2 * axis_c - c), (2 * axis_r - r, 2 * axis_c - c))
        for rr, cc in reflected:
            if 0 <= rr < h and 0 <= cc < w:
                if out[rr, cc] not in (BG, color):
                    return None
                out[rr, cc] = color
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    layout = _sample_layout()
    if layout is None:
        return None

    h, w, axis_r, axis_c, cells, marker_count, color_count = layout
    cross_color, marker_cells, assignments = _sample_markers(cells, marker_count, color_count)
    pair = _render_pair(h, w, axis_r, axis_c, cross_color, marker_cells, assignments)
    if pair is None:
        return None
    inp, out = pair

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
