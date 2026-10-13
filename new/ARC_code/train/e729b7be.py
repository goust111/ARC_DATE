import numpy as np

from arc_utils import get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the scene by adding the point-reflected copy of the colored object across the central marker.

    Input structure
    - The grid has odd dimensions, a dominant background, and a center-row marker color appearing only at the left edge, center, and right edge.
    - A single divider color fills the center column except for the center marker.
    - All remaining foreground pixels form the object to copy, entirely on the left side of the divider.
    Rule
    - Keep the input unchanged.
    - For every object pixel, paint the pixel rotated 180 degrees around the center marker with the same color.
    Key insight
    - The center marker is the rotation center; the yellow-looking divider is not mirrored, it only separates the source from the empty target side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h % 2 == 1 and w % 2 == 1

    cr, cc = h // 2, w // 2
    bg = int(get_bg(g))

    marker_color = int(g[cr, cc])
    assert marker_color != bg
    marker_mask = g == marker_color
    expected_markers = np.zeros(g.shape, dtype=bool)
    expected_markers[cr, 0] = True
    expected_markers[cr, cc] = True
    expected_markers[cr, w - 1] = True
    assert np.array_equal(marker_mask, expected_markers)

    center_col = g[:, cc]
    divider_colors = {int(color) for color in np.unique(center_col) if int(color) != marker_color}
    assert len(divider_colors) == 1
    divider_color = next(iter(divider_colors))
    assert divider_color != bg
    expected_divider = np.zeros(g.shape, dtype=bool)
    expected_divider[:, cc] = True
    expected_divider[cr, cc] = False
    assert np.array_equal(g == divider_color, expected_divider)

    source_mask = (g != bg) & (g != marker_color) & (g != divider_color)
    source_coords = np.argwhere(source_mask)
    assert source_coords.size > 0
    assert np.all(source_coords[:, 1] < cc)

    out = g.copy()
    for r, c in source_coords:
        rr = 2 * cr - int(r)
        rc = 2 * cc - int(c)
        assert 0 <= rr < h and 0 <= rc < w
        assert out[rr, rc] == bg
        out[rr, rc] = g[r, c]

    return out

# --------------- generator (o5x) ---------------

# Colors and canvas bounds.
BG_COLOR, DIVIDER_COLOR, MARKER_COLOR = 7, 4, 8
GRID_SIZES, MIN_EMPTY_BORDER_ROWS = (15, 17, 19), 1

# Convex source shape bounds.
SOURCE_SPAN_RANGE = (10, 15)
SOURCE_MAX_WIDTH_RANGE = (5, 7)
SOURCE_EDGE_WIDTH_RANGE = (1, 3)
WIDTH_STEP_CAP, SOURCE_PIXEL_RANGE = 2, (35, 83)

# Color palette, band orders, and retry budget.
SOURCE_COLORS = (0, 1, 2, 3, 5, 6, 9)
SOURCE_COLOR_COUNT_RANGE, GEN_ATTEMPTS = (3, 7), 300
COLORING_KEYS = (
    lambda item: (item[2], item[3], item[1]),
    lambda item: (item[3], item[2], item[1]),
    lambda item: (item[2] + item[3], item[2], item[1]),
    lambda item: (item[2] - item[3], item[2], item[1]),
)


def _gen_bounded_steps(total, slots, cap):
    steps = []
    remaining = total
    for i in range(slots):
        left = slots - i - 1
        lo = max(0, remaining - cap * left)
        hi = min(cap, remaining)
        step = random.randint(lo, hi)
        steps.append(step)
        remaining -= step
    return steps


def _gen_width_profile(span, max_width):
    edge_width_hi = min(SOURCE_EDGE_WIDTH_RANGE[1], max_width)
    start_width = random.randint(SOURCE_EDGE_WIDTH_RANGE[0], edge_width_hi)
    end_width = random.randint(SOURCE_EDGE_WIDTH_RANGE[0], edge_width_hi)
    min_up = (max_width - start_width + WIDTH_STEP_CAP - 1) // WIDTH_STEP_CAP
    min_down = (max_width - end_width + WIDTH_STEP_CAP - 1) // WIDTH_STEP_CAP
    if min_up + min_down + 1 > span:
        return None

    peak_start = random.randint(min_up, span - 1 - min_down)
    peak_end = random.randint(peak_start, span - 1 - min_down)

    widths = [start_width]
    for step in _gen_bounded_steps(max_width - start_width, peak_start, WIDTH_STEP_CAP):
        widths.append(widths[-1] + step)

    widths.extend([max_width] * (peak_end - peak_start))

    current = max_width
    for step in _gen_bounded_steps(max_width - end_width, span - 1 - peak_end, WIDTH_STEP_CAP):
        current -= step
        widths.append(current)

    area = sum(widths)
    if not SOURCE_PIXEL_RANGE[0] <= area <= SOURCE_PIXEL_RANGE[1]:
        return None
    return widths


def _choose_source_extent(n, cr):
    valid_spans = [s for s in range(SOURCE_SPAN_RANGE[0], SOURCE_SPAN_RANGE[1] + 1) if s <= n - 2 * MIN_EMPTY_BORDER_ROWS]
    span = random.choice(valid_spans)
    r0_min = max(MIN_EMPTY_BORDER_ROWS, cr - span + 1)
    r0_max = min(cr, n - MIN_EMPTY_BORDER_ROWS - span)
    r0 = random.randint(r0_min, r0_max)
    return span, r0, cr - r0


def _paint_source_cells(out, r0, widths, outer, cc):
    interior = []
    source = []
    for i, width in enumerate(widths):
        r = r0 + i
        c0 = cc - width
        for j in range(width):
            c = c0 + j
            source.append((r, c))
            if j == 0:
                out[r, c] = outer
            else:
                interior.append((r, c, i, j))
    return source, interior


def _gen_color_source(out, r0, widths, colors, cc):
    outer = colors[0]
    inner = colors[1:]
    source, interior = _paint_source_cells(out, r0, widths, outer, cc)
    if len(interior) < len(inner):
        return None

    key = COLORING_KEYS[random.randrange(len(COLORING_KEYS))]
    interior.sort(key=key)
    band_count = len(inner)
    for rank, (r, c, _, _) in enumerate(interior):
        out[r, c] = inner[min(band_count - 1, rank * band_count // len(interior))]

    return source


def _reflect_source(out, source, cr, cc):
    reflected = []
    for r, c in source:
        rr = 2 * cr - r
        rc = 2 * cc - c
        if out[rr, rc] != BG_COLOR:
            return None
        out[rr, rc] = out[r, c]
        reflected.append((rr, rc))
    return reflected


def _gen_attempt():
    n = random.choice(GRID_SIZES)
    cr = cc = n // 2
    out = np.full((n, n), BG_COLOR, dtype=int)
    out[:, cc] = DIVIDER_COLOR
    out[cr, [0, cc, n - 1]] = MARKER_COLOR

    span, r0, center_i = _choose_source_extent(n, cr)
    max_width = random.randint(*SOURCE_MAX_WIDTH_RANGE)
    widths = _gen_width_profile(span, max_width)
    if widths is None:
        return None
    if widths[center_i] >= cc:
        return None

    color_count = random.randint(*SOURCE_COLOR_COUNT_RANGE)
    colors = random.sample(SOURCE_COLORS, color_count)
    source = _gen_color_source(out, r0, widths, colors, cc)
    if source is None:
        return None

    reflected = _reflect_source(out, source, cr, cc)
    if reflected is None:
        return None

    inp = out.copy()
    for r, c in reflected:
        inp[r, c] = BG_COLOR

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input():
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid e729b7be input")
