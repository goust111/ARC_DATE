import numpy as np

from arc_utils import colors_present, get_bg, in_bounds


def _full_row_bars(g: np.ndarray, bg: int) -> list[tuple[int, int]]:
    rows = []
    for r in range(g.shape[0]):
        vals = set(map(int, g[r, :]))
        if len(vals) == 1:
            color = vals.pop()
            if color != bg:
                rows.append((r, color))
    return rows


def _full_col_bars(g: np.ndarray, bg: int) -> list[tuple[int, int]]:
    cols = []
    for c in range(g.shape[1]):
        vals = set(map(int, g[:, c]))
        if len(vals) == 1:
            color = vals.pop()
            if color != bg:
                cols.append((c, color))
    return cols


def _marker_color(g: np.ndarray, bg: int, bars: list[tuple[int, int]]) -> int:
    bar_colors = {color for _, color in bars}
    markers = colors_present(g, exclude={bg, *bar_colors})
    assert len(markers) == 1
    return markers[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint one cell from each marker toward its nearer colored edge bar.

    Input structure
    - The background is the dominant color.
    - Two complete, opposite edge bars appear as either full rows or full columns.
    - The remaining foreground color marks isolated control pixels between the bars.

    Rule
    - For each control pixel, compare its distance to the two edge bars along the bar-to-bar axis.
    - Paint the adjacent background cell one step toward the nearer bar.
    - Use the nearer bar's color for that new cell and leave all existing pixels unchanged.

    Key insight
    - The full edge bars both choose the direction and supply the color for each marker's one-cell extension.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = get_bg(g)

    row_bars = _full_row_bars(g, bg)
    col_bars = _full_col_bars(g, bg)
    out = g.copy()

    if row_bars:
        assert len(row_bars) == 2
        assert not col_bars
        assert {r for r, _ in row_bars} == {0, H - 1}
        bars = sorted(row_bars)
        marker = _marker_color(g, bg, bars)

        for r0, c0 in np.argwhere(g == marker):
            r = int(r0)
            c = int(c0)
            top_dist = abs(r - bars[0][0])
            bottom_dist = abs(r - bars[1][0])
            assert top_dist != bottom_dist
            if top_dist < bottom_dist:
                nr, nc = r - 1, c
                color = bars[0][1]
            else:
                nr, nc = r + 1, c
                color = bars[1][1]
            assert in_bounds(g.shape, nr, nc)
            assert g[nr, nc] == bg
            out[nr, nc] = color

        return out

    assert len(col_bars) == 2
    assert {c for c, _ in col_bars} == {0, W - 1}
    bars = sorted(col_bars)
    marker = _marker_color(g, bg, bars)

    for r0, c0 in np.argwhere(g == marker):
        r = int(r0)
        c = int(c0)
        left_dist = abs(c - bars[0][0])
        right_dist = abs(c - bars[1][0])
        assert left_dist != right_dist
        if left_dist < right_dist:
            nr, nc = r, c - 1
            color = bars[0][1]
        else:
            nr, nc = r, c + 1
            color = bars[1][1]
        assert in_bounds(g.shape, nr, nc)
        assert g[nr, nc] == bg
        out[nr, nc] = color

    return out

# --------------- generator (o5x) ---------------
# Canvas, colors, and retry limits.
ORIENTATION_ROWS = "rows"
ORIENTATIONS = (ORIENTATION_ROWS, "cols")
AXIS_SIZE_CHOICES = [10, 12, 14, 16, 18]
CROSS_SIZE_RANGE = (10, 19)
BG_COLOR = 0
MARKER_COLOR = 1
BAR_COLOR_CHOICES = np.arange(2, 10)
GEN_RETRY_LIMIT = 300

# Marker placement.
MARKER_COUNT_RANGE = (6, 13)
BAR_CLEARANCE = 2
NEIGHBOR_DELTAS = (-1, 0, 1)


def _gen_paint_cell(r, c, orientation, H, W, color_a, color_b):
    if orientation == ORIENTATION_ROWS:
        top_dist = r
        bottom_dist = H - 1 - r
        if top_dist == bottom_dist:
            return None
        if top_dist < bottom_dist:
            return (r - 1, c), color_a
        return (r + 1, c), color_b

    left_dist = c
    right_dist = W - 1 - c
    if left_dist == right_dist:
        return None
    if left_dist < right_dist:
        return (r, c - 1), color_a
    return (r, c + 1), color_b


def _gen_forbid_neighbors(available, r, c):
    for dr in NEIGHBOR_DELTAS:
        for dc in NEIGHBOR_DELTAS:
            available.discard((r + dr, c + dc))


def _gen_expected_output(g, orientation, color_a, color_b, marker_color):
    H, W = g.shape
    out = g.copy()
    for r0, c0 in np.argwhere(g == marker_color):
        painted = _gen_paint_cell(int(r0), int(c0), orientation, H, W, color_a, color_b)
        if painted is None:
            return None
        (nr, nc), color = painted
        if g[nr, nc] != BG_COLOR:
            return None
        out[nr, nc] = color
    return out


def _gen_seed_layout(rng, orientation):
    """Create the empty barred grid and candidate marker cells."""
    axis_size = int(rng.choice(AXIS_SIZE_CHOICES))
    cross_size = int(rng.integers(*CROSS_SIZE_RANGE))
    H, W = (axis_size, cross_size) if orientation == ORIENTATION_ROWS else (cross_size, axis_size)
    color_a, color_b = map(int, rng.choice(BAR_COLOR_CHOICES, size=2, replace=False))
    g = np.zeros((H, W), dtype=int)
    if orientation == ORIENTATION_ROWS:
        g[0, :] = color_a
        g[H - 1, :] = color_b
        available = {(r, c) for r in range(BAR_CLEARANCE, H - BAR_CLEARANCE) for c in range(W)}
    else:
        g[:, 0] = color_a
        g[:, W - 1] = color_b
        available = {(r, c) for r in range(H) for c in range(BAR_CLEARANCE, W - BAR_CLEARANCE)}

    return g, color_a, color_b, available


def _gen_attempt():
    rng = np.random.default_rng()
    orientation = ORIENTATIONS[int(rng.integers(len(ORIENTATIONS)))]
    g, color_a, color_b, available = _gen_seed_layout(rng, orientation)
    H, W = g.shape
    marker_count = int(rng.integers(*MARKER_COUNT_RANGE))
    markers = set()
    targets = set()

    for _ in range(marker_count):
        valid = []
        for r, c in available:
            painted = _gen_paint_cell(r, c, orientation, H, W, color_a, color_b)
            if painted is None:
                continue
            target, _ = painted
            if target not in markers and target not in targets:
                valid.append((r, c))
        if not valid:
            return None

        r, c = valid[int(rng.integers(len(valid)))]
        target, _ = _gen_paint_cell(r, c, orientation, H, W, color_a, color_b)
        markers.add((r, c))
        targets.add(target)
        g[r, c] = MARKER_COLOR
        _gen_forbid_neighbors(available, r, c)
        available.discard(target)

    expected = _gen_expected_output(g, orientation, color_a, color_b, MARKER_COLOR)
    if expected is None:
        return None
    try:
        actual = main(g)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return g


def generate_input():
    for _ in range(GEN_RETRY_LIMIT):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("failed to generate valid 642248e4 input")
