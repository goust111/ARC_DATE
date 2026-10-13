import numpy as np

from arc_utils import Color, get_bg


def _single_coord(g: np.ndarray, color: int) -> tuple[int, int]:
    coords = np.argwhere(g == color)
    assert coords.shape == (1, 2), f"expected exactly one pixel of color {color}"
    return int(coords[0, 0]), int(coords[0, 1])


def _sign(n: int) -> int:
    return (n > 0) - (n < 0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect the blue marker to the red marker with a green shortest king-move path.

    Input structure
    - A mostly background grid contains exactly one blue pixel and one red pixel.
    - The marker pixels are separated in both row and column.

    Rule
    - Preserve the input pixels.
    - Starting next to the blue pixel, paint green diagonal steps toward the red
      pixel until one diagonal step remains.
    - Paint the remaining straight middle run along the longer axis, leaving the
      final diagonal step into the red pixel unpainted.

    Key insight
    - The path is a Chebyshev-shortest route whose straight segment absorbs the
      unequal row and column distances between the two markers.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert set(map(int, np.unique(g))) - {bg} == {Color.BLUE, Color.RED}, "expected only blue and red markers"

    blue_r, blue_c = _single_coord(g, Color.BLUE)
    red_r, red_c = _single_coord(g, Color.RED)

    step_r = _sign(red_r - blue_r)
    step_c = _sign(red_c - blue_c)
    row_dist = abs(red_r - blue_r)
    col_dist = abs(red_c - blue_c)
    assert row_dist > 0 and col_dist > 0, "markers must differ in both row and column"
    assert row_dist != col_dist, "path must have one longer axis"

    out = g.copy()
    r, c = blue_r, blue_c

    for _ in range(min(row_dist, col_dist) - 1):
        r += step_r
        c += step_c
        assert int(out[r, c]) == bg, "path must cross only background cells"
        out[r, c] = Color.GREEN

    if row_dist > col_dist:
        for _ in range(row_dist - col_dist):
            r += step_r
            assert int(out[r, c]) == bg, "path must cross only background cells"
            out[r, c] = Color.GREEN
    else:
        assert col_dist > row_dist
        for _ in range(col_dist - row_dist):
            c += step_c
            assert int(out[r, c]) == bg, "path must cross only background cells"
            out[r, c] = Color.GREEN

    assert (r + step_r, c + step_c) == (red_r, red_c), "path must end one diagonal step before red"
    return out

# --------------- generator (o5x) ---------------

# Canvas bounds; randint upper bounds are exclusive.
GRID_HEIGHT_RANGE, GRID_WIDTH_RANGE = (13, 17), (12, 17)
# Marker placement and distance constraints.
EDGE_BAND = 4
ROW_DIST_RANGE, COL_DIST_RANGE = (7, 13), (3, 11)
MIN_CHEBYSHEV_DIST = 9
DIFFERENT_HALF_COL_PROB = 0.75
MAX_GENERATION_ATTEMPTS = 300  # Retry budget.

def _gen_choice(seq):
    return seq[int(np.random.randint(len(seq)))]


def _gen_edge_pairs(length, dist):
    edge = set(range(min(EDGE_BAND, length))) | set(range(max(0, length - EDGE_BAND), length))
    return [(a, b) for a in edge for b in edge if abs(a - b) == dist]


def _gen_axis_pairs(length, dist):
    return [(a, a + dist) for a in range(length - dist)] + [
        (a + dist, a) for a in range(length - dist)]


def _gen_different_half(pair, length):
    return (pair[0] * 2 < length) != (pair[1] * 2 < length)


def _gen_distance_pairs(h, w):
    row_dists = [d for d in range(*ROW_DIST_RANGE) if _gen_edge_pairs(h, d)]
    col_dists = [d for d in range(*COL_DIST_RANGE) if d < w]
    return [
        (rd, cd)
        for rd in row_dists
        for cd in col_dists
        if rd != cd and max(rd, cd) >= MIN_CHEBYSHEV_DIST
    ]


def _gen_attempt():
    h = int(np.random.randint(*GRID_HEIGHT_RANGE))
    w = int(np.random.randint(*GRID_WIDTH_RANGE))

    dist_pairs = _gen_distance_pairs(h, w)
    if not dist_pairs:
        return None

    row_dist, col_dist = _gen_choice(dist_pairs)
    row_pair = _gen_choice(_gen_edge_pairs(h, row_dist))

    col_pairs = _gen_axis_pairs(w, col_dist)
    half_col_pairs = [p for p in col_pairs if _gen_different_half(p, w)]
    if half_col_pairs and np.random.random() < DIFFERENT_HALF_COL_PROB:
        col_pair = _gen_choice(half_col_pairs)
    else:
        col_pair = _gen_choice(col_pairs)

    grid = np.zeros((h, w), dtype=int)
    grid[row_pair[0], col_pair[0]] = Color.BLUE
    grid[row_pair[1], col_pair[1]] = Color.RED
    return grid


def generate_input():
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp = _gen_attempt()
        if inp is None:
            continue
        try:
            out = main(inp)
        except Exception:
            continue
        if out.shape == inp.shape:
            return inp
    raise RuntimeError("failed to generate a valid 992798f6 input")
