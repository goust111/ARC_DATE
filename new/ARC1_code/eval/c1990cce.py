import numpy as np

from arc_utils import Color, colors_present, get_bg


DIAGONAL_PERIOD = 4


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the centered red pixel into a square red V with a periodic blue diagonal fill.

    Input structure
    - The input is a single odd-length row with black background.
    - Exactly one foreground pixel is present, and it is red at the center column.

    Rule
    - Create an N x N background canvas, where N is the input width.
    - Draw red diagonals from the top-center marker down-left and down-right until they hit the side edges.
    - Paint blue on the NW-SE diagonal lattice with the marker column's modulo-4 phase, clipped to the strict interior of the red V above the midpoint and unclipped below it.

    Key insight
    - The lone red pixel is the apex of a triangular pattern; the blue cells are one repeated diagonal family inside and below that triangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] == 1, "expected a single input row"

    bg = get_bg(g)
    assert bg == Color.BLACK, "expected black background"
    colors = colors_present(g, exclude={bg})
    assert colors == [Color.RED], "expected exactly one red foreground color"

    width = g.shape[1]
    assert width % 2 == 1, "expected odd input width"
    apex_col = width // 2
    marker_cols = np.flatnonzero(g[0] == Color.RED)
    assert marker_cols.size == 1, "expected exactly one red marker"
    assert int(marker_cols[0]) == apex_col, "expected red marker at center"

    out = np.full((width, width), bg, dtype=int)
    for r in range(width):
        if r <= apex_col:
            out[r, apex_col - r] = Color.RED
            out[r, apex_col + r] = Color.RED

        for c in range(width):
            inside_v = abs(c - apex_col) < r
            below_v = r > apex_col
            on_blue_lattice = (c - r - apex_col) % DIAGONAL_PERIOD == 0
            if (inside_v or below_v) and on_blue_lattice:
                out[r, c] = Color.BLUE

    return out

# --------------- generator (o5x) ---------------

# Width sampling.
MIN_WIDTH, MAX_WIDTH, WIDTH_STEP = 5, 29, 2
WIDTH_CHOICES = np.arange(MIN_WIDTH, MAX_WIDTH + 1, WIDTH_STEP)
# Colors and lattice phase.
GEN_BG, GEN_RED, GEN_BLUE = Color.BLACK, Color.RED, Color.BLUE
GEN_DIAGONAL_PERIOD = DIAGONAL_PERIOD
# Oracle retry limit.
GEN_MAX_ATTEMPTS = 300


def _gen_expected_output(width: int) -> np.ndarray:
    apex_col = width // 2
    out = np.full((width, width), GEN_BG, dtype=int)

    red_rows = np.arange(apex_col + 1)
    out[red_rows, apex_col - red_rows] = GEN_RED
    out[red_rows, apex_col + red_rows] = GEN_RED

    rows, cols = np.indices((width, width))
    inside_or_below_v = (np.abs(cols - apex_col) < rows) | (rows > apex_col)
    blue_lattice = (cols - rows - apex_col) % GEN_DIAGONAL_PERIOD == 0
    out[inside_or_below_v & blue_lattice] = GEN_BLUE

    return out


def _gen_attempt() -> np.ndarray | None:
    width = int(np.random.choice(WIDTH_CHOICES))
    grid = np.full((1, width), GEN_BG, dtype=int)
    grid[0, width // 2] = GEN_RED
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        inp = _gen_attempt()
        if inp is None:
            continue

        expected = _gen_expected_output(inp.shape[1])
        if np.array_equal(main(inp), expected):
            return inp

    raise RuntimeError("failed to generate a valid c1990cce input")
