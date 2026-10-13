import numpy as np

from arc_utils import Color


def main(grid: np.ndarray) -> np.ndarray:
    """Stamp the whole input pattern into output block slots selected by grey pixels.

    Input structure
    - The input is a non-empty 2D grid containing grey control pixels.
    - The output is divided into blocks with the same height and width as the input.

    Rule
    - Treat the input grid as both the source pattern and a placement mask.
    - For each grey input pixel, copy the entire input grid into the corresponding
      output block.
    - For each non-grey input pixel, leave the corresponding output block black.

    Key insight
    - Grey pixels do not expand as solid grey blocks; they select where full
      copies of the original pattern are stamped.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert np.all((0 <= g) & (g <= 9)), "expected ARC color ids"
    assert np.any(g == Color.GREY), "expected grey placement pixels"

    h, w = map(int, g.shape)
    out = np.zeros((h * h, w * w), dtype=int)
    for r in range(h):
        for c in range(w):
            if int(g[r, c]) == Color.GREY:
                out[r * h : (r + 1) * h, c * w : (c + 1) * w] = g
    return out

# --------------- generator (o5x) ---------------
# Grid, mask, and filler bounds from the observed task distribution.
GRID_SIZES = (3, 4, 5)
GREY_COUNT_MIN, GREY_COUNT_MAX = 3, 12
GREY_MIN_COVERAGE_DEN = 5
GREY_MAX_COVERAGE_NUM, GREY_MAX_COVERAGE_DEN = 3, 4
COLORED_COUNT_MIN, COLORED_COUNT_MAX = 2, 10
MIN_DISTINCT_COLORED_COLORS, MAX_BLACK_PIXELS = 2, 10

# Generation retry and palette choices.
GEN_RETRY_LIMIT = 300
COLOR_PALETTE = [c for c in Color.ALL_COLORS if c not in (Color.BLACK, Color.GREY)]


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.choice(GRID_SIZES))
    total = n * n

    grey_lo = max(GREY_COUNT_MIN, (total + GREY_MIN_COVERAGE_DEN - 1) // GREY_MIN_COVERAGE_DEN)
    grey_hi = min(GREY_COUNT_MAX, (GREY_MAX_COVERAGE_NUM * total) // GREY_MAX_COVERAGE_DEN)
    if grey_lo > grey_hi:
        return None
    grey_count = int(np.random.randint(grey_lo, grey_hi + 1))

    remaining = total - grey_count
    colored_lo = max(COLORED_COUNT_MIN, remaining - MAX_BLACK_PIXELS)
    colored_hi = min(COLORED_COUNT_MAX, remaining)
    if colored_lo > colored_hi:
        return None
    colored_count = int(np.random.randint(colored_lo, colored_hi + 1))

    grid = np.zeros((n, n), dtype=int)
    flat = grid.ravel()
    order = np.random.permutation(total)
    grey_idxs = order[:grey_count]
    colored_idxs = order[grey_count : grey_count + colored_count]

    flat[grey_idxs] = Color.GREY
    pair = np.random.choice(COLOR_PALETTE, size=MIN_DISTINCT_COLORED_COLORS, replace=False).astype(int)
    colored_values = np.empty(colored_count, dtype=int)
    colored_values[:MIN_DISTINCT_COLORED_COLORS] = pair
    if colored_count > MIN_DISTINCT_COLORED_COLORS:
        colored_values[MIN_DISTINCT_COLORED_COLORS:] = np.random.choice(
            pair, size=colored_count - MIN_DISTINCT_COLORED_COLORS
        )
    np.random.shuffle(colored_values)
    flat[colored_idxs] = colored_values

    expected = np.kron(grid == Color.GREY, grid)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid ad7e01d0 input")
