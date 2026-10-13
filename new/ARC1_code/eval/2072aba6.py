import numpy as np

from arc_utils import Color


def main(grid: np.ndarray) -> np.ndarray:
    """Expand each input pixel to a 2x2 block, replacing every grey pixel with a blue/red checkerboard block.

    Input structure
    A rectangular grid whose pixels are only black background and grey foreground.

    Rule
    Double the height and width. Each black pixel becomes a black 2x2 block, and each grey pixel becomes
    [[blue, red], [red, blue]] in the corresponding 2x2 output block.

    Key insight
    The output is a fixed local rewrite of each input pixel, not a movement or global pattern completion.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(map(int, np.unique(g))) <= {Color.BLACK, Color.GREY}

    h, w = g.shape
    out = np.full((2 * h, 2 * w), Color.BLACK, dtype=int)

    rr, cc = np.nonzero(g == Color.GREY)
    out[2 * rr, 2 * cc] = Color.BLUE
    out[2 * rr, 2 * cc + 1] = Color.RED
    out[2 * rr + 1, 2 * cc] = Color.RED
    out[2 * rr + 1, 2 * cc + 1] = Color.BLUE
    return out

# --------------- generator (o5x) ---------------

# Canvas bounds, foreground constraints, and retry limit.
MIN_GRID_SIZE, MAX_GRID_SIZE = 2, 5
MIN_GREY_DENSITY, MAX_GREY_DENSITY = 0.25, 0.60
MIN_GREY_PIXELS, MAX_GREY_PIXELS = 3, 5
GENERATION_RETRIES = 300


def _gen_expected_output(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    out = np.full((2 * h, 2 * w), Color.BLACK, dtype=int)
    rr, cc = np.nonzero(grid == Color.GREY)
    out[2 * rr, 2 * cc] = out[2 * rr + 1, 2 * cc + 1] = Color.BLUE
    out[2 * rr, 2 * cc + 1] = out[2 * rr + 1, 2 * cc] = Color.RED
    return out


def _gen_attempt() -> np.ndarray | None:
    h = int(np.random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE + 1))
    w = int(np.random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE + 1))
    total = h * w

    min_grey = max(MIN_GREY_PIXELS, int(np.ceil(MIN_GREY_DENSITY * total)))
    max_grey = min(MAX_GREY_PIXELS, int(np.floor(MAX_GREY_DENSITY * total)))
    if min_grey > max_grey:
        return None

    grey_count = int(np.random.randint(min_grey, max_grey + 1))
    grid = np.full((h, w), Color.BLACK, dtype=int)
    grid.flat[np.random.choice(total, size=grey_count, replace=False)] = Color.GREY

    expected = _gen_expected_output(grid)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
