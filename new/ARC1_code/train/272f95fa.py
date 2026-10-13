import numpy as np
from arc_utils import Color, get_bg
import random


def _bands(size: int, separators: list[int]) -> list[tuple[int, int]]:
    bands = []
    start = 0
    for cut in separators:
        assert start < cut, "separator lines must leave nonempty regions"
        bands.append((start, cut - 1))
        start = cut + 1
    assert start < size, "separator lines must leave a trailing region"
    bands.append((start, size - 1))
    return bands


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the plus-shaped five cells of a separator-defined 3x3 lattice with fixed colors.

    Input structure
    - The grid contains a dominant background and one separator color.
    - The separator color forms exactly two full-width rows and two full-height columns, dividing the grid into nine nonempty rectangular regions.

    Rule
    - Keep the separator bars and corner regions unchanged.
    - Fill the top-middle region red, middle-left yellow, center pink, middle-right green, and bottom-middle blue.

    Key insight
    - The separator lattice defines relative region roles; the colored output is the fixed plus shape centered in that 3x3 partition.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    row_colors = {
        int(g[r, 0])
        for r in range(h)
        if np.all(g[r, :] == g[r, 0]) and int(g[r, 0]) != bg
    }
    col_colors = {
        int(g[0, c])
        for c in range(w)
        if np.all(g[:, c] == g[0, c]) and int(g[0, c]) != bg
    }
    sep_colors = row_colors & col_colors
    assert len(sep_colors) == 1, "there must be exactly one full-row/full-column separator color"
    sep = next(iter(sep_colors))
    assert set(map(int, np.unique(g))) <= {bg, sep}, "input must contain only background and separator colors"

    sep_rows = [r for r in range(h) if np.all(g[r, :] == sep)]
    sep_cols = [c for c in range(w) if np.all(g[:, c] == sep)]
    assert len(sep_rows) == 2, "there must be exactly two separator rows"
    assert len(sep_cols) == 2, "there must be exactly two separator columns"

    row_bands = _bands(h, sep_rows)
    col_bands = _bands(w, sep_cols)
    assert len(row_bands) == 3 and len(col_bands) == 3, "separator lattice must define a 3x3 partition"

    out = g.copy()
    fills = {
        (0, 1): Color.RED,
        (1, 0): Color.YELLOW,
        (1, 1): Color.PINK,
        (1, 2): Color.GREEN,
        (2, 1): Color.BLUE,
    }
    for (ri, ci), color in fills.items():
        r0, r1 = row_bands[ri]
        c0, c1 = col_bands[ci]
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == bg), "filled lattice regions must start as background"
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out

# --------------- generator (o5x) ---------------

# Canvas dimension and region-length sampling bounds.
GRID_HEIGHT_RANGE = (12, 19)
GRID_WIDTH_RANGE = (14, 19)
BAND_SIZE_RANGE = (2, 10)
SEPARATOR_WIDTH = 1

# Task palette and fixed output colors.
BACKGROUND_COLOR, SEPARATOR_COLOR = Color.BLACK, Color.TEAL
PLUS_REGION_FILLS = {
    (0, 1): Color.RED,
    (1, 0): Color.YELLOW,
    (1, 1): Color.PINK,
    (1, 2): Color.GREEN,
    (2, 1): Color.BLUE,
}

# Rejection-sampling retry budget.
GENERATION_RETRIES = 300


def _gen_band_lengths(total_size: int) -> tuple[int, int, int]:
    target = total_size - 2 * SEPARATOR_WIDTH
    min_band, max_band = BAND_SIZE_RANGE
    valid = [
        (a, b, target - a - b)
        for a in range(min_band, max_band + 1)
        for b in range(min_band, max_band + 1)
        if min_band <= target - a - b <= max_band
    ]
    return random.choice(valid)


def _gen_separators(bands: tuple[int, int, int]) -> tuple[int, int]:
    return bands[0], bands[0] + SEPARATOR_WIDTH + bands[1]


def _gen_expected_output(
    inp: np.ndarray,
    row_bands: tuple[int, int, int],
    col_bands: tuple[int, int, int],
) -> np.ndarray:
    row_starts = [0, row_bands[0] + SEPARATOR_WIDTH, row_bands[0] + row_bands[1] + 2 * SEPARATOR_WIDTH]
    col_starts = [0, col_bands[0] + SEPARATOR_WIDTH, col_bands[0] + col_bands[1] + 2 * SEPARATOR_WIDTH]
    out = inp.copy()
    for (ri, ci), color in PLUS_REGION_FILLS.items():
        r0 = row_starts[ri]
        c0 = col_starts[ci]
        out[r0 : r0 + row_bands[ri], c0 : c0 + col_bands[ci]] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    row_bands = _gen_band_lengths(h)
    col_bands = _gen_band_lengths(w)
    sep_rows = _gen_separators(row_bands)
    sep_cols = _gen_separators(col_bands)

    inp = np.full((h, w), BACKGROUND_COLOR, dtype=int)
    inp[list(sep_rows), :] = SEPARATOR_COLOR
    inp[:, list(sep_cols)] = SEPARATOR_COLOR

    expected = _gen_expected_output(inp, row_bands, col_bands)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for 272f95fa")
