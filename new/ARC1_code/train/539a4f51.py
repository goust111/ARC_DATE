import numpy as np

from arc_utils import Color
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a diagonal-index color table by cycling its non-black header sequence.

    Input structure
    - The input is a nonempty square grid.
    - The top row and left column are the same header sequence.
    - Every cell is the header color at index max(row, column); black header cells
      appear only as a trailing blank suffix.

    Rule
    - Drop the trailing black blanks from the header.
    - Repeat the remaining header colors until it is twice as long as the input side.
    - Fill the doubled square so each cell again takes the repeated header color
      at index max(row, column).

    Key insight
    - The visible pattern is not a 2D tile; it is a one-dimensional header sequence
      projected through the max(row, column) diagonal-index rule.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height == width and height > 0, "expected a nonempty square table"

    header = g[0, :].copy()
    assert np.array_equal(g[:, 0], header), "expected matching top row and left column"

    if np.any(header == Color.BLACK):
        first_blank = int(np.argmax(header == Color.BLACK))
        assert np.all(header[first_blank:] == Color.BLACK), "black header cells must be trailing blanks"
        active_header = header[:first_blank]
    else:
        active_header = header
    assert active_header.size > 0, "expected a nonempty active header"
    assert np.all(active_header != Color.BLACK), "active header must be non-black"

    in_index = np.maximum.outer(np.arange(height), np.arange(width))
    assert np.array_equal(g, header[in_index]), "expected a diagonal-index table"

    out_size = height * 2
    repeated_header = np.resize(active_header, out_size)
    out_index = np.maximum.outer(np.arange(out_size), np.arange(out_size))
    return repeated_header[out_index]

# --------------- generator (o5x) ---------------
# Sampling bounds and retry budget.
GRID_SIDE_RANGE = (4, 7)
TRAILING_BLACK_RANGE = (0, 1)
DISTINCT_COLOR_RANGE = (2, 5)
NON_BLACK_COLORS = range(1, 10)
GENERATION_ATTEMPTS = 300

def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*GRID_SIDE_RANGE)
    trailing_blacks = random.randint(*TRAILING_BLACK_RANGE)
    active_len = n - trailing_blacks

    num_distinct = random.randint(DISTINCT_COLOR_RANGE[0], min(DISTINCT_COLOR_RANGE[1], active_len))
    palette = random.sample(NON_BLACK_COLORS, num_distinct)

    active_header = palette[:]
    active_header.extend(random.choice(palette) for _ in range(active_len - num_distinct))
    random.shuffle(active_header)

    header = np.array(active_header + [Color.BLACK] * trailing_blacks, dtype=int)
    index = np.maximum.outer(np.arange(n), np.arange(n))
    return header[index]


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is None:
            continue
        try:
            main(grid)
        except Exception:
            continue
        return grid
    raise RuntimeError("failed to generate valid input")
