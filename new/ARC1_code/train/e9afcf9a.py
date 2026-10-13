import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn two solid color rows into a two-row checkerboard that alternates the row colors by column.

    Input structure:
    - The grid has exactly two rows and at least one column.
    - Each row is a solid color.
    - The two row colors are distinct.

    Rule:
    - Preserve the top/bottom color order in even-indexed columns.
    - Swap the two row colors in odd-indexed columns.

    Key insight:
    - The input gives the two checkerboard colors and the phase: column 0 keeps the original row order, then every next column flips it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height == 2, "expected exactly two rows"
    assert width > 0, "expected at least one column"

    assert np.all(g[0, :] == g[0, 0]), "top row must be solid"
    assert np.all(g[1, :] == g[1, 0]), "bottom row must be solid"
    top = int(g[0, 0])
    bottom = int(g[1, 0])
    assert top != bottom, "row colors must be distinct"

    out = np.empty_like(g)
    for c in range(width):
        if c % 2 == 0:
            out[0, c] = top
            out[1, c] = bottom
        else:
            out[0, c] = bottom
            out[1, c] = top
    return out

# --------------- generator (o5x) ---------------

# Canvas, palette, and retry bounds.
INPUT_HEIGHT = 2
WIDTH_BOUNDS = (3, 21)
COLOR_VALUES = np.arange(10)
MAX_GENERATION_ATTEMPTS = 300

def _gen_attempt() -> np.ndarray | None:
    width = int(np.random.randint(*WIDTH_BOUNDS))
    top, bottom = np.random.choice(COLOR_VALUES, size=2, replace=False)
    grid = np.empty((INPUT_HEIGHT, width), dtype=int)
    grid[0, :] = int(top)
    grid[1, :] = int(bottom)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp = _gen_attempt()
        if inp is None:
            continue

        expected = np.empty_like(inp)
        expected[:, 0::2] = inp[:, 0::2]
        expected[:, 1::2] = inp[::-1, 1::2]
        if np.array_equal(main(inp), expected):
            return inp

    raise RuntimeError("failed to generate a valid e9afcf9a input")
