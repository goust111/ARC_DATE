import numpy as np

from arc_utils import Color, get_bg


PALETTE_ORDER = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW)


def main(grid: np.ndarray) -> np.ndarray:
    """Collect every non-background pixel into a bottom-aligned histogram column in fixed palette order.

    Input structure
    - A black-background grid contains exactly the four foreground colors blue,
      red, green, and yellow as scattered pixels.
    - The grid is tall enough to hold each color's pixel count as one vertical bar.

    Rule
    - Count pixels of each foreground color.
    - Clear the canvas to background, then assign the colors to the leftmost four
      columns in the palette order blue, red, green, yellow.
    - Fill each assigned column upward from the bottom with as many cells as that
      color's input pixel count.

    Key insight
    - Positions of the original pixels are irrelevant; the output preserves only
      per-color pixel counts, rendered as a compact bottom-left bar chart.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    assert bg == Color.BLACK, "background must be black"

    palette = set(PALETTE_ORDER)
    assert set(map(int, np.unique(g))) == palette | {bg}

    height, width = g.shape
    assert width >= len(PALETTE_ORDER)

    counts = [int(np.count_nonzero(g == color)) for color in PALETTE_ORDER]
    assert all(0 < count <= height for count in counts)

    out = np.full_like(g, bg)
    for col, (color, count) in enumerate(zip(PALETTE_ORDER, counts)):
        out[height - count : height, col] = color

    return out

# --------------- generator (o5x) ---------------

# Generator configuration: canvas/palette, count bounds, and retry budget.
_GRID_SHAPE = (10, 10)
_GEN_COLORS = (1, 2, 3, 4)
_COUNT_MIN = 2
_COUNT_MAX_EXCLUSIVE = 9
_MAX_ATTEMPTS = 300


def _gen_attempt() -> np.ndarray | None:
    grid = np.zeros(_GRID_SHAPE, dtype=int)
    counts = np.random.randint(_COUNT_MIN, _COUNT_MAX_EXCLUSIVE, size=len(_GEN_COLORS))
    cells = np.random.permutation(grid.size)[: int(counts.sum())]

    start = 0
    for color, count in zip(_GEN_COLORS, counts):
        chosen = cells[start : start + int(count)]
        grid.flat[chosen] = color
        start += int(count)

    expected = np.zeros_like(grid)
    for col, (color, count) in enumerate(zip(_GEN_COLORS, counts)):
        expected[grid.shape[0] - int(count) :, col] = color

    if np.array_equal(main(grid), expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
