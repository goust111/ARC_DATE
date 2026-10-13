import numpy as np

from arc_utils import Color, colors_present


def main(grid: np.ndarray) -> np.ndarray:
    """Summarize the non-black pixels as one horizontal row of their color.

    Input structure
    The input is a grid with black background and exactly one foreground color
    appearing in one or more pixels.
    Rule
    Count the non-black pixels and return a 1 x count row filled with the
    foreground color.
    Key insight
    The positions of the foreground pixels are irrelevant; only their color and
    population determine the compact output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    fg_colors = colors_present(g, exclude={Color.BLACK})
    assert len(fg_colors) == 1
    fg_color = fg_colors[0]

    fg_count = int(np.count_nonzero(g != Color.BLACK))
    assert fg_count > 0
    return np.full((1, fg_count), fg_color, dtype=int)

# --------------- generator (o5x) ---------------
# Canvas, palette, foreground count, and retry bounds.
GRID_SHAPE, CELL_COUNT = (3, 3), 9
FG_COLOR_RANGE, FG_COUNT_RANGE = (1, 10), (1, 9)
MAX_ATTEMPTS = 300

def _gen_attempt() -> np.ndarray | None:
    fg_color = int(np.random.randint(*FG_COLOR_RANGE))
    fg_count = int(np.random.randint(*FG_COUNT_RANGE))

    grid = np.zeros(GRID_SHAPE, dtype=int)
    cells = np.random.choice(CELL_COUNT, size=fg_count, replace=False)
    grid.flat[cells] = fg_color

    expected = np.full((1, fg_count), fg_color, dtype=int)
    if np.array_equal(main(grid), expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid d631b094 input")
