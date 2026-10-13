import numpy as np

from arc_utils import get_bg, scale_sprite
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Scale the whole input so each cell becomes a square block whose side length is the number of foreground pixels.

    Input structure
    A small grid with a uniquely dominant background color and at least one non-background pixel.

    Rule
    Count the non-background pixels, then replace every input pixel by a solid square block of that side length.

    Key insight
    The colored pixels are not moved independently; their count sets the uniform magnification factor for the entire grid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    colors, counts = np.unique(g, return_counts=True)
    max_count = int(counts.max())
    assert int(np.count_nonzero(counts == max_count)) == 1

    bg = get_bg(g)
    scale = int(np.count_nonzero(g != bg))
    assert scale > 0

    return scale_sprite(g, scale)

# --------------- generator (o5x) ---------------
# Grid geometry, colors, and retry bounds.
GRID_SIZE = 3
BG_COLOR = 0
FG_COUNT_RANGE = (2, 5)
FG_COLORS = range(1, 10)
MAX_ATTEMPTS = 300

def _gen_attempt() -> np.ndarray | None:
    n_fg = random.randint(*FG_COUNT_RANGE)
    grid = np.full((GRID_SIZE, GRID_SIZE), BG_COLOR, dtype=int)
    positions = random.sample(range(GRID_SIZE * GRID_SIZE), n_fg)
    colors = random.sample(FG_COLORS, n_fg)
    for pos, color in zip(positions, colors):
        grid[divmod(pos, GRID_SIZE)] = color
    expected = np.repeat(np.repeat(grid, n_fg, axis=0), n_fg, axis=1)
    if not np.array_equal(main(grid), expected):
        return None
    return grid

def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
