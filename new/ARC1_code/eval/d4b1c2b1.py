import numpy as np

from arc_utils import scale_sprite


def main(grid: np.ndarray) -> np.ndarray:
    """
    Enlarge the whole 3x3 grid by the number of distinct colors present.

    Input structure
    The input is a 3x3 color grid, and every pixel belongs to the source pattern.

    Rule
    Count all distinct colors in the input. Replace each input pixel by a solid
    square block with side length equal to that count.

    Key insight
    The output scale factor is encoded by palette diversity, not by any specific
    color identity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape == (3, 3), "expected a 3x3 source grid"

    scale = len(np.unique(g))
    assert scale >= 1, "expected at least one color"

    return scale_sprite(g, scale)

# --------------- generator (o5x) ---------------
# Direct sampler bounds and validation budget.
GRID_SIDE, MIN_COLORS, MAX_COLORS = 3, 1, 5
NUM_CELLS, GENERATION_ATTEMPTS = GRID_SIDE * GRID_SIDE, 300
COLOR_VALUES = np.arange(1, 10)


def _sample_grid() -> np.ndarray:
    n_colors = int(np.random.randint(MIN_COLORS, MAX_COLORS + 1))
    palette = np.random.choice(COLOR_VALUES, size=n_colors, replace=False).astype(int)
    cells = np.empty(NUM_CELLS, dtype=int)
    cells[:n_colors] = palette
    if n_colors < NUM_CELLS:
        cells[n_colors:] = np.random.choice(palette, size=NUM_CELLS - n_colors, replace=True)
    np.random.shuffle(cells)
    return cells.reshape(GRID_SIDE, GRID_SIDE)


def _matches_solver(inp: np.ndarray) -> bool:
    scale = len(np.unique(inp))
    expected = np.repeat(np.repeat(inp, scale, axis=0), scale, axis=1)
    return np.array_equal(main(inp), expected)


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        inp = _sample_grid()
        if _matches_solver(inp):
            return inp

    raise RuntimeError("failed to generate a valid input")
