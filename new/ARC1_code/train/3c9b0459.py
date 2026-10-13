import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate the entire input grid 180 degrees.

    Input structure
    - The input is a 3x3 grid whose pixels are all part of the pattern.

    Rule
    - Map each pixel to the position opposite it through the grid center.

    Key insight
    - Color counts and shape are preserved; only the global orientation changes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape == (3, 3), "expected the task's 3x3 grid"

    return np.rot90(g, 2).copy()

# --------------- generator (o5x) ---------------
# Canvas/palette constraints.
GRID_SHAPE = (3, 3)
COLOR_COUNT_RANGE = (2, 4)
NON_BLACK_COLORS = np.arange(1, 10)
# Half-turn validation and retry controls.
HALF_TURN_K = 2
ROTATION_PAIRS = (((0, 0), (2, 2)), ((0, 1), (2, 1)), ((0, 2), (2, 0)), ((1, 0), (1, 2)))
MAX_ATTEMPTS = 300

def _gen_attempt() -> np.ndarray | None:
    num_colors = int(np.random.randint(*COLOR_COUNT_RANGE))
    palette = np.random.choice(NON_BLACK_COLORS, size=num_colors, replace=False)

    extra = np.random.choice(palette, size=GRID_SHAPE[0] * GRID_SHAPE[1] - num_colors, replace=True)
    cells = list(palette) + extra.tolist()
    np.random.shuffle(cells)
    grid = np.array(cells, dtype=int).reshape(GRID_SHAPE)

    if all(grid[a] == grid[b] for a, b in ROTATION_PAIRS):
        a, b = ROTATION_PAIRS[int(np.random.randint(len(ROTATION_PAIRS)))]
        choices = palette[palette != grid[a]]
        grid[b] = int(np.random.choice(choices))

    out = np.rot90(grid, HALF_TURN_K).copy()
    if not np.array_equal(main(grid), out):
        return None
    if np.array_equal(grid, out):
        return None
    return grid

def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
