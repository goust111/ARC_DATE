import numpy as np
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Arrange the square input into a 2x2 canvas of its four quarter-turn orientations.

    Input structure:
    - The input is a nonempty square color grid.

    Rule:
    - Place the original grid in the upper-left quadrant.
    - Place the 90-degree clockwise rotation in the upper-right quadrant.
    - Place the 90-degree counterclockwise rotation in the lower-left quadrant.
    - Place the 180-degree rotation in the lower-right quadrant.

    Key insight:
    - Each output quadrant is the whole input seen from a different rotation, not a magnified copy of individual pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height > 0 and width > 0, "expected a nonempty grid"
    assert height == width, "expected a square grid"

    return np.block([
        [g, np.rot90(g, -1)],
        [np.rot90(g, 1), np.rot90(g, 2)],
    ])

# --------------- generator (o5x) ---------------

# Canvas and palette bounds.
GRID_SIZE_MIN = 2
GRID_SIZE_MAX = 5
COLOR_COUNT_MIN = 2
COLOR_COUNT_MAX = 3
COLOR_POOL = range(10)

# Retry budget.
MAX_GENERATION_ATTEMPTS = 300


def _gen_has_quarter_turn_symmetry(grid: np.ndarray) -> bool:
    return bool(np.array_equal(grid, np.rot90(grid, 1)))


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    color_count = random.randint(COLOR_COUNT_MIN, COLOR_COUNT_MAX)
    colors = random.sample(COLOR_POOL, color_count)
    grid = np.random.choice(np.array(colors, dtype=int), size=(n, n))

    marker_cells = random.sample([(r, c) for r in range(n) for c in range(n)], color_count)
    marker_colors = colors[:]
    random.shuffle(marker_colors)
    for (r, c), color in zip(marker_cells, marker_colors):
        grid[r, c] = color

    break_cells = [
        (r, c)
        for r in range(n)
        for c in range(n)
        if (r, c) != (c, n - 1 - r)
    ]
    r, c = random.choice(break_cells)
    rr, cc = c, n - 1 - r
    a, b = random.sample(colors, 2)
    grid[r, c] = a
    grid[rr, cc] = b

    protected = {(r, c), (rr, cc)}
    filler_cells = [
        (fr, fc)
        for fr in range(n)
        for fc in range(n)
        if (fr, fc) not in protected
    ]
    random.shuffle(filler_cells)
    for color in colors:
        if not np.any(grid == color):
            fr, fc = filler_cells.pop()
            grid[fr, fc] = color

    expected = np.block([
        [grid, np.rot90(grid, -1)],
        [np.rot90(grid, 1), np.rot90(grid, 2)],
    ])
    if (
        grid.shape == (n, n)
        and COLOR_COUNT_MIN <= len(set(map(int, grid.ravel()))) <= COLOR_COUNT_MAX
        and not _gen_has_quarter_turn_symmetry(grid)
        and np.array_equal(main(grid), expected)
    ):
        return grid.astype(int)
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_GENERATION_ATTEMPTS} attempts")
