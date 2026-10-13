import numpy as np

from arc_utils import colors_present, get_bg, scale_sprite
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Enlarge the whole 3x3 grid by the number of distinct foreground colors.

    Input structure
    The input is a 3x3 grid with one unique dominant background color and at
    least one foreground color.

    Rule
    Count the distinct non-background colors. Replace every input pixel with a
    square block of that side length, preserving the pixel's color.

    Key insight
    The output scale is not fixed; it is encoded by the number of different
    foreground colors in the small input pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape == (3, 3), "expected the task's 3x3 input grid"

    bg = get_bg(g)
    vals, counts = np.unique(g, return_counts=True)
    assert np.count_nonzero(counts == counts.max()) == 1, "expected a unique dominant background"
    assert int(counts[vals == bg][0]) == int(counts.max()), "background must be the dominant color"

    foreground_colors = colors_present(g, exclude={bg})
    assert foreground_colors, "expected at least one foreground color"
    return scale_sprite(g, len(foreground_colors))

# --------------- generator (o5x) ---------------
# Canvas and color domain.
GRID_SHAPE, GRID_CELLS = (3, 3), 9
FOREGROUND_COLOR_POOL = range(1, 10)
# Foreground structure.
FOREGROUND_COLOR_BOUNDS, FOREGROUND_CELL_BOUNDS = (2, 4), (4, 5)
MAX_CELLS_PER_FOREGROUND_COLOR = 3
# Retry policy.
MAX_GENERATION_ATTEMPTS = 300


def _draw_color_counts(n_fg: int, n_cells: int) -> list[int]:
    max_color_count = min(MAX_CELLS_PER_FOREGROUND_COLOR, GRID_CELLS - n_cells - 1)
    color_counts = [1] * n_fg
    for _ in range(n_cells - n_fg):
        eligible = [i for i, count in enumerate(color_counts) if count < max_color_count]
        color_counts[random.choice(eligible)] += 1
    return color_counts


def _render_grid(colors: list[int], color_counts: list[int], positions: list[int]) -> np.ndarray:
    assigned = []
    for color, count in zip(colors, color_counts):
        assigned.extend([color] * count)
    random.shuffle(assigned)
    grid = np.zeros(GRID_CELLS, dtype=int)
    grid[positions] = assigned
    return grid.reshape(GRID_SHAPE)


def _gen_attempt() -> np.ndarray | None:
    n_fg, n_cells = random.randint(*FOREGROUND_COLOR_BOUNDS), random.randint(*FOREGROUND_CELL_BOUNDS)
    colors, positions = random.sample(FOREGROUND_COLOR_POOL, n_fg), random.sample(range(GRID_CELLS), n_cells)
    grid = _render_grid(colors, _draw_color_counts(n_fg, n_cells), positions)
    out = np.kron(grid, np.ones((n_fg, n_fg), dtype=int))
    return grid if np.array_equal(main(grid), out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
