import numpy as np
import random


def _repeat_2x2(sprite: np.ndarray) -> np.ndarray:
    return np.tile(sprite, (2, 2))


def main(grid: np.ndarray) -> np.ndarray:
    """Build a four-quadrant rotational tiling of the input square.

    Input structure
    - The input is a square grid treated as one complete sprite.

    Rule
    - Each output quadrant is a 2x2 repetition of a rotated copy of the whole input.
    - The quadrant rotations are upper-left 180 degrees, upper-right 90 degrees
      clockwise, lower-left 90 degrees counterclockwise, and lower-right unchanged.

    Key insight
    - The output is a 4x4 grid of original-size blocks, not a per-pixel scale-up.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w and h > 0

    upper_left = _repeat_2x2(np.rot90(g, 2))
    upper_right = _repeat_2x2(np.rot90(g, -1))
    lower_left = _repeat_2x2(np.rot90(g, 1))
    lower_right = _repeat_2x2(g)

    return np.block([[upper_left, upper_right], [lower_left, lower_right]])

# --------------- generator (att2) ---------------

# Grid and palette bounds.
_GEN_CF5FD0AD_BG_COLOR = 8
_GEN_CF5FD0AD_GRID_SIZES = (3, 4, 5)
_GEN_CF5FD0AD_NON_BG_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)

# Sparse cell sampling and retry budget.
_GEN_CF5FD0AD_CELL_COUNT_RANGE = (1, 4)
_GEN_CF5FD0AD_MAX_DISTINCT_COLORS = 3
_GEN_CF5FD0AD_RETRY_LIMIT = 300

def _gen_cf5fd0ad_expected_output(grid: np.ndarray) -> np.ndarray:
    upper_left = np.tile(np.rot90(grid, 2), (2, 2))
    upper_right = np.tile(np.rot90(grid, -1), (2, 2))
    lower_left = np.tile(np.rot90(grid, 1), (2, 2))
    lower_right = np.tile(grid, (2, 2))
    return np.block([[upper_left, upper_right], [lower_left, lower_right]])


def _gen_cf5fd0ad_sparse_grid() -> np.ndarray:
    n = random.choice(_GEN_CF5FD0AD_GRID_SIZES)
    grid = np.full((n, n), _GEN_CF5FD0AD_BG_COLOR, dtype=int)
    cell_count = random.randint(*_GEN_CF5FD0AD_CELL_COUNT_RANGE)
    distinct_count = random.randint(1, min(_GEN_CF5FD0AD_MAX_DISTINCT_COLORS, cell_count))
    colors = random.sample(_GEN_CF5FD0AD_NON_BG_COLORS, distinct_count)
    positions = random.sample(range(n * n), cell_count)
    assigned_colors = colors[:]
    assigned_colors.extend(random.choice(colors) for _ in range(cell_count - distinct_count))
    random.shuffle(assigned_colors)

    for pos, color in zip(positions, assigned_colors):
        grid[pos // n, pos % n] = color
    return grid


def _gen_attempt() -> np.ndarray | None:
    grid = _gen_cf5fd0ad_sparse_grid()
    expected = _gen_cf5fd0ad_expected_output(grid)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_CF5FD0AD_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError('failed to generate cf5fd0ad input')
