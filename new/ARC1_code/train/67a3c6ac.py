import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror the entire grid left-to-right.

    Input structure
    - A non-empty rectangular grid of ARC colors.

    Rule
    - Reverse the column order of every row, preserving all colors and row positions.

    Key insight
    - The output has the same shape and color counts as the input because it is the horizontal reflection of the full grid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    return np.fliplr(g).copy()

# --------------- generator (o5x) ---------------

# CFG: observed colors, square side bounds, and generation retry budget.
GEN_COLORS_67A3C6AC = np.array([1, 2, 6, 7], dtype=int)
GEN_SIDE_MIN_67A3C6AC = 3
GEN_SIDE_MAX_EXCLUSIVE_67A3C6AC = 8
GEN_RETRY_LIMIT_67A3C6AC = 300

def _seed_asymmetry_and_all_colors(grid: np.ndarray) -> None:
    """Overwrite a mirror pair and the two other colors to guarantee valid samples."""
    n = grid.shape[0]
    row = int(np.random.randint(n))
    left = int(np.random.randint(n // 2))
    right = n - 1 - left
    a, b = np.random.choice(GEN_COLORS_67A3C6AC, size=2, replace=False)
    grid[row, left] = int(a)
    grid[row, right] = int(b)
    remaining = [int(color) for color in GEN_COLORS_67A3C6AC if color not in {int(a), int(b)}]
    reserved = {(row, left), (row, right)}
    free_positions = [(r, c) for r in range(n) for c in range(n) if (r, c) not in reserved]
    chosen = np.random.choice(len(free_positions), size=len(remaining), replace=False)
    for color, pos_i in zip(remaining, chosen):
        grid[free_positions[int(pos_i)]] = color


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.randint(GEN_SIDE_MIN_67A3C6AC, GEN_SIDE_MAX_EXCLUSIVE_67A3C6AC))
    grid = np.random.choice(GEN_COLORS_67A3C6AC, size=(n, n))
    _seed_asymmetry_and_all_colors(grid)
    out = np.fliplr(grid).copy()
    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT_67A3C6AC):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError('failed to generate input')
