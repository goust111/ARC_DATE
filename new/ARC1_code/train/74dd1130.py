import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect the square grid across its main diagonal.

    Input structure:
    - The input is a nonempty square grid.
    - The colors are arbitrary visible categories and are preserved as a multiset.

    Rule:
    - Move each pixel at row r and column c to row c and column r.
    - Return the transposed square grid.

    Key insight:
    - Every training output is the input read by columns instead of rows, which is
      exactly a main-diagonal reflection.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height > 0 and width > 0, "expected a nonempty grid"
    assert height == width, "expected a square grid"

    return g.T.copy()

# --------------- generator (o5x) ---------------

# Grid and palette constraints.
GRID_SHAPE = (3, 3)
PALETTE_SIZE = 3
COLOR_VALUES = np.arange(1, 10)

# Symmetry and retry controls.
ASYMMETRIC_PAIRS = ((0, 1), (0, 2), (1, 2))
MAX_GENERATION_ATTEMPTS = 300

_GEN_RNG = np.random.default_rng()


def _gen_attempt() -> np.ndarray | None:
    colors = _GEN_RNG.choice(COLOR_VALUES, size=PALETTE_SIZE, replace=False)
    grid = _GEN_RNG.choice(colors, size=GRID_SHAPE, replace=True).astype(int)

    r, c = ASYMMETRIC_PAIRS[int(_GEN_RNG.integers(len(ASYMMETRIC_PAIRS)))]
    a, b = _GEN_RNG.choice(colors, size=2, replace=False)
    grid[r, c] = int(a)
    grid[c, r] = int(b)

    reserved = {(r, c), (c, r)}
    remaining = [cell for cell in np.ndindex(GRID_SHAPE) if cell not in reserved]
    seed_idxs = _GEN_RNG.choice(len(remaining), size=PALETTE_SIZE, replace=False)
    for idx, color in zip(seed_idxs, colors):
        rr, cc = remaining[int(idx)]
        grid[rr, cc] = int(color)

    if grid.shape != GRID_SHAPE:
        return None
    if np.array_equal(grid, grid.T):
        return None
    if set(np.unique(grid).tolist()) != set(colors.tolist()):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp = _gen_attempt()
        if inp is None:
            continue
        out = inp.T.copy()
        if np.array_equal(main(inp), out):
            return inp
    raise RuntimeError("failed to generate valid 74dd1130 input")
