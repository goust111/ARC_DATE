import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the upper-left 2x2 seed tile from the larger patterned grid.

    Input structure
    - The input is a two-dimensional grid with even height and width.
    - The compact seed pattern is located in the upper-left 2x2 region.

    Rule
    - Return the first two rows and first two columns unchanged.

    Key insight
    - The surrounding cells are shifted/permuted repetitions of the seed, so the
      unshifted representation is already present in the upper-left corner.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 2 and w >= 2
    assert h % 2 == 0 and w % 2 == 0
    return g[:2, :2].copy()

# --------------- generator (o5x) ---------------
# Seed color and block geometry controls.
COLOR_PALETTE = np.arange(10)
REPEATED_SEED_COLOR_COUNT, BLOCK_SIZE, ROTATION_COUNT = 3, 2, 4
SEED_COLOR_COUNTS = [REPEATED_SEED_COLOR_COUNT, ROTATION_COUNT]
# Canvas dimensions and retry limit.
GRID_SIDE_CHOICES = np.arange(4, 13, BLOCK_SIZE)
GENERATION_RETRIES = 300

def _gen_seed_tile():
    n_colors = int(np.random.choice(SEED_COLOR_COUNTS))
    colors = np.random.choice(COLOR_PALETTE, size=n_colors, replace=False)
    if n_colors == REPEATED_SEED_COLOR_COUNT:
        colors = np.concatenate([colors, colors[np.random.randint(0, REPEATED_SEED_COLOR_COUNT)][None]])
    np.random.shuffle(colors)
    seed = colors.reshape(BLOCK_SIZE, BLOCK_SIZE).astype(int)

    rotations = [np.rot90(seed, k) for k in range(ROTATION_COUNT)]
    assert len({tuple(rot.ravel()) for rot in rotations}) == ROTATION_COUNT
    return seed

def _gen_attempt():
    seed = _gen_seed_tile()
    h = int(np.random.choice(GRID_SIDE_CHOICES))
    w = int(np.random.choice(GRID_SIDE_CHOICES))
    grid = np.empty((h, w), dtype=int)

    for br in range(h // BLOCK_SIZE):
        for bc in range(w // BLOCK_SIZE):
            k = (br + bc) % ROTATION_COUNT
            r0, c0 = BLOCK_SIZE * br, BLOCK_SIZE * bc
            grid[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE] = np.rot90(seed, k)

    if not np.array_equal(main(grid), seed):
        return None
    return grid

def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
