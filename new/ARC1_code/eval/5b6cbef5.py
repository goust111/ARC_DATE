import numpy as np

from arc_utils import Color, colors_present


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a monochrome seed grid by stamping the whole seed into every foreground-marked block.

    Input structure
    - The input is a 4x4 grid using black as the blank background and one non-black foreground color.
    - The same grid is both the seed sprite and the placement mask.

    Rule
    - Create a black output divided into 4x4 blocks, one block per input pixel.
    - For each non-background input pixel, copy the whole input grid into the corresponding output block.
    - For each background input pixel, leave the corresponding output block black.

    Key insight
    - The foreground mask selects locations for full self-copies; this is a self-similar block expansion, not a nearest-neighbor scale-up.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert (h, w) == (4, 4), "expected a 4x4 seed grid"

    bg = Color.BACKGROUND
    assert np.any(g == bg), "expected at least one background cell"
    foreground_colors = colors_present(g, exclude={bg})
    assert len(foreground_colors) == 1, "expected exactly one foreground color"

    out = np.full((h * h, w * w), bg, dtype=int)
    for r in range(h):
        for c in range(w):
            if int(g[r, c]) != bg:
                out[r * h : (r + 1) * h, c * w : (c + 1) * w] = g
    return out

# --------------- generator (o5x) ---------------

# Grid geometry.
SEED_SIZE, BACKGROUND_COLOR = 4, 0
CELL_COUNT = SEED_SIZE * SEED_SIZE
# Numpy randint bounds; high values are exclusive.
FOREGROUND_COLOR_BOUNDS = (1, 10)
FOREGROUND_COUNT_BOUNDS = (4, 11)
GENERATION_RETRY_LIMIT = 300

def _gen_attempt() -> np.ndarray | None:
    color = int(np.random.randint(*FOREGROUND_COLOR_BOUNDS))
    fg_count = int(np.random.randint(*FOREGROUND_COUNT_BOUNDS))
    inp = np.zeros((SEED_SIZE, SEED_SIZE), dtype=int)
    cells = np.random.choice(CELL_COUNT, size=fg_count, replace=False)
    inp.flat[cells] = color
    out = np.zeros((CELL_COUNT, CELL_COUNT), dtype=int)
    for r in range(SEED_SIZE):
        for c in range(SEED_SIZE):
            if inp[r, c] != BACKGROUND_COLOR:
                out[r * SEED_SIZE : (r + 1) * SEED_SIZE, c * SEED_SIZE : (c + 1) * SEED_SIZE] = inp
    if not np.array_equal(main(inp), out):
        return None
    return inp

def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRY_LIMIT):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid input for 5b6cbef5")
