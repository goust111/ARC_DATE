import numpy as np

from arc_utils import get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress a 3x3 partitioned grid by replacing each large block with its lone foreground color.

    Input structure
    - The input is a 2D grid whose height and width are each divisible by 3.
    - The dominant background color fills almost all cells.
    - Each cell of the implicit 3x3 partition contains exactly one non-background pixel.

    Rule
    - Split the input into equal rectangular blocks in a 3x3 layout.
    - For each block, copy its single non-background color into the corresponding output cell.

    Key insight
    - The scattered pixels are already positioned inside an enlarged 3x3 grid; only the block scale is removed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height % 3 == 0 and width % 3 == 0, "grid must split evenly into a 3x3 partition"

    block_h = height // 3
    block_w = width // 3
    bg = get_bg(g)
    out = np.empty((3, 3), dtype=int)

    for out_r in range(3):
        for out_c in range(3):
            block = g[
                out_r * block_h : (out_r + 1) * block_h,
                out_c * block_w : (out_c + 1) * block_w,
            ]
            foreground = block[block != bg]
            assert foreground.size == 1, "expected one foreground pixel in each block"
            out[out_r, out_c] = int(foreground[0])

    return out

# --------------- generator (o5x) ---------------

# Generator CFG: layout, colors, sampled block sizes, and retry budget.
GRID_BLOCKS, BACKGROUND_COLOR, MAX_ATTEMPTS = 3, 0, 300
BLOCK_SIZE_CHOICES, FOREGROUND_COLORS = [2, 3], np.arange(1, GRID_BLOCKS**2 + 1, dtype=int)

def _gen_attempt() -> np.ndarray | None:
    block_size = int(np.random.choice(BLOCK_SIZE_CHOICES))
    out = np.random.permutation(FOREGROUND_COLORS).reshape(GRID_BLOCKS, GRID_BLOCKS)
    inp = np.full((GRID_BLOCKS * block_size, GRID_BLOCKS * block_size), BACKGROUND_COLOR, dtype=int)

    for block_r in range(GRID_BLOCKS):
        for block_c in range(GRID_BLOCKS):
            r = block_r * block_size + int(np.random.randint(block_size))
            c = block_c * block_size + int(np.random.randint(block_size))
            inp[r, c] = int(out[block_r, block_c])

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid input for 5783df64")
