import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the 3x3 block from the upper-right corner of the 9x9 input.

    Input structure:
    - The input is a 9x9 grid arranged as a 3x3 array of 3x3 blocks.
    - The desired block is located at the top row and rightmost column of that block layout.

    Rule:
    - Return rows 0 through 2 and columns 6 through 8 unchanged as the 3x3 output.

    Key insight:
    - The output is not computed from color statistics; it is the fixed upper-right block of the partitioned input.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (9, 9), "expected a 9x9 input grid"
    return g[:3, 6:9].copy()

# --------------- generator (o5x) ---------------

# Canvas and 3x3 block geometry.
GRID_SHAPE = (9, 9)
BLOCK_SIZE = 3
BLOCKS_PER_SIDE = 3
BLOCK_COUNT, BLOCK_CELL_COUNT = BLOCKS_PER_SIDE**2, BLOCK_SIZE**2
# Fill count ranges.
TOTAL_FILL_RANGE = (18, 43)
BLOCK_FILL_RANGE = (1, 7)
# Color, retry, and output settings.
BACKGROUND, COLOR_MIN, COLOR_STOP = 0, 1, 10
COLOR_VALUES = np.arange(COLOR_MIN, COLOR_STOP, dtype=int)
RETRY_LIMIT = 300
OUTPUT_BLOCK = np.s_[0:3, 6:9]


def _gen_block_counts(total):
    counts = np.full(BLOCK_COUNT, BLOCK_FILL_RANGE[0], dtype=int)
    remaining = total - BLOCK_COUNT * BLOCK_FILL_RANGE[0]
    while remaining:
        choices = np.flatnonzero(counts < BLOCK_FILL_RANGE[1])
        counts[choices[int(np.random.randint(0, len(choices)))]] += 1
        remaining -= 1
    return counts


def _sample_filled_coords(block_counts):
    coords = []
    for block_i, count in enumerate(block_counts):
        br = (block_i // BLOCKS_PER_SIDE) * BLOCK_SIZE
        bc = (block_i % BLOCKS_PER_SIDE) * BLOCK_SIZE
        cells = np.random.choice(BLOCK_CELL_COUNT, size=int(count), replace=False)
        coords.extend((br + int(cell // BLOCK_SIZE), bc + int(cell % BLOCK_SIZE)) for cell in cells)
    return np.asarray(coords, dtype=int)


def _gen_attempt():
    target_fill = int(np.random.randint(TOTAL_FILL_RANGE[0], TOTAL_FILL_RANGE[1] + 1))
    grid = np.full(GRID_SHAPE, BACKGROUND, dtype=int)
    coords = _sample_filled_coords(_gen_block_counts(target_fill))
    colors = np.empty(target_fill, dtype=int)
    colors[: len(COLOR_VALUES)] = COLOR_VALUES
    if target_fill > len(COLOR_VALUES):
        colors[len(COLOR_VALUES) :] = np.random.randint(COLOR_MIN, COLOR_STOP, size=target_fill - len(COLOR_VALUES))
    np.random.shuffle(colors)
    grid[coords[:, 0], coords[:, 1]] = colors
    return grid


def generate_input():
    for _ in range(RETRY_LIMIT):
        inp = _gen_attempt()
        if np.array_equal(main(inp), inp[OUTPUT_BLOCK].copy()):
            return inp
    raise AssertionError("failed to generate a valid input")
