import numpy as np

from arc_utils import Color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeat the 3x3 input sprite into the first foreground-count slots of a black block grid.

    Input structure
    - The input is a 3x3 sprite containing black background and exactly one non-black foreground color.
    - The number of black pixels determines the square block-grid side length.
    - The number of non-black pixels determines how many copies of the sprite are emitted.

    Rule
    - Build a black output canvas with black-pixel-count slots per side, where each slot is 3x3.
    - Copy the entire input sprite into the first non-black-pixel-count slots in row-major order.
    - Leave all remaining slots black.

    Key insight
    - The colors are not changed; the two pixel counts control the output size and the number of repeated input sprites.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape == (3, 3), "expected a 3x3 sprite"

    colors = set(map(int, np.unique(g)))
    assert Color.BLACK in colors, "expected black background pixels"
    foreground_colors = colors - {Color.BLACK}
    assert len(foreground_colors) == 1, "expected exactly one foreground color"

    tile_h, tile_w = g.shape
    slot_count = int(np.sum(g == Color.BLACK))
    copy_count = int(np.sum(g != Color.BLACK))
    assert slot_count > 0, "expected at least one black slot marker"
    assert copy_count > 0, "expected at least one foreground pixel"
    assert copy_count <= slot_count * slot_count, "expected enough row-major slots for all copies"

    out = np.full((tile_h * slot_count, tile_w * slot_count), Color.BLACK, dtype=int)
    for index in range(copy_count):
        block_r, block_c = divmod(index, slot_count)
        r0 = block_r * tile_h
        c0 = block_c * tile_w
        out[r0 : r0 + tile_h, c0 : c0 + tile_w] = g

    return out

# --------------- generator (o5x) ---------------

# Sprite dimensions and sampling bounds.
GRID_SHAPE = (3, 3)
CELL_COUNT = GRID_SHAPE[0] * GRID_SHAPE[1]
BLACK_COUNT_BOUNDS = (3, 9)
FOREGROUND_COLOR_BOUNDS = (1, 10)
MAX_ATTEMPTS = 300  # Solver-oracle retry budget.


def _gen_expected_output(grid: np.ndarray) -> np.ndarray:
    slot_count = int(np.sum(grid == 0))
    copy_count = int(np.sum(grid != 0))
    tile_h, tile_w = grid.shape
    out = np.zeros((tile_h * slot_count, tile_w * slot_count), dtype=int)
    for index in range(copy_count):
        block_r, block_c = divmod(index, slot_count)
        r0 = block_r * tile_h
        c0 = block_c * tile_w
        out[r0 : r0 + tile_h, c0 : c0 + tile_w] = grid
    return out

def _gen_attempt() -> np.ndarray | None:
    black_count = int(np.random.randint(*BLACK_COUNT_BOUNDS))
    foreground_count = CELL_COUNT - black_count
    foreground_color = int(np.random.randint(*FOREGROUND_COLOR_BOUNDS))

    grid = np.zeros(GRID_SHAPE, dtype=int)
    foreground_positions = np.random.choice(CELL_COUNT, size=foreground_count, replace=False)
    grid.flat[foreground_positions] = foreground_color

    expected = _gen_expected_output(grid)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for 91413438")
