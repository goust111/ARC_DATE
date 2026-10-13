from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the input into a block grid equal to the self-product of its black-cell mask.

    Input structure
    - The input is a non-empty 2D grid with black background cells and exactly one foreground color.
    - Every output block has the same height and width as the input.

    Rule
    - Treat the black cells of the input as a binary mask.
    - Replace each black input cell by that same black-cell mask, recolored with the foreground color.
    - Replace each foreground input cell by an all-black block.

    Key insight
    The enlarged output mask is the Kronecker product of the input's black-cell mask with itself.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = Color.BLACK
    assert np.any(g == bg), "expected black background cells"
    foreground_colors = colors_present(g, exclude={bg})
    assert len(foreground_colors) == 1, "expected exactly one foreground color"
    fg = foreground_colors[0]

    background_mask = g == bg
    output_mask = np.kron(background_mask, background_mask)
    return np.where(output_mask, fg, bg).astype(int)

# --------------- generator (o5x) ---------------

# Canvas and palette.
INPUT_SHAPE, INPUT_CELL_COUNT = (3, 3), 9
FOREGROUND_COLOR_RANGE = (1, 10)
# Mask sampling and oracle validation.
BLACK_COUNT_RANGE = (3, 6)
GENERATION_RETRIES = 300


def _sample_input_grid() -> np.ndarray:
    fg = int(np.random.randint(*FOREGROUND_COLOR_RANGE))
    black_count = int(np.random.randint(*BLACK_COUNT_RANGE))

    grid = np.full(INPUT_SHAPE, fg, dtype=int)
    black_cells = np.random.choice(INPUT_CELL_COUNT, size=black_count, replace=False)
    grid.flat[black_cells] = Color.BLACK
    return grid


def _gen_expected_output(inp: np.ndarray) -> np.ndarray:
    mask = inp == Color.BLACK
    fg = int(inp[inp != Color.BLACK][0])
    return np.where(np.kron(mask, mask), fg, Color.BLACK).astype(int)


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        inp = _sample_input_grid()
        out = _gen_expected_output(inp)
        if np.array_equal(main(inp), out):
            return inp
    raise RuntimeError("failed to generate a valid input")
