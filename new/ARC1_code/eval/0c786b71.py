import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a double-sized mirror mosaic with the original grid in the lower-right quadrant.

    Input structure
    - A rectangular 2D color grid.

    Rule
    - Place the original grid in the lower-right quadrant of the output.
    - Fill the lower-left quadrant with a left-right reflection of the input.
    - Fill the upper-right quadrant with an up-down reflection of the input.
    - Fill the upper-left quadrant with the 180-degree rotation of the input.

    Key insight
    - The output is a mirrored extension of the input upward and left, not a pixel-scale zoom.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    return np.block([[np.flipud(np.fliplr(g)), np.flipud(g)], [np.fliplr(g), g]])

# --------------- generator (o5x) ---------------
# Grid dimensions sampled with high-exclusive numpy bounds.
GRID_HEIGHT_BOUNDS = (2, 7)
GRID_WIDTH_BOUNDS = (2, 7)
# Palette contains three distinct non-black ARC colors.
COLOR_VALUES = np.arange(1, 10)
PALETTE_SIZE = 3
# Symmetry rejection and oracle validation retry budget.
MAX_GENERATION_ATTEMPTS = 300


def _gen_has_degenerate_symmetry(grid: np.ndarray) -> bool:
    return (
        np.array_equal(grid, np.fliplr(grid))
        or np.array_equal(grid, np.flipud(grid))
        or np.array_equal(grid, np.flipud(np.fliplr(grid)))
    )


def _gen_expected_output(grid: np.ndarray) -> np.ndarray:
    return np.block([[np.flipud(np.fliplr(grid)), np.flipud(grid)], [np.fliplr(grid), grid]])


def _gen_attempt() -> np.ndarray | None:
    height = int(np.random.randint(*GRID_HEIGHT_BOUNDS))
    width = int(np.random.randint(*GRID_WIDTH_BOUNDS))
    colors = np.random.choice(COLOR_VALUES, size=PALETTE_SIZE, replace=False)

    grid = np.random.choice(colors, size=(height, width)).astype(int)

    # Force all three colors to appear while breaking LR, UD, and 180-degree symmetry.
    corner_colors = np.random.permutation(colors)
    top_left = int(corner_colors[0])
    top_right = int(corner_colors[1])
    bottom_left = int(corner_colors[2])
    bottom_right = int(np.random.choice([top_right, bottom_left]))
    grid[0, 0] = top_left
    grid[0, -1] = top_right
    grid[-1, 0] = bottom_left
    grid[-1, -1] = bottom_right

    if _gen_has_degenerate_symmetry(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is None:
            continue
        expected = _gen_expected_output(grid)
        if np.array_equal(main(grid), expected):
            return grid
    raise RuntimeError("failed to generate a valid 0c786b71 input")
