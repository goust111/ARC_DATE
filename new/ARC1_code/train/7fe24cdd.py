import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a 2x2 mosaic from the input grid and its three quarter-turn rotations.

    Input structure:
    - A square 2D grid.

    Rule:
    - Put the original grid in the upper-left block.
    - Put the 90-degree clockwise rotation in the upper-right block.
    - Put the 90-degree counterclockwise rotation in the lower-left block.
    - Put the 180-degree rotation in the lower-right block.

    Key insight:
    - Each output quadrant is the entire input seen from one of the four square rotations.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height == width, "rotation mosaic requires a square grid"

    return np.block(
        [
            [g, np.rot90(g, -1)],
            [np.rot90(g, 1), np.rot90(g, 2)],
        ]
    )

# --------------- generator (o5x) ---------------

# Bounds and retry budget for the direct random generator.
SIDE_MIN, SIDE_MAX_EXCLUSIVE = 3, 6
COLOR_COUNT_MIN, COLOR_COUNT_MAX_EXCLUSIVE = 3, 6
ARC_COLOR_COUNT = 10
MAX_GENERATION_ATTEMPTS = 300


def _gen_expected_output(inp: np.ndarray) -> np.ndarray:
    return np.block(
        [
            [inp, np.rot90(inp, -1)],
            [np.rot90(inp, 1), np.rot90(inp, 2)],
        ]
    )


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.randint(SIDE_MIN, SIDE_MAX_EXCLUSIVE))
    color_count = int(np.random.randint(COLOR_COUNT_MIN, COLOR_COUNT_MAX_EXCLUSIVE))
    colors = np.random.choice(np.arange(ARC_COLOR_COUNT), size=color_count, replace=False)

    flat = np.random.choice(colors, size=n * n, replace=True)
    forced_positions = np.random.choice(np.arange(n * n), size=color_count, replace=False)
    flat[forced_positions] = colors
    grid = flat.reshape(n, n).astype(int)

    if np.array_equal(grid, np.rot90(grid, -1)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp = _gen_attempt()
        if inp is None:
            continue
        expected = _gen_expected_output(inp)
        if np.array_equal(main(inp), expected):
            return inp
    raise RuntimeError("failed to generate a valid 7fe24cdd input")
