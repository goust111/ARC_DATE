import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """Build a larger square from the input grid and its three quarter-turn rotations.

    Input structure
    A square grid whose whole contents form the pattern; there is no separate
    background object or marker.

    Rule
    Make a 2n by 2n output from four n by n quadrants: the original grid in the
    upper-left, the grid rotated 90 degrees counterclockwise in the upper-right,
    the grid rotated 180 degrees in the lower-left, and the grid rotated 270
    degrees counterclockwise in the lower-right.

    Key insight
    The output repeats the entire input under rotation, so every input cell
    appears once in each quadrant rather than being expanded in place.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w

    top = np.concatenate([g, np.rot90(g, 1)], axis=1)
    bottom = np.concatenate([np.rot90(g, 2), np.rot90(g, 3)], axis=1)
    return np.concatenate([top, bottom], axis=0)

# --------------- generator (o5x) ---------------

# Canvas bounds.
GEN_SIDE_MIN, GEN_SIDE_MAX_EXCLUSIVE = 3, 6
# Color constraints.
GEN_COLOR_VALUES, GEN_COLOR_COUNT = np.arange(1, 10), 4
# Rejection and retry controls.
GEN_ROTATION_STEPS, GEN_RETRY_LIMIT = (1, 2, 3), 300


def _gen_has_quarter_turn_symmetry(grid: np.ndarray) -> bool:
    return any(np.array_equal(grid, np.rot90(grid, k)) for k in GEN_ROTATION_STEPS)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n = int(rng.integers(GEN_SIDE_MIN, GEN_SIDE_MAX_EXCLUSIVE))
    colors = rng.choice(GEN_COLOR_VALUES, size=GEN_COLOR_COUNT, replace=False)

    grid = rng.choice(colors, size=(n, n))
    forced = rng.choice(n * n, size=GEN_COLOR_COUNT, replace=False)
    grid.ravel()[forced] = colors

    if _gen_has_quarter_turn_symmetry(grid):
        return None

    expected = np.block([
        [grid, np.rot90(grid, 1)],
        [np.rot90(grid, 2), np.rot90(grid, 3)],
    ])
    if not np.array_equal(main(grid), expected):
        return None

    return grid.astype(int)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
