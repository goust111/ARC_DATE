import numpy as np
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate the entire input grid 180 degrees around its center.

    Input structure
    - The input is a 3x3 grid whose pixels form one complete pattern.

    Rule
    - Move each pixel to the cell opposite it through the grid center.

    Key insight
    - The output preserves all colors and positions relative to each other, but the global orientation is reversed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape == (3, 3), "expected the task's 3x3 grid"

    return np.rot90(g, 2).copy()

# --------------- generator (o5x) ---------------
# Grid layout.
GRID_SHAPE = (3, 3)
REQUIRED_NONZERO_POSITIONS = [(0, 0), (0, 1), (1, 0)]
OPTIONAL_NONZERO_POSITIONS = [(0, 2), (1, 1), (2, 0)]
# Color/count sampling.
NONZERO_COLOR_POOL = range(1, 10)
DISTINCT_COLOR_RANGE = (3, 5)
NONZERO_COUNT_RANGE = (4, 6)
# Transform oracle and retry guard.
ROTATION_K = 2
MAX_GENERATION_ATTEMPTS = 300

def _gen_color_values(_gen_count: int, _gen_distinct: int) -> list[int]:
    _gen_palette = random.sample(NONZERO_COLOR_POOL, _gen_distinct)
    _gen_values = list(_gen_palette)
    _gen_values.extend(random.choice(_gen_palette) for _ in range(_gen_count - _gen_distinct))
    random.shuffle(_gen_values)
    return _gen_values


def _gen_attempt() -> np.ndarray | None:
    _gen_grid = np.zeros(GRID_SHAPE, dtype=int)
    _gen_distinct = random.randint(*DISTINCT_COLOR_RANGE)
    _gen_nonzero_count = random.randint(max(NONZERO_COUNT_RANGE[0], _gen_distinct), NONZERO_COUNT_RANGE[1])
    _gen_optional_count = _gen_nonzero_count - len(REQUIRED_NONZERO_POSITIONS)
    _gen_positions = REQUIRED_NONZERO_POSITIONS + random.sample(OPTIONAL_NONZERO_POSITIONS, _gen_optional_count)
    random.shuffle(_gen_positions)

    for (_gen_r, _gen_c), _gen_color in zip(
        _gen_positions, _gen_color_values(_gen_nonzero_count, _gen_distinct)
    ):
        _gen_grid[_gen_r, _gen_c] = _gen_color

    _gen_output = np.rot90(_gen_grid, ROTATION_K).copy()
    if np.array_equal(main(_gen_grid), _gen_output):
        return _gen_grid
    return None


def generate_input() -> np.ndarray:
    for _gen_attempt_i in range(MAX_GENERATION_ATTEMPTS):
        _gen_grid = _gen_attempt()
        if _gen_grid is not None:
            return _gen_grid
    raise RuntimeError("failed to generate a valid 6150a2bd input")
