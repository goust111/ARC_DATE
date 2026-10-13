import numpy as np
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the whole grid by replacing each input pixel with a same-color 2x2 block.

    Input structure
    A two-dimensional color grid; every pixel is part of the source pattern.

    Rule
    Double the grid height and width by copying each pixel to the corresponding
    2x2 block in the output.

    Key insight
    The transformation preserves all colors and positions at pixel level, applying
    only a uniform scale factor of two.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    return np.repeat(np.repeat(g, 2, axis=0), 2, axis=1)

# --------------- generator (o5x) ---------------
# Grid dimension distribution.
SQUARE_SHAPE_PROB, SQUARE_SIZE_RANGE, FREE_SIZE_RANGE = 0.7, (2, 5), (2, 6)
# Palette and fill distribution.
COLOR_COUNT_RANGE, COLOR_POOL = (3, 7), range(1, 10)
BLACK_PROB, BLACK_MAX_FRACTION_DENOM = 0.65, 2
DIRICHLET_ALPHA_RANGE = (0.7, 2.2)
MAX_GENERATION_ATTEMPTS = 300  # Rejection sampling guard.


def _gen_attempt() -> np.ndarray | None:
    if random.random() < SQUARE_SHAPE_PROB:
        rows = cols = random.randint(*SQUARE_SIZE_RANGE)
    else:
        rows = random.randint(*FREE_SIZE_RANGE)
        cols = random.randint(*FREE_SIZE_RANGE)

    cells = rows * cols
    color_count = random.randint(COLOR_COUNT_RANGE[0], min(COLOR_COUNT_RANGE[1], cells))
    colors = random.sample(COLOR_POOL, color_count)

    remaining = cells - color_count
    values = list(colors)

    include_black = remaining > 0 and random.random() < BLACK_PROB
    if include_black:
        zero_count = random.randint(1, min(remaining, cells // BLACK_MAX_FRACTION_DENOM))
    else:
        zero_count = 0
    values.extend([0] * zero_count)
    remaining -= zero_count

    if remaining:
        weights = np.random.dirichlet(np.ones(color_count) * random.uniform(*DIRICHLET_ALPHA_RANGE))
        extra = np.random.choice(colors, size=remaining, replace=True, p=weights)
        values.extend(int(v) for v in extra)

    random.shuffle(values)
    grid = np.array(values, dtype=int).reshape(rows, cols)

    expected = np.repeat(np.repeat(grid, 2, axis=0), 2, axis=1)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid c59eb873 input")
