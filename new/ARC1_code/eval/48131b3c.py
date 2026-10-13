import numpy as np
import math
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap black with the single foreground color, then tile the swapped grid in a 2-by-2 block.

    Input structure:
    - The grid contains exactly two colors: black and one non-black foreground color.
    - The colored pixels form the visible pattern, and black is the empty/background color.

    Rule:
    - Replace every black pixel with the foreground color and every foreground pixel with black.
    - Repeat that swapped grid twice horizontally and twice vertically.

    Key insight:
    The output is not a pixel-scale enlargement; it is four copies of the color-complemented input pattern.
    """
    g = np.asarray(grid, dtype=int)
    colors = set(map(int, np.unique(g)))
    assert 0 in colors and len(colors) == 2, "expected black and one foreground color"

    foreground_colors = [color for color in colors if color != 0]
    assert len(foreground_colors) == 1, "expected exactly one non-black foreground color"
    foreground = foreground_colors[0]

    swapped = np.where(g == 0, foreground, 0)
    return np.tile(swapped, (2, 2))

# --------------- generator (o5x) ---------------

# Canvas/palette bounds, density limits, and retry budget.
GRID_SIZE_RANGE = (2, 5)
FOREGROUND_COLOR_RANGE = (1, 9)
FOREGROUND_FRACTION_RANGE = (0.30, 0.50)
GENERATION_ATTEMPTS = 300


def _has_both_colors_per_axis(mask: np.ndarray) -> bool:
    foreground_axes = mask.any(axis=0).all() and mask.any(axis=1).all()
    background_axes = (~mask).any(axis=0).all() and (~mask).any(axis=1).all()
    return foreground_axes and background_axes


def _gen_binary_pattern(size: int, foreground_count: int) -> np.ndarray | None:
    mask = np.zeros((size, size), dtype=bool)

    cols = list(range(size))
    random.shuffle(cols)
    for r, c in enumerate(cols):
        mask[r, c] = True

    row_counts = mask.sum(axis=1).astype(int).tolist()
    col_counts = mask.sum(axis=0).astype(int).tolist()

    remaining = foreground_count - size
    while remaining:
        candidates = [
            (r, c)
            for r in range(size)
            for c in range(size)
            if not mask[r, c] and row_counts[r] < size - 1 and col_counts[c] < size - 1
        ]
        if not candidates:
            return None
        r, c = random.choice(candidates)
        mask[r, c] = True
        row_counts[r] += 1
        col_counts[c] += 1
        remaining -= 1

    if not _has_both_colors_per_axis(mask):
        return None
    return mask


def _gen_attempt() -> np.ndarray | None:
    size = random.randint(*GRID_SIZE_RANGE)
    foreground = random.randint(*FOREGROUND_COLOR_RANGE)
    density_min, density_max = FOREGROUND_FRACTION_RANGE
    total = size * size
    min_count = max(size, math.ceil(density_min * total))
    max_count = min(total - size, math.floor(density_max * total))
    if min_count > max_count:
        return None

    foreground_count = random.randint(min_count, max_count)
    mask = _gen_binary_pattern(size, foreground_count)
    if mask is None:
        return None

    grid = np.zeros((size, size), dtype=int)
    grid[mask] = foreground
    expected = np.tile(np.where(grid == 0, foreground, 0), (2, 2))
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 48131b3c input")
