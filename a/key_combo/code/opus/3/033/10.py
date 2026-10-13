import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
MAX_RETRIES = 300


def _scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    """Nearest-neighbor integer scaling, matching arc_utils.scale_sprite."""
    k = int(factor)
    if k <= 0:
        raise ValueError("scale factor must be positive")
    return np.kron(sprite, np.ones((k, k), dtype=sprite.dtype))


def _find_divider_row(grid: np.ndarray) -> int:
    divider_rows = np.flatnonzero(np.all(grid == GREY, axis=1))
    assert divider_rows.size == 1, "expected exactly one full-width grey divider row"
    divider = int(divider_rows[0])
    top_h = divider
    bottom_h = int(grid.shape[0] - divider - 1)
    assert top_h > 0 and bottom_h > 0 and top_h == bottom_h, "top and bottom masks must be equal in height"
    return divider


def main(grid: np.ndarray) -> np.ndarray:
    """
    Intersect the two mask halves and enlarge that intersection by its own foreground count.

    Input structure:
    - A black background grid contains one full-width grey divider row.
    - Above the divider is a red-on-black mask and below it is a blue-on-black mask.
    - The upper and lower masks have the same height and width and are aligned cell by cell.

    Rule:
    - Remove the grey divider and align the upper and lower masks.
    - Mark a cell green iff both aligned masks are non-black at that position; other cells stay black.
    - Let N be the number of green intersection cells, then scale the whole green/black intersection image by N in both dimensions.

    Key insight:
    - The boolean intersection serves two roles at once: it determines the output shape and its own pixel count determines the scale factor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    divider = _find_divider_row(g)
    top = g[:divider]
    bottom = g[divider + 1 :]
    assert top.shape == bottom.shape, "masks must have matching shapes"

    intersection = (top != BG) & (bottom != BG)
    scale = int(intersection.sum())
    assert 2 <= scale <= 4, "intersection size must stay in the intended range"

    pattern = np.where(intersection, GREEN, BG).astype(int)
    return _scale_sprite(pattern, scale)


def _sample_intersection_cells(height: int, width: int, overlap: int, rng: random.Random) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(height) for c in range(width)]
    return rng.sample(cells, overlap)


def _choose_unique_cells(
    available: list[tuple[int, int]],
    min_count: int,
    max_count: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    count = rng.randint(min_count, min(max_count, len(available)))
    return rng.sample(available, count)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    overlap = rng.choice((2, 3, 4))
    max_side = 30 // overlap
    height_max = min(9, (30 - 1) // 2, max_side)
    width_max = min(9, max_side)
    height = rng.randint(3, height_max)
    width = rng.randint(3, width_max)

    if height * width < overlap + 2:
        return None

    intersection_cells = _sample_intersection_cells(height, width, overlap, rng)
    intersection_set = set(intersection_cells)
    remaining = [(r, c) for r in range(height) for c in range(width) if (r, c) not in intersection_set]
    if len(remaining) < 2:
        return None

    top_extra = _choose_unique_cells(remaining, 1, min(4, len(remaining) - 1), rng)
    top_extra_set = set(top_extra)
    bottom_pool = [cell for cell in remaining if cell not in top_extra_set]
    if not bottom_pool:
        return None
    bottom_extra = _choose_unique_cells(bottom_pool, 1, min(4, len(bottom_pool)), rng)

    grid = np.full((height * 2 + 1, width), BG, dtype=int)
    grid[height, :] = GREY

    for r, c in intersection_cells:
        grid[r, c] = RED
        grid[height + 1 + r, c] = BLUE
    for r, c in top_extra:
        grid[r, c] = RED
    for r, c in bottom_extra:
        grid[height + 1 + r, c] = BLUE

    top_mask = grid[:height] != BG
    bottom_mask = grid[height + 1 :] != BG
    if int(np.count_nonzero(top_mask & bottom_mask)) != overlap:
        return None

    out = main(grid)
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
