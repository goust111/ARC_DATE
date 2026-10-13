import numpy as np

from arc_utils import scale_sprite
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the whole grid by replacing each input pixel with a same-color 2x2 block.

    Input structure
    A two-dimensional color grid; every pixel is part of the source pattern.

    Rule
    Double the grid height and width by copying each pixel into the corresponding
    2x2 output block.

    Key insight
    No object-specific movement or recoloring is needed; the entire pixel grid is
    uniformly magnified by a factor of two.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    return scale_sprite(g, 2)

# --------------- generator (o5x) ---------------

# Canvas, density, and retry bounds.
GRID_SIZE_RANGE = (3, 15)
REGION_FRACTION_RANGE = (0.20, 0.35)
MAX_ATTEMPTS = 300
UPSCALE_FACTOR = 2

# Edge protection, connectivity, and color split behavior.
NEIGHBOR_DELTAS = ((1, 0), (-1, 0), (0, 1), (0, -1))
NON_BLACK_COLORS = tuple(range(1, 10))
BAND_DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))
BAND_JITTER = 0.35


def _allowed_mask(size: int) -> np.ndarray:
    allowed = np.ones((size, size), dtype=bool)
    edge_masks = (allowed[0, :], allowed[-1, :], allowed[:, 0], allowed[:, -1])
    edge_masks[random.randrange(len(edge_masks))][:] = False
    return allowed


def _neighbors(cell: tuple[int, int], allowed: np.ndarray):
    h, w = allowed.shape
    r, c = cell
    for dr, dc in NEIGHBOR_DELTAS:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w and allowed[rr, cc]:
            yield rr, cc


def _grow_region(allowed: np.ndarray, area: int) -> set[tuple[int, int]] | None:
    """Grow a random 4-connected set inside `allowed`; return None if growth stalls."""
    allowed_cells = list(zip(*np.where(allowed)))
    start = random.choice(allowed_cells)
    cells = {start}
    frontier = {nb for nb in _neighbors(start, allowed)}
    while len(cells) < area:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for nb in _neighbors(cell, allowed):
            if nb not in cells:
                frontier.add(nb)
    return cells


def _paint_region(size: int, cells: set[tuple[int, int]], area: int) -> np.ndarray:
    """Split the connected cells into two noisy directional color bands."""
    colors = random.sample(NON_BLACK_COLORS, 2)
    grid = np.zeros((size, size), dtype=int)
    region = list(cells)
    dr, dc = random.choice(BAND_DIRECTIONS)
    region.sort(key=lambda cell: cell[0] * dr + cell[1] * dc + random.random() * BAND_JITTER)
    split = random.randint(1, area - 1)
    for color, part in ((colors[0], region[:split]), (colors[1], region[split:])):
        rr, cc = zip(*part)
        grid[rr, cc] = color
    return grid


def _gen_attempt() -> np.ndarray | None:
    size = random.randint(*GRID_SIZE_RANGE)
    min_area = max(2, int(np.ceil(size * size * REGION_FRACTION_RANGE[0])))
    max_area = max(min_area, int(np.floor(size * size * REGION_FRACTION_RANGE[1])))
    area = random.randint(min_area, max_area)
    cells = _grow_region(_allowed_mask(size), area)
    if cells is None:
        return None
    grid = _paint_region(size, cells, area)
    expected = np.repeat(np.repeat(grid, UPSCALE_FACTOR, axis=0), UPSCALE_FACTOR, axis=1)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for 60c09cac")
