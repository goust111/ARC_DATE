import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

INPUT_COLORS = (BLACK, BLUE, RED)
SIZE_OPTIONS = (3, 4)
GENERATION_RETRIES = 400

PATTERNS = {
    BLACK: np.array([[BLACK, BLACK], [BLACK, BLACK]], dtype=int),
    RED: np.array([[RED, YELLOW], [YELLOW, RED]], dtype=int),
    BLUE: np.array([[BLUE, BLUE], [GREEN, GREEN]], dtype=int),
}


def _expand_microstructure(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    out = np.zeros((2 * h, 2 * w), dtype=int)
    for r in range(h):
        for c in range(w):
            out[2 * r : 2 * r + 2, 2 * c : 2 * c + 2] = PATTERNS[int(grid[r, c])]
    return out


def _border_values(grid: np.ndarray) -> np.ndarray:
    top = grid[0, :]
    bottom = grid[-1, :]
    left = grid[1:-1, 0]
    right = grid[1:-1, -1]
    if left.size == 0:
        return np.concatenate((top, bottom))
    return np.concatenate((top, bottom, left, right))


def _valid_input_grid(grid: np.ndarray) -> bool:
    if grid.shape[0] != grid.shape[1] or grid.shape[0] not in SIZE_OPTIONS:
        return False
    if set(map(int, np.unique(grid))) != set(INPUT_COLORS):
        return False

    border = _border_values(grid)
    if set(map(int, np.unique(border))) != set(INPUT_COLORS):
        return False

    corners = [int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]
    if len(set(corners)) < 2:
        return False
    if BLUE not in corners and RED not in corners:
        return False

    if len(set(map(int, grid[0, :]))) < 2:
        return False
    if len(set(map(int, grid[:, 0]))) < 2:
        return False
    if len(set(map(int, grid[-1, :]))) < 2:
        return False
    if len(set(map(int, grid[:, -1]))) < 2:
        return False

    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each small input pixel into a color-dependent 2x2 microstructure, then copy the enlarged border outward by one cell.

    Input structure:
    - The input is a 3x3 or 4x4 square grid.
    - Every cell is one of three colors: black, blue, or red.

    Rule:
    - Replace each black cell with a 2x2 all-black block.
    - Replace each red cell with [[red, yellow], [yellow, red]].
    - Replace each blue cell with [[blue, blue], [green, green]].
    - After the whole grid is enlarged, copy the enlarged top row upward, the bottom row downward, the left column leftward, and the right column rightward to form a one-cell border; corner pixels extend in both directions.

    Key insight:
    - The final frame is not copied from the original coarse pixels; it is copied from the already-expanded microstructure, so edge colors depend on which subpixels lie on the enlarged boundary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] == g.shape[1] and g.shape[0] in SIZE_OPTIONS, "expected a 3x3 or 4x4 square grid"
    assert set(map(int, np.unique(g))) <= set(INPUT_COLORS), "unexpected input color"

    expanded = _expand_microstructure(g)
    return np.pad(expanded, 1, mode="edge")


def _gen_attempt() -> np.ndarray | None:
    size = random.choice(SIZE_OPTIONS)
    total = size * size

    cells = list(INPUT_COLORS)
    cells.extend(random.choice(INPUT_COLORS) for _ in range(total - len(INPUT_COLORS)))
    random.shuffle(cells)

    grid = np.array(cells, dtype=int).reshape(size, size)
    if not _valid_input_grid(grid):
        return None

    out = main(grid)
    if out.shape != (2 * size + 2, 2 * size + 2):
        return None
    if not set(map(int, np.unique(out))) <= {BLACK, BLUE, RED, GREEN, YELLOW}:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
