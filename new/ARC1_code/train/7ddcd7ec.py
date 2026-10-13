import numpy as np

from arc_utils import colors_present, get_bg, in_bounds
import random


DIAGONAL_DIRECTIONS = (
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)


def main(grid: np.ndarray) -> np.ndarray:
    """Extend each diagonal control pixel attached to a foreground 2x2 square into a ray to the grid edge.

    Input structure
    - The dominant background surrounds one monochrome foreground object.
    - The object contains a unique solid 2x2 square.
    - Any extra foreground pixels are single diagonal neighbors of the square's corners.

    Rule
    - Preserve the input grid.
    - For each extra foreground pixel diagonally attached to a square corner, continue in the same diagonal direction away from the square.
    - Paint every background cell on that ray with the foreground color until the ray leaves the grid.

    Key insight
    - The 2x2 square is the anchor; its diagonal corner pixels are ray seeds that point outward from the square.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))
    foreground_colors = colors_present(g, exclude={bg})
    assert len(foreground_colors) == 1, "expected one foreground color"
    fg = int(foreground_colors[0])

    height, width = map(int, g.shape)
    square_origins: list[tuple[int, int]] = []
    for r in range(height - 1):
        for c in range(width - 1):
            if np.all(g[r : r + 2, c : c + 2] == fg):
                square_origins.append((r, c))
    assert len(square_origins) == 1, "expected a unique foreground 2x2 square"

    square_r, square_c = square_origins[0]
    square_cells = {
        (square_r, square_c),
        (square_r, square_c + 1),
        (square_r + 1, square_c),
        (square_r + 1, square_c + 1),
    }

    seeds: list[tuple[int, int, int, int]] = []
    for dr, dc in DIAGONAL_DIRECTIONS:
        corner_r = square_r if dr < 0 else square_r + 1
        corner_c = square_c if dc < 0 else square_c + 1
        seed_r = corner_r + dr
        seed_c = corner_c + dc
        if in_bounds(g.shape, seed_r, seed_c) and g[seed_r, seed_c] == fg:
            seeds.append((seed_r, seed_c, dr, dc))
    assert seeds, "expected at least one diagonal ray seed"

    expected_foreground = square_cells | {(r, c) for r, c, _dr, _dc in seeds}
    actual_foreground = {tuple(map(int, coord)) for coord in np.argwhere(g == fg)}
    assert actual_foreground == expected_foreground, "foreground must consist only of the square and diagonal seeds"

    out = g.copy()
    for seed_r, seed_c, dr, dc in seeds:
        r = seed_r + dr
        c = seed_c + dc
        while in_bounds(g.shape, r, c):
            assert g[r, c] == bg, "ray continuation must pass through background"
            out[r, c] = fg
            r += dr
            c += dc

    return out

# --------------- generator (o5x) ---------------

# Canvas, color, square, seed, and retry bounds.
GRID_HEIGHT = GRID_WIDTH = 10
BACKGROUND, SQUARE_SIZE = 0, 2
FOREGROUND_MIN, FOREGROUND_MAX = 1, 9
SQUARE_ORIGIN_MIN, SQUARE_ORIGIN_EDGE_MARGIN = 2, 4
MIN_ACTIVE_SEEDS, MAX_ACTIVE_SEEDS = 1, 4
GENERATE_ATTEMPTS = 300

def _gen_ray_cells_7ddcd7ec(seed_r, seed_c, dr, dc, height, width):
    cells = []
    r, c = seed_r + dr, seed_c + dc
    while 0 <= r < height and 0 <= c < width:
        cells.append((r, c))
        r, c = r + dr, c + dc
    return cells

def _gen_corner_seeds(square_r, square_c):
    return [
        (square_r - 1, square_c - 1, -1, -1),
        (square_r - 1, square_c + SQUARE_SIZE, -1, 1),
        (square_r + SQUARE_SIZE, square_c - 1, 1, -1),
        (square_r + SQUARE_SIZE, square_c + SQUARE_SIZE, 1, 1),
    ]

def _gen_output(fg, square_r, square_c, active):
    """Render the solved grid and return ray-only cells for input erasure."""
    out = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)
    out[square_r : square_r + SQUARE_SIZE, square_c : square_c + SQUARE_SIZE] = fg

    ray_cells = set()
    for seed_r, seed_c, dr, dc in active:
        out[seed_r, seed_c] = fg
        for cell in _gen_ray_cells_7ddcd7ec(seed_r, seed_c, dr, dc, GRID_HEIGHT, GRID_WIDTH):
            ray_cells.add(cell)
            out[cell] = fg
    return out, ray_cells

def _gen_attempt():
    fg = random.randint(FOREGROUND_MIN, FOREGROUND_MAX)
    square_r = random.randint(SQUARE_ORIGIN_MIN, GRID_HEIGHT - SQUARE_ORIGIN_EDGE_MARGIN)
    square_c = random.randint(SQUARE_ORIGIN_MIN, GRID_WIDTH - SQUARE_ORIGIN_EDGE_MARGIN)
    active = random.sample(_gen_corner_seeds(square_r, square_c), random.randint(MIN_ACTIVE_SEEDS, MAX_ACTIVE_SEEDS))

    out, ray_cells = _gen_output(fg, square_r, square_c, active)
    inp = out.copy()
    for r, c in ray_cells:
        inp[r, c] = BACKGROUND

    try:
        return inp if np.array_equal(main(inp), out) else None
    except Exception:
        return None


def generate_input():
    for _ in range(GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 7ddcd7ec input")
