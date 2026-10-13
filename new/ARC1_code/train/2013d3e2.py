import numpy as np

from arc_utils import bbox, get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Extract the upper-left tile from a four-quadrant rotational collage.

    Input structure
    - A mostly background grid contains one even-sided square motif bbox.
    - The motif bbox is split into four equal square quadrants.

    Rule
    - Use the upper-left quadrant as the source tile.
    - Assert the upper-right, lower-left, and lower-right quadrants are that tile
      rotated 90 degrees clockwise, 90 degrees counterclockwise, and 180 degrees.
    - Return the upper-left quadrant.

    Key insight
    - The input object is a duplicated rotational pattern; the output is the
      unrotated source tile embedded at the upper-left of the motif.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    box = bbox(g != bg)
    assert box is not None, "expected a foreground motif"
    r0, c0, r1, c1 = box
    motif = g[r0 : r1 + 1, c0 : c1 + 1]

    height, width = motif.shape
    assert height == width and height % 2 == 0, "expected an even-sided square motif"
    half = height // 2
    source = motif[:half, :half]

    assert np.array_equal(
        motif[:half, half:], np.rot90(source, -1)
    ), "upper-right quadrant must be the clockwise rotation"
    assert np.array_equal(
        motif[half:, :half], np.rot90(source, 1)
    ), "lower-left quadrant must be the counterclockwise rotation"
    assert np.array_equal(
        motif[half:, half:], np.rot90(source, 2)
    ), "lower-right quadrant must be the 180-degree rotation"

    return source.copy()

# --------------- generator (o5x) ---------------
# Canvas, motif, and placement bounds.
GRID_SIZE_RANGE, TILE_SIZE_RANGE = range(8, 15), range(2, 5)
MOTIF_SIDE_FACTOR, MIN_MOTIF_MARGIN = 2, 1
VALID_SIZE_PAIRS = [
    (g, n) for g in GRID_SIZE_RANGE for n in TILE_SIZE_RANGE
    if g >= MOTIF_SIDE_FACTOR * n + 2 * MIN_MOTIF_MARGIN
]

# Tile occupancy and palette constraints.
MIN_TILE_PIXELS, MAX_TILE_PIXELS = 3, 9
MIN_TILE_FILL, MAX_TILE_FILL = 0.40, 0.80
MIN_TILE_COLORS, MAX_TILE_COLORS = 3, 5
NON_BG_COLORS = range(1, 10)

# Generation retry limit.
MAX_GENERATION_ATTEMPTS = 300


def _gen_rot90_source_coord(r, c, n):
    return c, n - 1 - r


def _gen_tile(n):
    total = n * n
    min_pixels = max(MIN_TILE_PIXELS, int(np.ceil(total * MIN_TILE_FILL)))
    max_pixels = min(MAX_TILE_PIXELS, total - 1, int(np.floor(total * MAX_TILE_FILL)))
    if min_pixels > max_pixels:
        return None

    k = random.randint(min_pixels, max_pixels)
    edge_cells = [(0, c) for c in range(n)] + [(r, n - 1) for r in range(1, n)]
    random.shuffle(edge_cells)

    for witness in edge_cells:
        blocked = _gen_rot90_source_coord(*witness, n)
        if blocked == witness:
            continue
        all_cells = [(r, c) for r in range(n) for c in range(n) if (r, c) != blocked]
        if len(all_cells) < k or witness not in all_cells:
            continue

        chosen = [witness]
        candidates = [cell for cell in all_cells if cell != witness]
        chosen.extend(random.sample(candidates, k - 1))
        random.shuffle(chosen)

        n_colors = random.randint(MIN_TILE_COLORS, min(MAX_TILE_COLORS, k))
        palette = random.sample(NON_BG_COLORS, n_colors)
        values = palette[:] + [random.choice(palette) for _ in range(k - n_colors)]
        random.shuffle(values)

        tile = np.zeros((n, n), dtype=int)
        rows, cols = zip(*chosen)
        tile[rows, cols] = values
        return tile

    return None


def _assemble_motif(tile):
    n = tile.shape[0]
    motif = np.zeros((2 * n, 2 * n), dtype=int)
    motif[:n, :n], motif[:n, n:] = tile, np.rot90(tile, -1)
    motif[n:, :n], motif[n:, n:] = np.rot90(tile, 1), np.rot90(tile, 2)
    return motif


def _place_motif(grid_size, motif):
    motif_side = motif.shape[0]
    grid = np.zeros((grid_size, grid_size), dtype=int)
    r0 = random.randint(MIN_MOTIF_MARGIN, grid_size - motif_side - MIN_MOTIF_MARGIN)
    c0 = random.randint(MIN_MOTIF_MARGIN, grid_size - motif_side - MIN_MOTIF_MARGIN)
    grid[r0 : r0 + motif_side, c0 : c0 + motif_side] = motif
    return grid


def _gen_attempt():
    g, n = random.choice(VALID_SIZE_PAIRS)
    tile = _gen_tile(n)
    if tile is None:
        return None

    grid = _place_motif(g, _assemble_motif(tile))
    return grid if np.array_equal(main(grid), tile) else None


def generate_input():
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
