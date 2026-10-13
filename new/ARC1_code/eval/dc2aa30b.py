import numpy as np

from arc_utils import Color
import random


def _segments(length: int, dividers: list[int]) -> list[tuple[int, int]]:
    segments = []
    start = 0
    for divider in dividers + [length]:
        segments.append((start, divider))
        start = divider + 1
    return segments


def main(grid: np.ndarray) -> np.ndarray:
    """Sort the separated 3x3 tiles by their number of blue pixels and repack them from bottom-left to top-right.

    Input structure
    The grid is an 11x11 canvas divided by two full black rows and two full
    black columns into a 3x3 array of 3x3 tiles. Each tile contains only blue
    and red pixels.

    Rule
    Count the blue pixels in each tile. Sort the nine tiles by that count in
    increasing order, then place them unchanged into the tile slots starting at
    the bottom-left, moving left-to-right across each row, and continuing upward.

    Key insight
    The internal tile patterns are not edited; the puzzle is an ordering task
    where blue-pixel density determines each tile's rank.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape == (11, 11)

    bg = Color.BLACK
    separator_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == bg)]
    separator_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == bg)]
    assert len(separator_rows) == 2
    assert len(separator_cols) == 2

    row_segments = _segments(g.shape[0], separator_rows)
    col_segments = _segments(g.shape[1], separator_cols)
    assert len(row_segments) == 3
    assert len(col_segments) == 3
    assert all(end - start == 3 for start, end in row_segments)
    assert all(end - start == 3 for start, end in col_segments)

    tiles = []
    for r0, r1 in row_segments:
        for c0, c1 in col_segments:
            tile = g[r0:r1, c0:c1]
            assert set(map(int, tile.ravel())) <= {Color.BLUE, Color.RED}
            blue_count = int(np.count_nonzero(tile == Color.BLUE))
            tiles.append((blue_count, tile.copy()))

    blue_counts = [blue_count for blue_count, _ in tiles]
    assert len(set(blue_counts)) == len(blue_counts)

    out = np.full_like(g, bg)
    ordered_tiles = [tile for _, tile in sorted(tiles, key=lambda item: item[0])]
    destination_slots = [
        (r0, r1, c0, c1)
        for r0, r1 in reversed(row_segments)
        for c0, c1 in col_segments
    ]
    for tile, (r0, r1, c0, c1) in zip(ordered_tiles, destination_slots):
        out[r0:r1, c0:c1] = tile

    return out

# --------------- generator (o5x) ---------------
# Grid geometry/color constants.
GRID_SHAPE = (11, 11)
TILE_SIZE = 3
SLOT_ORIGINS = (0, 4, 8)
INPUT_SLOTS = [(r, c) for r in SLOT_ORIGINS for c in SLOT_ORIGINS]
OUTPUT_SLOTS = [(r, c) for r in reversed(SLOT_ORIGINS) for c in SLOT_ORIGINS]
BG_COLOR, BLUE_COLOR, RED_COLOR = Color.BLACK, Color.BLUE, Color.RED
# Count sampling and retry budget.
BLUE_COUNTS = tuple(range(TILE_SIZE * TILE_SIZE + 1))
TILE_CELLS = range(TILE_SIZE * TILE_SIZE)
MAX_GENERATION_ATTEMPTS = 300

def _gen_tile(blue_count: int) -> np.ndarray:
    tile = np.full((TILE_SIZE, TILE_SIZE), RED_COLOR, dtype=int)
    np.put(tile, random.sample(TILE_CELLS, blue_count), BLUE_COLOR)
    return tile

def _place_tiles(tiles, slots) -> np.ndarray:
    grid = np.full(GRID_SHAPE, BG_COLOR, dtype=int)
    for tile, (r, c) in zip(tiles, slots):
        grid[r:r + TILE_SIZE, c:c + TILE_SIZE] = tile
    return grid

def _gen_attempt() -> np.ndarray | None:
    counts = list(BLUE_COUNTS)
    counts.remove(random.choice(BLUE_COUNTS))
    tiles = [_gen_tile(count) for count in counts]
    shuffled_tiles = tiles[:]
    random.shuffle(shuffled_tiles)
    inp = _place_tiles(shuffled_tiles, INPUT_SLOTS)
    out = _place_tiles(tiles, OUTPUT_SLOTS)
    if not np.array_equal(main(inp), out):
        return None
    return inp

def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid dc2aa30b input")
