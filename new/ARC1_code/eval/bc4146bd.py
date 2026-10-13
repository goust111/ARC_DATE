import numpy as np
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a five-block horizontal strip by alternating the input tile with its left-right reflection.

    Input structure
    - A nonempty 2D color grid; all cells are part of the source tile.
    - The examples use a 4x4 tile and produce a same-height strip five tiles wide.

    Rule
    - Concatenate five whole-tile blocks across columns.
    - Use the original tile in blocks 1, 3, and 5.
    - Use the left-right mirror of the tile in blocks 2 and 4.

    Key insight
    The output is mirrored horizontal wallpaper, not pixel scaling: whole-grid copies alternate orientation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) > 0, "expected a nonempty 2D tile"

    mirrored = np.fliplr(g)
    return np.concatenate((g, mirrored, g, mirrored, g), axis=1)

# --------------- generator (o5x) ---------------

# Tile/color sampling bounds and generation retry limit.
TILE_H, TILE_W = 4, 4
COLOR_COUNT_RANGE = (2, 3)
COLOR_POOL = range(1, 10)
MAX_ATTEMPTS = 300


def _ensure_palette_present(tile: np.ndarray, palette: list[int], protected: set[tuple[int, int]]) -> None:
    present = set(int(x) for x in tile.ravel())
    free_cells = [(r, c) for r in range(TILE_H) for c in range(TILE_W) if (r, c) not in protected]
    random.shuffle(free_cells)
    for color in palette:
        if color not in present:
            r, c = free_cells.pop()
            tile[r, c] = color
            present.add(color)


def _gen_attempt() -> np.ndarray | None:
    color_count = random.randint(*COLOR_COUNT_RANGE)
    palette = random.sample(COLOR_POOL, color_count)

    tile = np.array(
        [random.choice(palette) for _ in range(TILE_H * TILE_W)],
        dtype=int,
    ).reshape(TILE_H, TILE_W)

    row = random.randrange(TILE_H)
    left_col = random.randrange(TILE_W // 2)
    right_col = TILE_W - 1 - left_col
    tile[row, left_col] = palette[0]
    tile[row, right_col] = palette[1]

    _ensure_palette_present(tile, palette, {(row, left_col), (row, right_col)})
    return tile


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        inp = _gen_attempt()
        if inp is None:
            continue
        mirror = np.fliplr(inp)
        out = np.concatenate((inp, mirror, inp, mirror, inp), axis=1)
        if np.array_equal(main(inp), out):
            return inp
    raise RuntimeError("failed to generate a valid bc4146bd input")
