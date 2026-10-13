import numpy as np


AZURE = 8
ORANGE = 7
RED = 2
MAROON = 9


def _opposite(color):
    return ORANGE if color == AZURE else AZURE


def _checker_color(row, col, first_color):
    if (row + col) % 2 == 0:
        return first_color
    return _opposite(first_color)


def _infer_tiling(grid):
    h, w = grid.shape
    if h != w:
        raise ValueError("expected a square tiled grid")

    best = None
    for size in range(2, h + 1):
        if h % size:
            continue
        tiles = h // size
        if tiles < 2:
            continue

        for first_color in (AZURE, ORANGE):
            mismatches = 0
            red_penalty = 0
            for tr in range(tiles):
                for tc in range(tiles):
                    block = grid[tr * size:(tr + 1) * size, tc * size:(tc + 1) * size]
                    expected = _checker_color(tr, tc, first_color)
                    colored = block != RED
                    mismatches += int(np.count_nonzero(colored & (block != expected)))
                    red_count = int(np.count_nonzero(block == RED))
                    if red_count > 3:
                        red_penalty += (red_count - 3) * size * size

            score = mismatches + red_penalty
            candidate = (score, -tiles, size, tiles, first_color)
            if best is None or candidate < best:
                best = candidate

    if best is None:
        raise ValueError("could not infer tile size")
    _, _, size, tiles, first_color = best
    return size, tiles, first_color


def main(grid: np.ndarray) -> np.ndarray:
    """Flip marked checkerboard tiles by red-marker parity, then draw maroon boundaries.

    Input structure:
    A square grid is partitioned into equal square tiles arranged in a checkerboard
    of azure and orange tile backgrounds. Some tile interiors contain 1 to 3 red
    marker pixels, while unmarked tiles contain no red pixels and there is no
    outside border.

    Rule:
    Infer the tile size and checkerboard phase from the azure/orange pattern while
    ignoring red markers. For each tile, count red marker pixels: an odd count
    flips the tile's background color between azure and orange, and an even count
    leaves it unchanged. Remove all red markers by filling each tile with its
    resulting color. After all flips, compare every pair of horizontally or
    vertically adjacent tiles; when their resulting colors differ, paint the
    one-pixel strip on the upper or left tile side of their shared edge maroon.

    Key insight:
    The red markers are not output objects themselves; only the parity of their
    count controls the hidden tile-color flip, and the final maroon lines depend
    on color differences after those parity flips.
    """
    grid = np.asarray(grid)
    size, tiles, first_color = _infer_tiling(grid)

    tile_colors = np.empty((tiles, tiles), dtype=grid.dtype)
    for tr in range(tiles):
        for tc in range(tiles):
            block = grid[tr * size:(tr + 1) * size, tc * size:(tc + 1) * size]
            base = _checker_color(tr, tc, first_color)
            red_count = int(np.count_nonzero(block == RED))
            tile_colors[tr, tc] = _opposite(base) if red_count % 2 else base

    out = np.empty_like(grid)
    for tr in range(tiles):
        for tc in range(tiles):
            out[tr * size:(tr + 1) * size, tc * size:(tc + 1) * size] = tile_colors[tr, tc]

    for tr in range(tiles):
        row_slice = slice(tr * size, (tr + 1) * size)
        for tc in range(tiles - 1):
            if tile_colors[tr, tc] != tile_colors[tr, tc + 1]:
                out[row_slice, (tc + 1) * size - 1] = MAROON

    for tr in range(tiles - 1):
        col_slice = slice(0, tiles * size)
        for tc in range(tiles):
            if tile_colors[tr, tc] != tile_colors[tr + 1, tc]:
                out[(tr + 1) * size - 1, tc * size:(tc + 1) * size] = MAROON

    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    tiles = int(rng.integers(3, 7))
    max_size = min(8, 30 // tiles)
    size = int(rng.integers(5, max_size + 1))
    first_color = AZURE if bool(rng.integers(0, 2)) else ORANGE

    grid = np.empty((tiles * size, tiles * size), dtype=int)
    for tr in range(tiles):
        for tc in range(tiles):
            color = _checker_color(tr, tc, first_color)
            grid[tr * size:(tr + 1) * size, tc * size:(tc + 1) * size] = color

    total_tiles = tiles * tiles
    counts = [0, 1, 2, 3]
    while len(counts) < total_tiles:
        counts.append(int(rng.integers(0, 4)))
    rng.shuffle(counts)

    interior = [(r, c) for r in range(1, size - 1) for c in range(1, size - 1)]
    for index, marker_count in enumerate(counts):
        if marker_count == 0:
            continue
        tr, tc = divmod(index, tiles)
        choices = rng.choice(len(interior), size=marker_count, replace=False)
        for choice in choices:
            rr, cc = interior[int(choice)]
            grid[tr * size + rr, tc * size + cc] = RED

    return grid
