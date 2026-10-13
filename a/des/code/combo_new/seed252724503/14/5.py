import random

import numpy as np

GREY = 5
MAROON = 9


def _edge_colors(tile: np.ndarray):
    return {
        "top": int(tile[0, 1]),
        "bottom": int(tile[-1, 1]),
        "left": int(tile[1, 0]),
        "right": int(tile[1, -1]),
    }


def _rotations(tile: np.ndarray):
    return [np.rot90(tile, k) for k in range(4)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate every 4x4 tile so adjacent tile edges end up with matching colors.

    Input structure:
    - Grey separator lines divide the grid into equal 4x4 tiles.
    - Each tile has four uniform edge colors and a 2x2 core pattern.
    - Exactly one tile has maroon corner pixels and acts as the fixed anchor orientation.

    Rule:
    - Keep the anchor tile unchanged.
    - For every other tile, choose one of the four right-angle rotations so that all already-fixed neighboring edge colors match across shared borders.
    - Reassemble the grid with the same separators after every tile has been oriented.

    Key insight:
    - No tile can be oriented in isolation; the maroon-marked anchor fixes the first tile, and all other rotations are forced by edge-color compatibility with neighboring tiles.
    """
    line_rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    line_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    cells = []
    for r0, r1 in zip(line_rows, line_rows[1:]):
        row = []
        for c0, c1 in zip(line_cols, line_cols[1:]):
            row.append(grid[r0 + 1:r1, c0 + 1:c1].copy())
        cells.append(row)
    rows = len(cells)
    cols = len(cells[0])
    solved = [[None for _ in range(cols)] for _ in range(rows)]
    anchor = None
    for r in range(rows):
        for c in range(cols):
            tile = cells[r][c]
            if tile[0, 0] == MAROON and tile[0, -1] == MAROON and tile[-1, 0] == MAROON and tile[-1, -1] == MAROON:
                solved[r][c] = tile
                anchor = (r, c)
    pending = True
    while pending:
        pending = False
        for r in range(rows):
            for c in range(cols):
                if solved[r][c] is not None:
                    continue
                fits = []
                for candidate in _rotations(cells[r][c]):
                    edges = _edge_colors(candidate)
                    ok = True
                    if r > 0 and solved[r - 1][c] is not None:
                        ok &= _edge_colors(solved[r - 1][c])["bottom"] == edges["top"]
                    if r + 1 < rows and solved[r + 1][c] is not None:
                        ok &= _edge_colors(solved[r + 1][c])["top"] == edges["bottom"]
                    if c > 0 and solved[r][c - 1] is not None:
                        ok &= _edge_colors(solved[r][c - 1])["right"] == edges["left"]
                    if c + 1 < cols and solved[r][c + 1] is not None:
                        ok &= _edge_colors(solved[r][c + 1])["left"] == edges["right"]
                    if ok:
                        fits.append(candidate)
                if len(fits) == 1:
                    solved[r][c] = fits[0]
                    pending = True
    out = np.full_like(grid, GREY)
    for r in range(rows):
        for c in range(cols):
            assert solved[r][c] is not None, "tile rotations must be uniquely forced"
            r0, r1 = line_rows[r], line_rows[r + 1]
            c0, c1 = line_cols[c], line_cols[c + 1]
            out[r0 + 1:r1, c0 + 1:c1] = solved[r][c]
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    palette = [1, 2, 3, 4, 6, 7, 8]
    for _ in range(400):
        rows = rng.randint(2, 3)
        cols = rng.randint(2, 3)
        if rows * 5 + 1 > 30 or cols * 5 + 1 > 30:
            continue
        solved_tiles = []
        for r in range(rows):
            row_tiles = []
            for c in range(cols):
                top = _edge_colors(solved_tiles[r - 1][c])["bottom"] if r > 0 else rng.choice(palette)
                left = _edge_colors(row_tiles[c - 1])["right"] if c > 0 else rng.choice([p for p in palette if p != top])
                choices = [p for p in palette if p not in {top, left}]
                right = rng.choice(choices)
                bottom = rng.choice([p for p in choices if p != right])
                tile = np.zeros((4, 4), dtype=int)
                tile[0, :] = top
                tile[-1, :] = bottom
                tile[:, 0] = left
                tile[:, -1] = right
                core_colors = rng.sample([p for p in palette if p not in {left, right, top, bottom}], 2)
                tile[1:3, 1:3] = np.array([[core_colors[0], core_colors[1]], [core_colors[1], core_colors[0]]])
                row_tiles.append(tile)
            solved_tiles.append(row_tiles)
        ar = rng.randrange(rows)
        ac = rng.randrange(cols)
        solved_tiles[ar][ac][0, 0] = solved_tiles[ar][ac][0, -1] = solved_tiles[ar][ac][-1, 0] = solved_tiles[ar][ac][-1, -1] = MAROON
        grid = np.full((rows * 5 + 1, cols * 5 + 1), GREY, dtype=int)
        expected = np.full_like(grid, GREY)
        for r in range(rows):
            for c in range(cols):
                tile = solved_tiles[r][c].copy()
                expected[r * 5 + 1:r * 5 + 5, c * 5 + 1:c * 5 + 5] = tile
                if (r, c) != (ar, ac):
                    tile = np.rot90(tile, rng.randrange(4))
                grid[r * 5 + 1:r * 5 + 5, c * 5 + 1:c * 5 + 5] = tile
        try:
            if np.array_equal(main(grid), expected):
                return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate task 14/5")
