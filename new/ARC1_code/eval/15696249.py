from __future__ import annotations

import numpy as np
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeat the input tile across the output band indicated by the input's single solid row or column.

    Input structure
    The input is an n x n tile with exactly one full row or full column made of
    one repeated color.

    Rule
    Create an n-by-n block canvas of black n x n blocks. If the solid guide is
    row r, copy the input tile into every block of block row r. If the guide is
    column c, copy the input tile into every block of block column c.

    Key insight
    The solid row or column is a control line: its orientation and index select
    the row-band or column-band of the enlarged tiled canvas that remains visible.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height == width, "expected a square tile"
    assert height > 0, "expected a nonempty tile"

    n = height
    solid_rows = [r for r in range(n) if np.all(g[r, :] == int(g[r, 0]))]
    solid_cols = [c for c in range(n) if np.all(g[:, c] == int(g[0, c]))]
    assert len(solid_rows) + len(solid_cols) == 1, "expected exactly one solid guide line"

    out = np.zeros((n * n, n * n), dtype=int)
    if solid_rows:
        row = solid_rows[0]
        out[row * n : (row + 1) * n, :] = np.tile(g, (1, n))
    else:
        col = solid_cols[0]
        out[:, col * n : (col + 1) * n] = np.tile(g, (n, 1))

    return out

# --------------- generator (o5x) ---------------

# Tile and palette bounds.
GEN_TILE_SIZE = 3
GEN_COLOR_RANGE = range(1, 10)
GEN_COLOR_COUNT = 3

# Guide layout and retry budget.
GEN_GUIDE_ORIENTATIONS = ("row", "col")
GEN_MAX_ATTEMPTS = 300


def _gen_mixed_line(colors: tuple[int, int], n: int) -> list[int]:
    a, b = colors
    split = random.randint(1, n - 1)
    vals = [a] * split + [b] * (n - split)
    random.shuffle(vals)
    return vals


def _gen_expected_output(grid: np.ndarray) -> np.ndarray | None:
    n = grid.shape[0]
    solid_rows = [r for r in range(n) if np.all(grid[r, :] == grid[r, 0])]
    solid_cols = [c for c in range(n) if np.all(grid[:, c] == grid[0, c])]
    if len(solid_rows) + len(solid_cols) != 1:
        return None

    out = np.zeros((n * n, n * n), dtype=int)
    if solid_rows:
        r = solid_rows[0]
        out[r * n : (r + 1) * n, :] = np.tile(grid, (1, n))
    else:
        c = solid_cols[0]
        out[:, c * n : (c + 1) * n] = np.tile(grid, (n, 1))
    return out


def _paint_guided_grid(grid, orientation, index, guide_color, fill_colors) -> None:
    n = grid.shape[0]
    if orientation == "row":
        grid[index, :] = guide_color
        for r in range(n):
            if r != index:
                grid[r, :] = _gen_mixed_line(fill_colors, n)
    else:
        grid[:, index] = guide_color
        for c in range(n):
            if c != index:
                grid[:, c] = _gen_mixed_line(fill_colors, n)


def _gen_attempt() -> np.ndarray | None:
    n = GEN_TILE_SIZE
    colors = random.sample(GEN_COLOR_RANGE, GEN_COLOR_COUNT)
    guide_color = colors[0]
    fill_colors = (colors[1], colors[2])

    grid = np.empty((n, n), dtype=int)
    orientation = random.choice(GEN_GUIDE_ORIENTATIONS)
    index = random.randrange(n)
    _paint_guided_grid(grid, orientation, index, guide_color, fill_colors)

    non_guide = np.delete(grid, index, axis=0 if orientation == "row" else 1)
    if sorted(set(map(int, grid.flat))) != sorted(colors):
        return None
    if guide_color in set(map(int, non_guide.flat)):
        return None
    expected = _gen_expected_output(grid)
    if expected is None:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        inp = _gen_attempt()
        if inp is None:
            continue
        expected = _gen_expected_output(inp)
        if expected is None:
            continue
        try:
            out = main(inp)
        except AssertionError:
            continue
        if np.array_equal(out, expected):
            return inp
    raise RuntimeError("failed to generate valid input")
