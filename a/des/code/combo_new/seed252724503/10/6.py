from __future__ import annotations

import random

import numpy as np


BG = 0
LINE = 5
GEN_ATTEMPTS = 300


def _grid_structure(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = grid.shape
    row_lines = [r for r in range(h) if bool(np.all(grid[r, :] == LINE))]
    col_lines = [c for c in range(w) if bool(np.all(grid[:, c] == LINE))]
    assert row_lines and col_lines

    row_bands = [(row_lines[i] + 1, row_lines[i + 1] - 1) for i in range(len(row_lines) - 1)]
    col_bands = [(col_lines[i] + 1, col_lines[i + 1] - 1) for i in range(len(col_lines) - 1)]
    assert row_bands and col_bands
    assert all(b0 <= b1 for b0, b1 in row_bands)
    assert all(b0 <= b1 for b0, b1 in col_bands)
    return row_bands, col_bands


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the shared cell pattern and keep only each subcell's deviations from the consensus.

    Input structure:
    - Black background.
    - Gray full rows and full columns partition the grid into an equal matrix of subcells.
    - Each subcell contains a small pattern using black and one or two foreground colors.

    Rule:
    - For every pixel position inside the subcell coordinates, compute the most frequent color across all subcells.
    - Treat that per-position majority image as the common base pattern.
    - In each subcell, erase pixels that match the base pattern back to black and keep only pixels whose colors differ from the base.
    - Leave the gray partition lines unchanged.

    Key insight:
    - The repeated pattern is only a reference; the output intentionally preserves the exceptional pixels that vary from the cross-cell majority.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    row_bands, col_bands = _grid_structure(g)
    cell_h = row_bands[0][1] - row_bands[0][0] + 1
    cell_w = col_bands[0][1] - col_bands[0][0] + 1

    cells = []
    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            assert r1 - r0 + 1 == cell_h and c1 - c0 + 1 == cell_w
            cells.append(g[r0 : r1 + 1, c0 : c1 + 1].copy())

    base = np.zeros((cell_h, cell_w), dtype=int)
    for r in range(cell_h):
        for c in range(cell_w):
            counts = np.bincount([int(cell[r, c]) for cell in cells], minlength=10)
            base[r, c] = int(np.argmax(counts))

    out = g.copy()
    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            cell = g[r0 : r1 + 1, c0 : c1 + 1]
            diff = np.where(cell == base, BG, cell)
            out[r0 : r1 + 1, c0 : c1 + 1] = diff
    return out


def _random_base_pattern(cell_h: int, cell_w: int) -> np.ndarray:
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(1, 2))
    base = np.full((cell_h, cell_w), BG, dtype=int)
    for _ in range(random.randint(cell_h, cell_h + cell_w)):
        r = random.randrange(cell_h)
        c = random.randrange(cell_w)
        base[r, c] = random.choice(colors)
    if not np.any(base != BG):
        base[random.randrange(cell_h), random.randrange(cell_w)] = colors[0]
    return base


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        cell_h = random.randint(3, 5)
        cell_w = random.randint(3, 5)
        rows = random.randint(2, 4)
        cols = random.randint(2, 4)
        h = rows * cell_h + rows + 1
        w = cols * cell_w + cols + 1
        if h > 30 or w > 30:
            continue

        grid = np.full((h, w), BG, dtype=int)
        grid[:: cell_h + 1, :] = LINE
        grid[:, :: cell_w + 1] = LINE

        base = _random_base_pattern(cell_h, cell_w)
        for rr in range(rows):
            for cc in range(cols):
                cell = base.copy()
                for _ in range(random.randint(0, 3)):
                    r = random.randrange(cell_h)
                    c = random.randrange(cell_w)
                    options = [BG, 1, 2, 3, 4, 6, 7, 8, 9]
                    options.remove(int(cell[r, c]))
                    cell[r, c] = random.choice(options)
                r0 = 1 + rr * (cell_h + 1)
                c0 = 1 + cc * (cell_w + 1)
                grid[r0 : r0 + cell_h, c0 : c0 + cell_w] = cell
        return grid
    raise RuntimeError("failed to generate a valid instance")
