from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

GEN_ATTEMPTS = 500


def _effective_rows(grid: np.ndarray) -> list[int]:
    rows: list[int] = []
    for r in range(grid.shape[0]):
        if any(grid[r, c] == BLUE and grid[r, c + 1] == BLUE for c in range(grid.shape[1] - 1)):
            rows.append(r)
    return rows


def _effective_cols(grid: np.ndarray) -> list[int]:
    cols: list[int] = []
    for c in range(grid.shape[1]):
        if any(grid[r, c] == BLUE and grid[r + 1, c] == BLUE for r in range(grid.shape[0] - 1)):
            cols.append(c)
    return cols


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair broken blue grid lines in red, then invert black/yellow occupancy only in tiles marked by green.
    """
    g = np.asarray(grid, dtype=int)
    rows = _effective_rows(g)
    cols = _effective_cols(g)
    assert len(rows) >= 2 and len(cols) >= 2, "expected at least a 2x2 line lattice"

    r_min, r_max = min(rows), max(rows)
    c_min, c_max = min(cols), max(cols)
    out = g.copy()

    for r in rows:
        for c in range(c_min, c_max + 1):
            if out[r, c] == BLACK:
                out[r, c] = RED
    for c in cols:
        for r in range(r_min, r_max + 1):
            if out[r, c] == BLACK:
                out[r, c] = RED

    for top, bottom in zip(rows, rows[1:]):
        if bottom - top <= 1:
            continue
        for left, right in zip(cols, cols[1:]):
            if right - left <= 1:
                continue
            rr = slice(top + 1, bottom)
            cc = slice(left + 1, right)
            tile = out[rr, cc]
            if not np.any(tile == GREEN):
                continue
            tile_copy = tile.copy()
            tile[tile_copy == BLACK] = YELLOW
            tile[tile_copy == YELLOW] = BLACK
            tile[tile_copy == GREEN] = BLACK

    return out


def _choose_lines(limit: int, count: int) -> list[int]:
    for _ in range(200):
        lines = sorted(random.sample(range(1, limit - 1), count))
        if all(b - a >= 3 for a, b in zip(lines, lines[1:])):
            return lines
    return list(range(1, 1 + count * 3, 3))[:count]


def _has_adjacent_blue_on_row(grid: np.ndarray, r: int) -> bool:
    return any(grid[r, c] == BLUE and grid[r, c + 1] == BLUE for c in range(grid.shape[1] - 1))


def _has_adjacent_blue_on_col(grid: np.ndarray, c: int) -> bool:
    return any(grid[r, c] == BLUE and grid[r + 1, c] == BLUE for r in range(grid.shape[0] - 1))


def _break_lines(grid: np.ndarray, rows: list[int], cols: list[int]) -> None:
    r_min, r_max = min(rows), max(rows)
    c_min, c_max = min(cols), max(cols)
    for r in rows:
        candidates = [c for c in range(c_min, c_max + 1) if c not in cols]
        random.shuffle(candidates)
        breaks = random.randint(1, min(3, max(1, len(candidates) // 4)))
        made = 0
        for c in candidates:
            old = int(grid[r, c])
            grid[r, c] = BLACK
            if _has_adjacent_blue_on_row(grid, r):
                made += 1
            else:
                grid[r, c] = old
            if made >= breaks:
                break
    for c in cols:
        candidates = [r for r in range(r_min, r_max + 1) if r not in rows]
        random.shuffle(candidates)
        breaks = random.randint(1, min(3, max(1, len(candidates) // 4)))
        made = 0
        for r in candidates:
            old = int(grid[r, c])
            grid[r, c] = BLACK
            if _has_adjacent_blue_on_col(grid, c):
                made += 1
            else:
                grid[r, c] = old
            if made >= breaks:
                break


def _tile_bounds(rows: list[int], cols: list[int]) -> list[tuple[int, int, int, int]]:
    bounds = []
    for top, bottom in zip(rows, rows[1:]):
        for left, right in zip(cols, cols[1:]):
            if bottom - top > 1 and right - left > 1:
                bounds.append((top + 1, bottom - 1, left + 1, right - 1))
    return bounds


def _paint_tile_marks(grid: np.ndarray, bounds: tuple[int, int, int, int], marked: bool) -> None:
    r0, r1, c0, c1 = bounds
    cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    random.shuffle(cells)
    yellow_count = random.randint(0, min(5, max(0, len(cells) - 1)))
    for r, c in cells[:yellow_count]:
        grid[r, c] = YELLOW
    if marked:
        marker_cell = cells[yellow_count] if yellow_count < len(cells) else cells[-1]
        grid[marker_cell] = GREEN


def _valid_generated(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False
    return isinstance(out, np.ndarray) and out.ndim == 2 and max(out.shape) <= 30


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        height = random.randint(13, 21)
        width = random.randint(13, 21)
        row_count = random.randint(3, min(5, (height - 1) // 3))
        col_count = random.randint(3, min(5, (width - 1) // 3))
        rows = _choose_lines(height, row_count)
        cols = _choose_lines(width, col_count)

        grid = np.full((height, width), BLACK, dtype=int)
        for r in rows:
            grid[r, min(cols) : max(cols) + 1] = BLUE
        for c in cols:
            grid[min(rows) : max(rows) + 1, c] = BLUE

        _break_lines(grid, rows, cols)

        tiles = _tile_bounds(rows, cols)
        if len(tiles) < 2:
            continue
        random.shuffle(tiles)
        marked_count = random.randint(1, min(3, len(tiles)))
        marked_tiles = set(tiles[:marked_count])
        filler_count = random.randint(0, min(3, len(tiles) - marked_count))
        filler_tiles = set(tiles[marked_count : marked_count + filler_count])

        for bounds in marked_tiles:
            _paint_tile_marks(grid, bounds, marked=True)
        for bounds in filler_tiles:
            _paint_tile_marks(grid, bounds, marked=False)

        if _valid_generated(grid):
            return grid
    raise RuntimeError("failed to generate puzzle 030/6 input")

