import random

import numpy as np


def _segments(indices: list[int], length: int) -> list[tuple[int, int]]:
    bounds = [-1] + indices + [length]
    return [(bounds[i] + 1, bounds[i + 1]) for i in range(len(bounds) - 1)]


def main(grid: np.ndarray) -> np.ndarray:
    """Use red and green marker runs to choose filled lattice cells, color them from row controls, and paint red-green overlaps magenta."""
    h, w = grid.shape
    sep_rows = [r for r in range(h) if np.all(grid[r] == 8)]
    sep_cols = [c for c in range(w) if np.all(grid[:, c] == 8)]
    row_segs = _segments(sep_rows, h)
    col_segs = _segments(sep_cols, w)
    cell_rows, cell_cols = len(row_segs), len(col_segs)
    cell_vals = np.zeros((cell_rows, cell_cols), dtype=int)
    for i, (r0, r1) in enumerate(row_segs):
        for j, (c0, c1) in enumerate(col_segs):
            cell = grid[r0:r1, c0:c1]
            vals = [int(v) for v in np.unique(cell) if int(v) not in (0, 8)]
            cell_vals[i, j] = vals[0] if vals else 0

    controls = {r: int(cell_vals[r, 0]) for r in range(cell_rows) if int(cell_vals[r, 0]) in (1, 4, 7)}
    zones: dict[int, set[tuple[int, int]]] = {2: set(), 3: set()}
    for marker_color in (2, 3):
        positions = [(r, c) for r in range(cell_rows) for c in range(cell_cols) if int(cell_vals[r, c]) == marker_color]
        by_row: dict[int, list[int]] = {}
        by_col: dict[int, list[int]] = {}
        for r, c in positions:
            by_row.setdefault(r, []).append(c)
            by_col.setdefault(c, []).append(r)
        for r, cols in by_row.items():
            for c in range(min(cols), max(cols) + 1):
                zones[marker_color].add((r, c))
        for c, rows in by_col.items():
            for r in range(min(rows), max(rows) + 1):
                zones[marker_color].add((r, c))

    out = grid.copy()
    for r in range(cell_rows):
        for c in range(cell_cols):
            red = (r, c) in zones[2]
            green = (r, c) in zones[3]
            if not red and not green:
                continue
            color = 6 if red and green else None
            if color is None:
                if r in controls:
                    color = controls[r]
                else:
                    rows = sorted(controls)
                    best = min(rows, key=lambda rr: (abs(rr - r), rr > r))
                    color = controls[best]
            r0, r1 = row_segs[r]
            c0, c1 = col_segs[c]
            out[r0:r1, c0:c1] = color
    for r in sep_rows:
        out[r, :] = 8
    for c in sep_cols:
        out[:, c] = 8
    return out


def _pixel_grid_from_cells(cells: np.ndarray, cell_size: int) -> np.ndarray:
    rows, cols = cells.shape
    h = rows * cell_size + (rows - 1)
    w = cols * cell_size + (cols - 1)
    grid = np.zeros((h, w), dtype=int)
    for r in range(rows):
        for c in range(cols):
            r0 = r * (cell_size + 1)
            c0 = c * (cell_size + 1)
            grid[r0 : r0 + cell_size, c0 : c0 + cell_size] = cells[r, c]
    for r in range(cell_size, h, cell_size + 1):
        grid[r, :] = 8
    for c in range(cell_size, w, cell_size + 1):
        grid[:, c] = 8
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    while True:
        rows = rng.choice([4, 5])
        cols = rng.choice([5, 6])
        cells = np.zeros((rows, cols), dtype=int)
        red_rows = rng.sample(range(rows), k=rng.randint(1, max(1, rows - 2)))
        green_rows = rng.sample([r for r in range(rows) if r not in red_rows], k=1)
        control_palette = [1, 4, 7]
        for r in red_rows + green_rows:
            cells[r, 0] = rng.choice(control_palette)

        red_r1 = red_rows[0]
        red_r2 = rng.choice(red_rows)
        green_r1 = green_rows[0]
        green_r2 = rng.choice([r for r in range(rows) if r != green_r1])
        c_shared = rng.randint(2, cols - 2)
        red_c_other = rng.randint(1, cols - 1)
        green_c_other = rng.randint(1, cols - 1)
        if red_c_other == c_shared:
            red_c_other = cols - 1 if c_shared != cols - 1 else 1
        if green_c_other == c_shared:
            green_c_other = cols - 1 if c_shared != cols - 1 else 1

        red_marks = {(red_r1, min(red_c_other, c_shared)), (red_r1, max(red_c_other, c_shared)), (red_r2, c_shared)}
        green_marks = {(green_r1, min(green_c_other, c_shared)), (green_r1, max(green_c_other, c_shared)), (green_r2, c_shared)}
        if red_marks & green_marks:
            continue
        for r, c in red_marks:
            if c == 0:
                continue
            cells[r, c] = 2
        for r, c in green_marks:
            if c == 0 or cells[r, c] != 0:
                break
            cells[r, c] = 3
        else:
            if not np.any(cells == 2) or not np.any(cells == 3):
                continue
            if not any((r, c_shared) in red_marks for r in red_rows) or not any((r, c_shared) in green_marks for r in [green_r1, green_r2]):
                continue
            return _pixel_grid_from_cells(cells, rng.choice([2, 3]))
