import random

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
ORANGE = 7
FRAME_COLORS = [1, 2, 3, 6, 8, 9]


def _full_lines(grid: np.ndarray, color: int):
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == color)]
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == color)]
    return rows, cols


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each colored region with the template keyed by its yellow marker offset.

    Input structure:
    - Grey full-width/full-height grid lines partition the canvas into equal cells.
    - Left cells have colored borders, black interiors, and one yellow marker.
    - The rightmost template column has grey borders, orange interior motifs, and one
      yellow marker per template cell.

    Rule:
    - Use the yellow marker offset inside each template cell as a lookup key for its
      orange motif mask.
    - For each non-template cell, find the template with the same yellow marker offset.
    - Copy that template's orange motif into the cell interior, recolored to the cell's
      border color, and remove the yellow marker from the output.
    - Keep the template column unchanged.

    Key insight:
    - Marker position is the index: matching relative yellow coordinates selects a
      template independently of color, and the chosen shape is then recolored by the
      destination frame.
    """
    rows, cols = _full_lines(grid, GREY)
    out = grid.copy()
    templates = {}
    template_j = len(cols) - 2
    for i in range(len(rows) - 1):
        r0, r1 = rows[i], rows[i + 1]
        c0, c1 = cols[template_j], cols[template_j + 1]
        patch = grid[r0 + 1 : r1, c0 + 1 : c1]
        inner = patch[1:-1, 1:-1]
        y = np.argwhere(inner == YELLOW)
        if len(y) == 1:
            templates[tuple(y[0])] = inner == ORANGE
    for i in range(len(rows) - 1):
        for j in range(template_j):
            r0, r1 = rows[i], rows[i + 1]
            c0, c1 = cols[j], cols[j + 1]
            patch = grid[r0 + 1 : r1, c0 + 1 : c1]
            if patch.shape[0] < 3 or patch.shape[1] < 3:
                continue
            inner = patch[1:-1, 1:-1]
            y = np.argwhere(inner == YELLOW)
            if len(y) != 1:
                continue
            key = tuple(y[0])
            border_color = int(patch[0, 1])
            out[r0 + 2 : r1 - 1, c0 + 2 : c1 - 1] = BLACK
            mask = templates.get(key, next(iter(templates.values())))
            out[r0 + 2 : r1 - 1, c0 + 2 : c1 - 1][mask] = border_color
    return out


def generate_input() -> np.ndarray:
    cell = 6
    inner_n = cell - 3
    rows_n = random.randint(2, 3)
    cols_n = random.randint(2, 3)
    h = rows_n * (cell + 1) + 1
    w = (cols_n + 1) * (cell + 1) + 1
    grid = np.zeros((h, w), dtype=int)
    for r in range(0, h, cell + 1):
        grid[r, :] = GREY
    for c in range(0, w, cell + 1):
        grid[:, c] = GREY
    offsets = random.sample([(r, c) for r in range(inner_n) for c in range(inner_n)], rows_n)
    patterns = []
    for _ in range(rows_n):
        pat = np.zeros((inner_n, inner_n), dtype=bool)
        for _ in range(random.randint(3, 5)):
            pat[random.randint(0, inner_n - 1), random.randint(0, inner_n - 1)] = True
        patterns.append(pat)
    for i in range(rows_n):
        for j in range(cols_n):
            r0 = i * (cell + 1)
            c0 = j * (cell + 1)
            patch = grid[r0 + 1 : r0 + cell, c0 + 1 : c0 + cell]
            color = random.choice(FRAME_COLORS)
            patch[0, :] = color
            patch[-1, :] = color
            patch[:, 0] = color
            patch[:, -1] = color
            inner = patch[1:-1, 1:-1]
            inner[:] = BLACK
            idx = random.randrange(len(offsets))
            inner[offsets[idx]] = YELLOW
    template_col = cols_n
    for i in range(rows_n):
        r0 = i * (cell + 1)
        c0 = template_col * (cell + 1)
        patch = grid[r0 + 1 : r0 + cell, c0 + 1 : c0 + cell]
        patch[0, :] = GREY
        patch[-1, :] = GREY
        patch[:, 0] = GREY
        patch[:, -1] = GREY
        inner = patch[1:-1, 1:-1]
        inner[:] = BLACK
        idx = i % len(offsets)
        inner[patterns[idx]] = ORANGE
        inner[offsets[idx]] = YELLOW
    return grid
