import random
from collections import deque

import numpy as np


BLACK = 0
WHITE = 9


def _runs(values):
    runs = []
    start = None
    for i, value in enumerate(values):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(values)))
    return runs


def _extract_cells(grid):
    black_rows = np.all(grid == BLACK, axis=1)
    black_cols = np.all(grid == BLACK, axis=0)
    row_lines = [s for s, e in _runs(black_rows) if e - s == 1]
    col_lines = [s for s, e in _runs(black_cols) if e - s == 1]

    if len(row_lines) >= 2:
        row_spans = [
            (row_lines[i] + 1, row_lines[i + 1])
            for i in range(len(row_lines) - 1)
            if row_lines[i] + 1 < row_lines[i + 1]
        ]
    else:
        row_spans = [(0, grid.shape[0])]

    if len(col_lines) >= 2:
        col_spans = [
            (col_lines[i] + 1, col_lines[i + 1])
            for i in range(len(col_lines) - 1)
            if col_lines[i] + 1 < col_lines[i + 1]
        ]
    else:
        col_spans = [(0, grid.shape[1])]

    if not row_spans or not col_spans:
        return [], [], np.empty((0, 0), dtype=int)

    colors = np.zeros((len(row_spans), len(col_spans)), dtype=int)
    for r, (r0, r1) in enumerate(row_spans):
        for c, (c0, c1) in enumerate(col_spans):
            patch = grid[r0:r1, c0:c1]
            vals, counts = np.unique(patch[patch != BLACK], return_counts=True)
            colors[r, c] = int(vals[np.argmax(counts)]) if len(vals) else BLACK
    return row_spans, col_spans, colors


def _components(colors):
    h, w = colors.shape
    comp = np.full((h, w), -1, dtype=int)
    comp_colors = []
    cells_by_comp = []
    cid = 0
    for sr in range(h):
        for sc in range(w):
            if comp[sr, sc] != -1:
                continue
            color = colors[sr, sc]
            comp[sr, sc] = cid
            q = deque([(sr, sc)])
            cells = []
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and comp[nr, nc] == -1 and colors[nr, nc] == color:
                        comp[nr, nc] = cid
                        q.append((nr, nc))
            comp_colors.append(int(color))
            cells_by_comp.append(cells)
            cid += 1
    return comp, comp_colors, cells_by_comp


def main(grid: np.ndarray) -> np.ndarray:
    """Plant white cross marks in each colored cell according to neighboring region count.

    Input structure:
    A two-dimensional ARC grid is divided by one-pixel black grid lines into a regular
    array of rectangular colored cells.  Each cell interior is a single non-black color,
    and same-colored cells that touch across a grid-line edge form a cell-level colored
    region.

    Rule:
    For every colored region, count how many distinct adjacent regions of different
    colors share a side with it at the cell level.  If this count is K > 0, draw a white
    cross centered in every cell of the region, extending K pixels up, down, left, and
    right, clipped to that cell interior.  Regions with K = 0 receive no mark, and black
    grid lines remain unchanged.

    Key insight:
    The transformation must reason over the hidden cell grid: first group cells into
    color components, then count neighboring components, then render the count locally
    as a cross inside each cell.
    """
    out = np.array(grid, copy=True)
    row_spans, col_spans, colors = _extract_cells(out)
    if colors.size == 0:
        return out

    comp, comp_colors, cells_by_comp = _components(colors)
    neighbor_sets = [set() for _ in comp_colors]
    h, w = colors.shape
    for r in range(h):
        for c in range(w):
            cid = comp[r, c]
            for dr, dc in ((1, 0), (0, 1)):
                nr, nc = r + dr, c + dc
                if nr < h and nc < w:
                    nid = comp[nr, nc]
                    if nid != cid and comp_colors[nid] != comp_colors[cid]:
                        neighbor_sets[cid].add(nid)
                        neighbor_sets[nid].add(cid)

    for cid, cells in enumerate(cells_by_comp):
        arm = len(neighbor_sets[cid])
        if arm == 0:
            continue
        for r, c in cells:
            r0, r1 = row_spans[r]
            c0, c1 = col_spans[c]
            cr = (r0 + r1 - 1) // 2
            cc = (c0 + c1 - 1) // 2
            rr0 = max(r0, cr - arm)
            rr1 = min(r1 - 1, cr + arm)
            cc0 = max(c0, cc - arm)
            cc1 = min(c1 - 1, cc + arm)
            out[rr0:rr1 + 1, cc] = WHITE
            out[cr, cc0:cc1 + 1] = WHITE
    return out


def _smooth_color_layout(rows, cols, palette):
    color_grid = np.array([[random.choice(palette) for _ in range(cols)] for _ in range(rows)], dtype=int)
    steps = random.randint(0, max(1, rows * cols * 2))
    for _ in range(steps):
        r = random.randrange(rows)
        c = random.randrange(cols)
        nr, nc = random.choice(
            [(max(0, r - 1), c), (min(rows - 1, r + 1), c), (r, max(0, c - 1)), (r, min(cols - 1, c + 1))]
        )
        color_grid[r, c] = color_grid[nr, nc]
    return color_grid


def generate_input() -> np.ndarray:
    rows = random.randint(1, 5)
    cols = random.randint(1, 5)
    cell_h = random.randint(3, 5)
    cell_w = random.randint(3, 5)

    while rows * cell_h + rows + 1 > 30:
        rows -= 1
    while cols * cell_w + cols + 1 > 30:
        cols -= 1

    palette_pool = [1, 2, 3, 4, 5, 6, 7, 8]
    palette = random.sample(palette_pool, random.randint(2, min(5, len(palette_pool))))

    if random.random() < 0.18:
        color_grid = np.full((rows, cols), random.choice(palette), dtype=int)
    elif random.random() < 0.25 and len(palette) >= 2:
        color_grid = np.fromfunction(lambda r, c: (r + c) % 2, (rows, cols), dtype=int)
        color_grid = np.vectorize(lambda x: palette[int(x)])(color_grid).astype(int)
    else:
        color_grid = _smooth_color_layout(rows, cols, palette)

    height = rows * cell_h + rows + 1
    width = cols * cell_w + cols + 1
    grid = np.zeros((height, width), dtype=int)
    for r in range(rows):
        for c in range(cols):
            r0 = 1 + r * (cell_h + 1)
            c0 = 1 + c * (cell_w + 1)
            grid[r0:r0 + cell_h, c0:c0 + cell_w] = color_grid[r, c]
    return grid
