import random
from collections import deque

import numpy as np


BG = 0
GRID = 5


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _bbox(cells):
    pts = np.array(cells)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each outside fragment to an empty template slot by size, then fill that slot using the row/column color rule.

    Input structure:
    - A large grey template grid forms the biggest grey connected component.
    - The first template row contains column-color marker cells, and the first template column contains row-color marker cells.
    - Remaining template cells are either empty black slots or fully grey blocked slots.
    - Outside the template, each colored solid rectangle matches the size of exactly one empty slot.

    Rule:
    - Crop to the template bounding box for the output.
    - Read row colors from the left marker column and column colors from the top marker row.
    - Match each empty slot to the outside piece with the same height and width.
    - Fill the slot with the shared row/column color when they are equal; otherwise fill it with the matched piece color.

    Key insight:
    - Shape matching finds which piece belongs to which slot, but the slot's final color is decided by the visible row/column markers, not always by the piece itself.
    """
    g = np.asarray(grid, dtype=int)
    grid_comp = max(_components(g == GRID), key=len)
    r0, c0, r1, c1 = _bbox(grid_comp)
    crop = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    line_rows = [r for r in range(crop.shape[0]) if np.all(crop[r, :] == GRID)]
    line_cols = [c for c in range(crop.shape[1]) if np.all(crop[:, c] == GRID)]
    row_bands = list(zip(line_rows[:-1], line_rows[1:]))
    col_bands = list(zip(line_cols[:-1], line_cols[1:]))

    row_colors = []
    for a, b in row_bands[1:]:
        cell = crop[a + 1 : b, col_bands[0][0] + 1 : col_bands[0][1]]
        vals = [int(v) for v in np.unique(cell) if v not in (BG, GRID)]
        row_colors.append(vals[0])
    col_colors = []
    for a, b in col_bands[1:]:
        cell = crop[row_bands[0][0] + 1 : row_bands[0][1], a + 1 : b]
        vals = [int(v) for v in np.unique(cell) if v not in (BG, GRID)]
        col_colors.append(vals[0])

    pieces = {}
    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = BG
    for cells in _components(outside != BG):
        br0, bc0, br1, bc1 = _bbox(cells)
        color = int(g[cells[0]])
        pieces[(br1 - br0 + 1, bc1 - bc0 + 1)] = color

    for i, (ra, rb) in enumerate(row_bands[1:]):
        for j, (ca, cb) in enumerate(col_bands[1:]):
            inner = crop[ra + 1 : rb, ca + 1 : cb]
            if np.all(inner == BG):
                dims = inner.shape
                fill = row_colors[i] if row_colors[i] == col_colors[j] else pieces[dims]
                crop[ra + 1 : rb, ca + 1 : cb] = fill
    return crop


def _draw_template(grid, top, left, row_heights, col_widths, row_colors, col_colors, open_slots):
    total_h = 1 + sum(row_heights) + len(row_heights)
    total_w = 1 + sum(col_widths) + len(col_widths)
    r = top
    grid[r, left : left + total_w] = GRID
    for hh in row_heights:
        r += 1
        grid[r : r + hh, left] = GRID
        c = left
        for ww in col_widths:
            grid[r : r + hh, c + 1] = GRID
            c += ww + 1
        grid[r : r + hh, left + total_w - 1] = GRID
        r += hh
        grid[r, left : left + total_w] = GRID

    r = top + 1 + row_heights[0] + 1
    for hh, color in zip(row_heights[1:], row_colors):
        grid[r : r + hh, left + 1 : left + 1 + col_widths[0]] = color
        r += hh + 1
    c = left + 1 + col_widths[0] + 1
    for ww, color in zip(col_widths[1:], col_colors):
        grid[top + 1 : top + 1 + row_heights[0], c : c + ww] = color
        c += ww + 1

    rr = top + 1 + row_heights[0] + 1
    for i, hh in enumerate(row_heights[1:]):
        cc = left + 1 + col_widths[0] + 1
        for j, ww in enumerate(col_widths[1:]):
            if (i, j) not in open_slots:
                grid[rr : rr + hh, cc : cc + ww] = GRID
            cc += ww + 1
        rr += hh + 1
    return total_h, total_w


def generate_input() -> np.ndarray:
    for _ in range(300):
        rows = random.randint(2, 3)
        cols = random.randint(2, 3)
        row_heights = [random.randint(2, 3)] + random.sample([2, 3, 4], rows)
        col_widths = [random.randint(2, 3)] + random.sample([2, 3, 4], cols)
        total_h = 1 + sum(row_heights) + len(row_heights)
        total_w = 1 + sum(col_widths) + len(col_widths)
        h = total_h + random.randint(4, 10)
        w = total_w + random.randint(8, 14)
        if h > 30 or w > 30:
            continue
        grid = np.full((h, w), BG, dtype=int)
        top = random.randint(1, h - total_h - 1)
        left = random.randint(1, w - total_w - 1)
        row_colors = random.choices([1, 2, 3, 4, 6, 7, 8, 9], k=rows)
        col_colors = random.choices([1, 2, 3, 4, 6, 7, 8, 9], k=cols)
        slot_dims = [(row_heights[i + 1], col_widths[j + 1]) for i in range(rows) for j in range(cols)]
        if len(set(slot_dims)) < rows * cols - 1:
            continue
        open_slots = random.sample([(i, j) for i in range(rows) for j in range(cols)], random.randint(2, rows * cols))
        _draw_template(grid, top, left, row_heights, col_widths, row_colors, col_colors, set(open_slots))

        occ = grid != BG
        ok = True
        rr = top + 1 + row_heights[0] + 1
        slot_lookup = {}
        for i, hh in enumerate(row_heights[1:]):
            cc = left + 1 + col_widths[0] + 1
            for j, ww in enumerate(col_widths[1:]):
                slot_lookup[(i, j)] = (hh, ww)
                cc += ww + 1
            rr += hh + 1
        for ij in open_slots:
            hh, ww = slot_lookup[ij]
            color = random.choice([1, 2, 3, 4, 6, 7, 8, 9])
            placed = False
            for _ in range(100):
                pr = random.randint(1, h - hh - 1)
                pc = random.randint(1, w - ww - 1)
                if occ[pr : pr + hh, pc : pc + ww].any():
                    continue
                grid[pr : pr + hh, pc : pc + ww] = color
                occ[pr : pr + hh, pc : pc + ww] = True
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return grid
    raise RuntimeError("failed to generate input")
