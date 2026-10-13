import random
from collections import deque
import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not seen[r, c]:
                q = deque([(r, c)])
                seen[r, c] = True
                comp = []
                while q:
                    x, y = q.popleft()
                    comp.append((x, y))
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        xx, yy = x + dx, y + dy
                        if 0 <= xx < h and 0 <= yy < w and mask[xx, yy] and not seen[xx, yy]:
                            seen[xx, yy] = True
                            q.append((xx, yy))
                comps.append(comp)
    return comps


def _norm(comp):
    pts = np.array(comp)
    r0, c0 = pts.min(axis=0)
    return frozenset((int(r - r0), int(c - c0)) for r, c in comp)


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    blue = np.argwhere(grid == 1)
    br0, bc0 = blue.min(axis=0)
    br1, bc1 = blue.max(axis=0)
    rows = sorted({r for r in range(grid.shape[0]) if np.any((grid[r, :-1] == 1) & (grid[r, 1:] == 1))})
    cols = sorted({c for c in range(grid.shape[1]) if np.any((grid[:-1, c] == 1) & (grid[1:, c] == 1))})
    gaps = {(r, c) for r in range(int(br0), int(br1) + 1) for c in range(int(bc0), int(bc1) + 1) if grid[r, c] == 0 and (r in rows or c in cols)}
    objects = []
    for comp in _components((grid != 0) & (grid != 1)):
        pts = np.array(comp)
        rmean, cmean = pts.mean(axis=0)
        objects.append((comp, int(grid[comp[0][0], comp[0][1]]), rmean, cmean, _norm(comp)))
    placeholders = [o for o in objects if br0 < o[2] < br1 and bc0 < o[3] < bc1]
    templates = [o for o in objects if not (br0 < o[2] < br1 and bc0 < o[3] < bc1)]
    cell_for_gap = {}
    for comp, _, pr, pc, shape in placeholders:
        match = next(t for t in templates if t[4] == shape)
        _, tcolor, tr, tc, _ = match
        vertical_axis = abs(tc - pc) > abs(tr - pr)
        rr = max(i for i in range(len(rows) - 1) if rows[i] < pr < rows[i + 1])
        cc = max(i for i in range(len(cols) - 1) if cols[i] < pc < cols[i + 1])
        r0, r1 = rows[rr] + 1, rows[rr + 1] - 1
        c0, c1 = cols[cc] + 1, cols[cc + 1] - 1
        for r, c in comp:
            mr, mc = (r, c0 + c1 - c) if vertical_axis else (r0 + r1 - r, c)
            if out[mr, mc] == 0:
                out[mr, mc] = int(grid[r, c])
        for gap in gaps:
            gr, gc = gap
            if (gr in (rows[rr], rows[rr + 1]) and c0 <= gc <= c1) or (gc in (cols[cc], cols[cc + 1]) and r0 <= gr <= r1):
                cell_for_gap[gap] = tcolor
    for (r, c), color in cell_for_gap.items():
        if out[r, c] == 0:
            out[r, c] = color
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((16, 16), dtype=int)
    rows = [3, 8, 13]
    cols = [3, 8, 13]
    for r in rows:
        grid[r, cols[0] : cols[-1] + 1] = 1
    for c in cols:
        grid[rows[0] : rows[-1] + 1, c] = 1
    specs = [
        ((4, 4), (4, 0), (4, 3)),
        ((4, 9), (0, 9), (3, 9)),
        ((9, 4), (14, 4), (13, 4)),
    ]
    shapes = [[(0, 0), (1, 0), (1, 1)], [(0, 0), (0, 1), (1, 0)], [(0, 1), (1, 0), (1, 1)]]
    random.shuffle(specs)
    random.shuffle(shapes)
    pcolors = random.sample([5, 6, 7, 8, 9], 3)
    tcolors = random.sample([2, 3, 4, 5, 6, 7, 8, 9], 3)
    count = random.randint(2, 3)
    for i in range(count):
        (r0, c0), (tr, tc), gap = specs[i]
        shape = shapes[i]
        pcolor = pcolors[i]
        for dr, dc in shape:
            grid[r0 + dr, c0 + dc] = pcolor
        for dr, dc in shape:
            if 0 <= tr + dr < 16 and 0 <= tc + dc < 16:
                grid[tr + dr, tc + dc] = tcolors[i]
        if grid[gap] == 1:
            grid[gap] = 0
    return grid
