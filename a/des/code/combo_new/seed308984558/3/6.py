import random
from collections import Counter, deque

import numpy as np


BG = 0
ROT = {3: 0, 4: 3, 6: 2, 8: 1}


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
                x, y = q.popleft()
                cells.append((x, y))
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(cells)
    return comps


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each region's object by the angle encoded by the majority color among its surrounding separator lines.

    Input structure:
    - Colored full-span separator rows and columns partition the black canvas into rectangular regions.
    - Each region contains one small asymmetric object.
    - Separator colors encode quarter-turn rotations.

    Rule:
    - For each region, collect the colors of the existing surrounding separator lines and choose their majority color.
    - Rotate the region's object accordingly.
    - Recenter the rotated bbox in the region and leave the separators unchanged.

    Key insight:
    - The instruction color comes from voting across multiple boundary lines, not from reading a single border.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    sep_rows = [r for r in range(g.shape[0]) if len(set(g[r])) == 1 and g[r, 0] != BG]
    sep_cols = [c for c in range(g.shape[1]) if len(set(g[:, c])) == 1 and g[0, c] != BG]
    row_bounds = [-1] + sep_rows + [g.shape[0]]
    col_bounds = [-1] + sep_cols + [g.shape[1]]
    sep_set = set(np.unique(g[sep_rows + [0]][:, sep_cols + [0]]) if sep_rows or sep_cols else [])
    for i in range(len(row_bounds) - 1):
        for j in range(len(col_bounds) - 1):
            r0, r1 = row_bounds[i] + 1, row_bounds[i + 1] - 1
            c0, c1 = col_bounds[j] + 1, col_bounds[j + 1] - 1
            if r0 > r1 or c0 > c1:
                continue
            region = g[r0 : r1 + 1, c0 : c1 + 1]
            obj_mask = region != BG
            if not np.any(obj_mask):
                continue
            colors = []
            if row_bounds[i] >= 0:
                colors.append(int(g[row_bounds[i], c0]))
            if row_bounds[i + 1] < g.shape[0]:
                colors.append(int(g[row_bounds[i + 1], c0]))
            if col_bounds[j] >= 0:
                colors.append(int(g[r0, col_bounds[j]]))
            if col_bounds[j + 1] < g.shape[1]:
                colors.append(int(g[r0, col_bounds[j + 1]]))
            rot = ROT[Counter(colors).most_common(1)[0][0]]
            cells = max(_components(obj_mask), key=len)
            a0, b0, a1, b1 = _bbox(cells)
            patch = region[a0 : a1 + 1, b0 : b1 + 1]
            mask = patch != BG
            vals = patch.copy()
            vals[~mask] = 0
            vals = np.rot90(vals, rot)
            region[:] = np.where(region == BG, BG, region)
            region[region != BG] = BG
            rr = (region.shape[0] - vals.shape[0]) // 2
            cc = (region.shape[1] - vals.shape[1]) // 2
            region[rr : rr + vals.shape[0], cc : cc + vals.shape[1]][vals != 0] = vals[vals != 0]
            out[r0 : r1 + 1, c0 : c1 + 1] = region
    return out


SHAPE = np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1]])


def generate_input() -> np.ndarray:
    h, w = random.randint(18, 24), random.randint(18, 24)
    g = np.zeros((h, w), dtype=int)
    hr = sorted(random.sample(range(5, h - 5), 1))
    vc = sorted(random.sample(range(5, w - 5), 1))
    for r in hr:
        g[r, :] = random.choice(list(ROT))
    for c in vc:
        g[:, c] = random.choice(list(ROT))
    rows = [-1] + hr + [h]
    cols = [-1] + vc + [w]
    for i in range(len(rows) - 1):
        for j in range(len(cols) - 1):
            r0, r1 = rows[i] + 1, rows[i + 1] - 1
            c0, c1 = cols[j] + 1, cols[j + 1] - 1
            if r1 - r0 + 1 < SHAPE.shape[0] or c1 - c0 + 1 < SHAPE.shape[1]:
                continue
            color = random.choice([1, 2, 5, 7, 9])
            rr = r0 + (r1 - r0 + 1 - SHAPE.shape[0]) // 2
            cc = c0 + (c1 - c0 + 1 - SHAPE.shape[1]) // 2
            g[rr : rr + SHAPE.shape[0], cc : cc + SHAPE.shape[1]][SHAPE == 1] = color
    return g
