import random
from collections import Counter, deque

import numpy as np


BG = 0


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cur = []
            while q:
                x, y = q.popleft()
                cur.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(cur)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    out = np.zeros_like(grid)
    for comp in _components(grid != BG):
        colors = Counter(grid[r, c] for r, c in comp)
        order = [c for c, _ in sorted(colors.items(), key=lambda kv: (-kv[1], kv[0]))]
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        cr = (min(rows) + max(rows)) // 2
        cc = (min(cols) + max(cols)) // 2
        for layer, color in enumerate(order[::-1]):
            rad = layer
            if layer == 0:
                out[cr, cc] = color
            else:
                r0, r1 = cr - rad, cr + rad
                c0, c1 = cc - rad, cc + rad
                if 0 <= r0 and r1 < out.shape[0] and 0 <= c0 and c1 < out.shape[1]:
                    out[r0, c0 : c1 + 1] = color
                    out[r1, c0 : c1 + 1] = color
                    out[r0 : r1 + 1, c0] = color
                    out[r0 : r1 + 1, c1] = color
    return out


def generate_input() -> np.ndarray:
    h = random.randint(18, 26)
    w = random.randint(18, 26)
    grid = np.zeros((h, w), dtype=int)
    regions = random.randint(2, 4)
    palette = [1, 2, 3, 4, 6, 7, 8, 9]
    for _ in range(regions):
        top = random.randint(1, h - 8)
        left = random.randint(1, w - 8)
        rh = random.randint(4, 6)
        rw = random.randint(4, 6)
        colors = random.sample(palette, random.randint(2, 4))
        sizes = random.sample(range(2, rh * rw // 2 + 2), len(colors))
        cells = [(r, c) for r in range(top, top + rh) for c in range(left, left + rw)]
        random.shuffle(cells)
        anchor = cells.pop()
        comp_cells = [anchor]
        grid[anchor] = colors[0]
        used = {anchor}
        frontier = [anchor]
        for color, size in zip(colors, sizes):
            chosen = []
            while frontier and len(chosen) < size:
                x, y = random.choice(frontier)
                nbrs = []
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if top <= nx < top + rh and left <= ny < left + rw and (nx, ny) not in used:
                        nbrs.append((nx, ny))
                if not nbrs:
                    frontier.remove((x, y))
                    continue
                cell = random.choice(nbrs)
                used.add(cell)
                frontier.append(cell)
                chosen.append(cell)
                comp_cells.append(cell)
                grid[cell] = color
            frontier.extend(chosen)
        for r, c in comp_cells:
            if grid[r, c] == BG:
                grid[r, c] = random.choice(colors)
    return grid
