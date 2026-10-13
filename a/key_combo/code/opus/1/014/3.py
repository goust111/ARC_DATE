import random
from collections import deque
import numpy as np


def _components(grid, color):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and not seen[nx, ny] and grid[nx, ny] == color:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(comp)
    return comps


def _bbox(comp):
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    return min(rs), min(cs), max(rs), max(cs)


def _submask(comp, r0, r1):
    cells = [(r, c) for r, c in comp if r0 <= r <= r1]
    if not cells:
        return np.zeros((0, 0), dtype=int)
    br0, bc0, br1, bc1 = _bbox(cells)
    mask = np.zeros((r1 - r0 + 1, bc1 - bc0 + 1), dtype=int)
    for r, c in cells:
        mask[r - r0, c - bc0] = 1
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    comps = _components(grid, 3)
    comps.sort(key=lambda comp: _bbox(comp)[1])
    for a, b in zip(comps, comps[1:]):
        ar0, ac0, ar1, ac1 = _bbox(a)
        br0, bc0, br1, bc1 = _bbox(b)
        if ac1 >= bc0:
            continue
        rr0, rr1 = max(ar0, br0), min(ar1, br1)
        if rr0 > rr1:
            continue
        ma = _submask(a, rr0, rr1)
        mb = _submask(b, rr0, rr1)
        if ma.shape == mb.shape and ma.size and np.array_equal(np.fliplr(ma), mb):
            out[rr0:rr1 + 1, ac1 + 1:bc0] = np.where(out[rr0:rr1 + 1, ac1 + 1:bc0] == 0, 9, out[rr0:rr1 + 1, ac1 + 1:bc0])
    return out


def _place(g, shape, top, left):
    coords = [(top + r, left + c) for r, c in shape]
    if all(0 <= r < g.shape[0] and 0 <= c < g.shape[1] and g[r, c] == 0 for r, c in coords):
        for r, c in coords:
            g[r, c] = 3
        return True
    return False


def generate_input() -> np.ndarray:
    h, w = random.randint(14, 18), random.randint(18, 22)
    g = np.zeros((h, w), dtype=int)
    shapes = [
        [(0, 0), (1, 0), (1, 1), (2, 0), (3, 1)],
        [(0, 1), (1, 0), (1, 1), (2, 1), (3, 0)],
        [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)],
    ]
    col = 1
    for i in range(random.randint(4, 6)):
        shape = random.choice(shapes)
        if i % 2 == 1 and random.random() < 0.7:
            prev = shapes[0]
            shape = [(r, max(c for _, c in prev) - c) for r, c in prev]
        top = random.randint(2, h - 6)
        _place(g, shape, top, col)
        col += random.randint(4, 5)
        if col >= w - 3:
            break
    return g
