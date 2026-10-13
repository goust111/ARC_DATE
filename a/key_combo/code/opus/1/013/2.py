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


def _mask(coords):
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    r0, c0 = min(rs), min(cs)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _variants(mask):
    pts = list(mask)
    arrs = []
    for flip in (False, True):
        cur = pts
        for _ in range(4):
            rs = [r for r, _ in cur]
            cs = [c for _, c in cur]
            r0, c0 = min(rs), min(cs)
            arrs.append(tuple(sorted((r - r0, c - c0) for r, c in cur)))
            height = max(rs) - min(rs) + 1
            cur = [(c, height - 1 - r) for r, c in [(r - r0, c - c0) for r, c in cur]]
        if not flip:
            width = max(c for _, c in pts) + 1
            pts = [(r, width - 1 - c) for r, c in pts]
    return set(arrs)


def _shear(coords):
    bottom = max(r for r, _ in coords)
    return [(r, c - (bottom - r)) for r, c in coords if c - (bottom - r) >= 0]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    templates = _components(g, 8)
    if not templates:
        return g
    target = _variants(_mask(max(templates, key=len)))
    for comp in _components(g, 3):
        for r, c in comp:
            g[r, c] = 0
        moved = _shear(comp)
        if not moved:
            continue
        color = 8 if _mask(moved) in target else 0
        if color:
            for r, c in moved:
                if 0 <= r < g.shape[0] and 0 <= c < g.shape[1]:
                    g[r, c] = color
    return g


def _place(g, coords, color):
    h, w = g.shape
    if all(0 <= r < h and 0 <= c < w and g[r, c] == 0 for r, c in coords):
        for r, c in coords:
            g[r, c] = color
        return True
    return False


def generate_input() -> np.ndarray:
    h = w = random.randint(15, 18)
    g = np.zeros((h, w), dtype=int)
    templates = [
        [(0, 1), (1, 0), (1, 1), (2, 1), (2, 2)],
        [(0, 0), (1, 0), (1, 1), (1, 2), (2, 2)],
        [(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)],
    ]
    t = random.choice(templates)
    for r, c in t:
        g[1 + r, 1 + c] = 8
    vars_ = [list(v) for v in _variants(_mask(t))]

    made = 0
    attempts = 0
    while made < random.randint(2, 3) and attempts < 300:
        attempts += 1
        s = random.choice(vars_)
        tr = random.randint(5, h - 4)
        tc = random.randint(5, w - 4)
        moved = [(tr + r, tc + c) for r, c in s]
        bottom = max(r for r, _ in moved)
        original = [(r, c + (bottom - r)) for r, c in moved]
        if _place(g, original, 3):
            made += 1

    bad_shapes = [
        [(0, 0), (0, 1), (1, 1), (2, 1), (3, 1)],
        [(0, 0), (1, 0), (2, 0), (2, 1), (3, 1)],
        [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (3, 1)],
    ]
    for _ in range(random.randint(3, 4)):
        for _attempt in range(100):
            shape = random.choice(bad_shapes)
            r0 = random.randint(4, h - 5)
            c0 = random.randint(2, w - 5)
            coords = [(r0 + r, c0 + c) for r, c in shape]
            if _place(g, coords, 3):
                break
    return g
