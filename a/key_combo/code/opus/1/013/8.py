import random
from collections import Counter, deque
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
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < h and 0 <= ny < w and not seen[nx, ny] and grid[nx, ny] == color:
                            seen[nx, ny] = True
                            q.append((nx, ny))
            comps.append(comp)
    return comps


def _norm(coords):
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _canon(coords):
    pts = list(_norm(coords))
    variants = []
    for flip in (False, True):
        cur = pts
        for _ in range(4):
            variants.append(_norm(cur))
            height = max(r for r, _ in cur) + 1
            cur = [(c, height - 1 - r) for r, c in cur]
        width = max(c for _, c in pts) + 1
        pts = [(r, width - 1 - c) for r, c in pts]
    return min(variants)


def main(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    h, w = out.shape
    comps = _components(grid, 5)
    keys = [_canon(c) for c in comps]
    counts = Counter(keys)
    for comp in comps:
        for r, c in comp:
            out[r, c] = 0
    for comp, key in zip(comps, keys):
        cols = [c for _, c in comp]
        shift = -min(cols) if counts[key] >= 2 else (w - 1 - max(cols))
        for r, c in comp:
            out[r, c + shift] = 4

    sources = [(c, int(grid[0, c])) for c in range(w) if grid[0, c] not in (0, 4, 5)]
    for c0, color in sorted(sources):
        c = c0
        for r in range(1, h):
            delta = 0
            if out[r, 0] == 4:
                delta -= 1
            if out[r, w - 1] == 4:
                delta += 1
            c += delta
            if not (0 <= c < w):
                break
            if out[r, c] == 0:
                out[r, c] = color
    return out


def _place(g, shape, top, left):
    coords = [(top + r, left + c) for r, c in shape]
    if all(0 <= r < g.shape[0] and 0 <= c < g.shape[1] and g[r, c] == 0 for r, c in coords):
        for r, c in coords:
            g[r, c] = 5
        return True
    return False


def generate_input() -> np.ndarray:
    h = random.randint(16, 20)
    w = random.randint(14, 17)
    g = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 2, 3, 6, 7], random.randint(3, 5))
    cols = random.sample(range(2, w - 2), len(colors))
    for c, color in zip(cols, colors):
        g[0, c] = color
    shapes = [
        [(0, 0), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 1), (2, 1)],
        [(0, 1), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (1, 1), (2, 0)],
    ]
    plan = [shapes[0], shapes[0], shapes[1], random.choice(shapes[2:]), random.choice(shapes)]
    for shape in plan[:random.randint(4, 6)]:
        for _ in range(100):
            if _place(g, shape, random.randint(3, h - 5), random.randint(4, w - 6)):
                break
    return g
