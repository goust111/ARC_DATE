import random
from collections import deque
import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and not seen[nx, ny] and mask[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(comp)
    return comps


def _rot(pt, side, k):
    r, c = pt
    for _ in range(k):
        r, c = c, side - 1 - r
    return r, c


def main(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    blue_comps = _components(grid == 1)
    keep = [p for comp in blue_comps if len(comp) > 1 for p in comp]
    for comp in blue_comps:
        if len(comp) == 1:
            r, c = comp[0]
            out[r, c] = 0
    if not keep:
        return out

    h, w = out.shape
    rs = [r for r, _ in keep]
    cs = [c for _, c in keep]
    found = None
    for side in range(max(max(rs) - min(rs) + 1, max(cs) - min(cs) + 1), min(h, w) + 1):
        for top in range(max(0, max(rs) - side + 1), min(min(rs), h - side) + 1):
            for left in range(max(0, max(cs) - side + 1), min(min(cs), w - side) + 1):
                found = (top, left, side)
                break
            if found:
                break
        if found:
            break
    top, left, side = found
    for r, c in keep:
        rr, cc = r - top, c - left
        for k in (1, 2, 3):
            nr, nc = _rot((rr, cc), side, k)
            out[top + nr, left + nc] = 1

    square = out[top:top + side, left:left + side]
    black_comps = _components(square == 0)
    black_comps.sort(key=lambda comp: (-len(comp), min(r for r, _ in comp), min(c for _, c in comp)))
    for idx, comp in enumerate(black_comps):
        if idx % 3 == 0:
            for r, c in comp:
                out[top + r, left + c] = 2
    return out


def generate_input() -> np.ndarray:
    h = w = random.randint(13, 17)
    g = np.zeros((h, w), dtype=int)
    side = random.choice([7, 9])
    top = random.randint(1, h - side - 1)
    left = random.randint(1, w - side - 1)
    seeds = [(1, 1), (1, 2), (2, 1), (3, 1), (3, 2)]
    if random.random() < 0.5:
        seeds.append((2, 2))
    for r, c in seeds:
        g[top + r, left + c] = 1
    for _ in range(random.randint(2, 4)):
        for _attempt in range(100):
            r = random.randrange(h)
            c = random.randrange(w)
            if g[r, c] == 0 and not np.any(g[max(0, r - 1):r + 2, max(0, c - 1):c + 2] == 1):
                g[r, c] = 1
                break
    return g
