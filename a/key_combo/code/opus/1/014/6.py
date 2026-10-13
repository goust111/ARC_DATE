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


def _find_square(g):
    for r in range(g.shape[0] - 1):
        for c in range(g.shape[1] - 1):
            if np.all(g[r:r + 2, c:c + 2] == 1):
                return r, c
    return None


def main(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    h, w = out.shape
    sq = _find_square(grid)
    if sq is None:
        return out
    r, c = sq
    corners = {
        (-1, -1): (r - 1, c - 1),
        (-1, 1): (r - 1, c + 2),
        (1, -1): (r + 2, c - 1),
        (1, 1): (r + 2, c + 2),
    }
    for (dr, dc), (sr, sc) in corners.items():
        if 0 <= sr < h and 0 <= sc < w and grid[sr, sc] == 1:
            rr, cc = sr, sc
            while 0 <= rr < h and 0 <= cc < w:
                if out[rr, cc] == 0:
                    out[rr, cc] = 1
                rr += dr
                cc += dc
    comps = _components(out == 0)
    comps.sort(key=lambda comp: (-len(comp), min(r for r, _ in comp), min(c for _, c in comp)))
    for idx, comp in enumerate(comps):
        if idx % 3 == 0:
            for rr, cc in comp:
                out[rr, cc] = 4
    return out


def generate_input() -> np.ndarray:
    h = w = random.randint(13, 17)
    g = np.zeros((h, w), dtype=int)
    r = random.randint(5, h - 7)
    c = random.randint(5, w - 7)
    g[r:r + 2, c:c + 2] = 1
    dirs = random.sample([(-1, -1), (-1, 1), (1, -1), (1, 1)], random.randint(2, 4))
    for dr, dc in dirs:
        sr = r - 1 if dr == -1 else r + 2
        sc = c - 1 if dc == -1 else c + 2
        g[sr, sc] = 1
    return g
