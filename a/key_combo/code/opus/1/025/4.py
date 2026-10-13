import random
from collections import deque
import numpy as np


def _regions(grid):
    h, w = grid.shape
    mask = grid == 0
    lab = -np.ones((h, w), dtype=int)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or lab[r, c] >= 0:
                continue
            idx = len(comps)
            q = deque([(r, c)])
            lab[r, c] = idx
            comp = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and lab[nx, ny] < 0:
                        lab[nx, ny] = idx
                        q.append((nx, ny))
            comps.append(comp)
    return lab, comps


def _seed(grid):
    h, w = grid.shape
    for r in range(h - 1):
        for c in range(w - 1):
            vals = set(grid[r:r + 2, c:c + 2].ravel())
            if 1 in vals and 2 in vals and vals <= {1, 2}:
                return r, c
    return None


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    pos = _seed(grid)
    if pos is None:
        return out
    sr, sc = pos
    dirs = []
    corners = {(sr, sc): (-1, -1), (sr, sc + 1): (-1, 1), (sr + 1, sc): (1, -1), (sr + 1, sc + 1): (1, 1)}
    for p, d in corners.items():
        if grid[p] == 2:
            dirs.append(d)
    out[sr:sr + 2, sc:sc + 2] = 1
    work = grid.copy()
    work[sr:sr + 2, sc:sc + 2] = 0
    lab, comps = _regions(work)
    active = set()
    for r in range(sr, sr + 2):
        for c in range(sc, sc + 2):
            if lab[r, c] >= 0:
                active.add(int(lab[r, c]))
    h, w = grid.shape
    for dr, dc in dirs:
        r, c = sr, sc
        while 0 <= r < h and 0 <= c < w:
            if lab[r, c] >= 0:
                active.add(int(lab[r, c]))
            r += dr
            c += dc
    for idx in active:
        for r, c in comps[idx]:
            out[r, c] = 1
    out[sr:sr + 2, sc:sc + 2] = 1
    return out


def generate_input() -> np.ndarray:
    h = w = random.randint(13, 16)
    g = np.zeros((h, w), dtype=int)
    vr = random.randint(5, 8)
    hc = random.randint(5, 8)
    g[:, hc] = 2
    g[vr, :] = 2
    if random.random() < 0.5:
        g[2:h - 2, random.randint(2, 3)] = 2
    sr = random.choice([2, h - 4])
    sc = random.choice([2, w - 4])
    if g[sr:sr + 2, sc:sc + 2].any():
        sr, sc = 2, 2
    g[sr:sr + 2, sc:sc + 2] = 1
    for cr, cc in random.sample([(0, 0), (0, 1), (1, 0), (1, 1)], random.randint(1, 2)):
        g[sr + cr, sc + cc] = 2
    return g
