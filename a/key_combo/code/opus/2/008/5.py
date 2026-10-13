import random
from collections import deque

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    h, w = g.shape
    sep = next(r for r in range(h) if np.all(g[r] == 5))
    upper = g[:sep]
    uh, uw = upper.shape
    seen = np.zeros((uh, uw), dtype=bool)
    cavities = 0
    for r in range(uh):
        for c in range(uw):
            if upper[r, c] != 0 or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            touches_open = False
            while q:
                cr, cc = q.popleft()
                touches_open |= cr == 0 or cc == 0 or cc == uw - 1
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < uh and 0 <= nc < uw and upper[nr, nc] == 0 and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_open:
                cavities += 1
    coords = np.argwhere((g[sep + 1:] != 0) & (g[sep + 1:] != 5))
    rows = coords[:, 0] + sep + 1
    cols = coords[:, 1]
    template = g[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    th, tw = template.shape
    out = np.zeros((th, cavities * tw + cavities - 1), dtype=int)
    for i in range(cavities):
        out[:, i * (tw + 1):i * (tw + 1) + tw] = template
    return out


def generate_input() -> np.ndarray:
    h, w = random.choice([(15, 20), (16, 20), (15, 18)])
    sep = random.randint(9, 11)
    g = np.zeros((h, w), dtype=int)
    g[sep] = 5
    n = random.randint(2, 5)
    slots = [(1, 1), (1, 6), (1, 11), (5, 3), (5, 10)]
    random.shuffle(slots)
    for r0, c0 in slots[:n]:
        bh = random.choice([3, 4])
        bw = random.choice([3, 4])
        if r0 + bh < sep and c0 + bw < w - 1:
            g[r0, c0:c0 + bw] = 2
            g[r0 + bh - 1, c0:c0 + bw] = 2
            g[r0:r0 + bh, c0] = 2
            g[r0:r0 + bh, c0 + bw - 1] = 2
    sprites = [
        np.array([[0, 3, 0], [3, 3, 3], [0, 3, 3]]),
        np.array([[3, 3, 0], [0, 3, 0], [3, 3, 3]]),
        np.array([[3, 0, 0, 0], [3, 3, 3, 0], [0, 3, 0, 0], [0, 3, 3, 3]]),
    ]
    lower_h = h - sep - 1
    sprite = random.choice([s for s in sprites if s.shape[0] <= lower_h])
    sr = sep + 1 + random.randint(0, max(0, h - sep - 1 - sprite.shape[0]))
    sc = random.randint(1, w - sprite.shape[1] - 1)
    mask = sprite != 0
    patch = g[sr:sr + sprite.shape[0], sc:sc + sprite.shape[1]]
    patch[mask] = 3
    return g
