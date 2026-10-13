import random
from collections import deque
import numpy as np


def _components(mask):
    h, w = mask.shape
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


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    for color in [int(v) for v in np.unique(grid) if v not in (0, 5)]:
        pts = np.argwhere(grid == color)
        if len(pts) == 2:
            r0, c0 = pts[0]
            r1, c1 = pts[1]
            out[min(r0, r1):max(r0, r1) + 1, min(c0, c1):max(c0, c1) + 1] = color
    bg = (out == 0) | (out == 5)
    lab, comps = _components(bg)
    adj = [set() for _ in comps]
    h, w = out.shape
    for r in range(h):
        for c in range(w):
            if out[r, c] in (0, 5):
                continue
            for dr, dc in ((1, 0), (0, 1)):
                a = (r - dr, c - dc)
                b = (r + dr, c + dc)
                if 0 <= a[0] < h and 0 <= a[1] < w and 0 <= b[0] < h and 0 <= b[1] < w:
                    la, lb = lab[a], lab[b]
                    if la >= 0 and lb >= 0 and la != lb:
                        adj[la].add(lb)
                        adj[lb].add(la)
    anchors = np.argwhere(grid == 5)
    if len(anchors) == 0:
        return out
    start = lab[tuple(anchors[0])]
    colors = {}
    q = deque([(int(start), 3)])
    colors[int(start)] = 3
    while q:
        node, col = q.popleft()
        nxt = 7 if col == 3 else 3
        for nb in adj[node]:
            if nb not in colors:
                colors[nb] = nxt
                q.append((nb, nxt))
    for i, comp in enumerate(comps):
        col = colors.get(i)
        if col is not None:
            for r, c in comp:
                out[r, c] = col
    return out


def generate_input() -> np.ndarray:
    h = w = random.randint(12, 15)
    g = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 2, 4, 6, 8, 9], 3)
    c1 = random.randint(3, 4)
    c2 = random.randint(8, w - 3)
    for color, c in zip(colors[:2], (c1, c2)):
        g[0, c] = color
        g[h - 1, c + 1] = color
    hr = random.randint(4, h - 5)
    color = colors[2]
    g[hr, 0] = color
    g[hr + 1, w - 1] = color
    g[random.randint(1, hr - 1), random.randint(1, c1 - 1)] = 5
    return g
