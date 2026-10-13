import random
from collections import deque

import numpy as np


def _components(mask: np.ndarray):
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for r, c in zip(*np.where(mask)):
        if seen[r, c]:
            continue
        q = deque([(int(r), int(c))])
        seen[r, c] = True
        pts = []
        while q:
            rr, cc = q.popleft()
            pts.append((rr, cc))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        out.append(pts)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Translate each scattered fragment back toward the red blast center by its area."""
    g = np.array(grid, dtype=int)
    cr, cc = map(int, np.argwhere(g == 2)[0])
    color = [int(v) for v in set(map(int, g.ravel())) - {0, 2}][0]
    out = np.zeros_like(g)
    for pts in _components(g == color):
        rs = [p[0] for p in pts]
        cs = [p[1] for p in pts]
        mr, mc = np.mean(rs), np.mean(cs)
        dr = -1 if mr < cr - 0.5 else 1 if mr > cr + 0.5 else 0
        dc = -1 if mc < cc - 0.5 else 1 if mc > cc + 0.5 else 0
        for r, c in pts:
            out[r - dr * len(pts), c - dc * len(pts)] = color
    out[cr, cc] = color
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((random.randint(18, 24), random.randint(18, 24)), dtype=int)
    cr, cc = g.shape[0] // 2, g.shape[1] // 2
    g[cr, cc] = 2
    color = random.choice([1, 3, 4, 5, 6, 7, 8, 9])
    pieces = [[(0, 0), (1, 0)], [(0, 0), (0, 1), (1, 0)], [(0, 0), (1, 0), (1, 1)]]
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for shape, (dr, dc) in zip(random.choices(pieces, k=random.randint(3, 5)), random.sample(dirs, random.randint(3, 5))):
        dist = len(shape)
        base_r, base_c = cr + dr * dist, cc + dc * dist
        for rr, cc2 in shape:
            if 0 <= base_r + rr < g.shape[0] and 0 <= base_c + cc2 < g.shape[1]:
                g[base_r + rr, base_c + cc2] = color
    return g
