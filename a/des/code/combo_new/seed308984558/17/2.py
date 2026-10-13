import random
from collections import deque

import numpy as np


def _deg(mask: np.ndarray, r: int, c: int) -> int:
    return sum(0 <= r + dr < mask.shape[0] and 0 <= c + dc < mask.shape[1] and mask[r + dr, c + dc] for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))


def main(grid: np.ndarray) -> np.ndarray:
    """Split the grey network at junctions and recolor each segment by length rank."""
    g = np.array(grid, dtype=int)
    path = g == 5
    palette = [int(c) for c in g[-1] if int(c) != 0]
    junction = np.zeros_like(path, dtype=bool)
    for r, c in zip(*np.where(path)):
        junction[r, c] = _deg(path, int(r), int(c)) >= 3
    segments = []
    used = set()
    for r, c in zip(*np.where(path)):
        r, c = int(r), int(c)
        if junction[r, c]:
            continue
        if (r, c) in used:
            continue
        pts = []
        q = deque([(r, c)])
        used.add((r, c))
        while q:
            rr, cc = q.popleft()
            pts.append((rr, cc))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < g.shape[0] - 1 and 0 <= nc < g.shape[1] and path[nr, nc] and not junction[nr, nc] and (nr, nc) not in used:
                    used.add((nr, nc))
                    q.append((nr, nc))
        segments.append(pts)
    segments.sort(key=len)
    out = g.copy()
    for i, pts in enumerate(segments):
        color = palette[min(i, len(palette) - 1)]
        for r, c in pts:
            out[r, c] = color
    return out


def generate_input() -> np.ndarray:
    h, w = random.randint(16, 22), random.randint(16, 22)
    g = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 4)
    g[-1, 2 : 2 + len(colors)] = colors
    cx, cy = random.randint(5, h - 7), random.randint(5, w - 7)
    g[cx - 3 : cx + 4, cy] = 5
    g[cx, cy - 4 : cy + 5] = 5
    g[cx - 2, cy + 1 : cy + 4] = 5
    g[cx + 1 : cx + 5, cy - 2] = 5
    return g
