import random
from collections import deque

import numpy as np


def _rooms(g: np.ndarray):
    mask = g == 0
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
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and g[nr, nc] != 5 and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        out.append(pts)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each room using the seed color and the seed position rule."""
    g = np.array(grid, dtype=int)
    out = g.copy()
    for pts in _rooms(g):
        cells = set(pts)
        colors = [(r, c) for r, c in pts if g[r, c] not in (0, 5)]
        if not colors:
            continue
        sr, sc = colors[0]
        color = int(g[sr, sc])
        rows = [r for r, _ in pts]
        cols = [c for _, c in pts]
        top, left = min(rows), min(cols)
        for r, c in pts:
            if g[r, c] == 5:
                continue
            if sr == top:
                out[r, c] = color if (r - sr) % 2 == 0 else 0
            elif sc == left:
                out[r, c] = color if (c - sc) % 2 == 0 else 0
            else:
                out[r, c] = color
    return out


def generate_input() -> np.ndarray:
    rows, cols = random.randint(3, 4), random.randint(3, 4)
    rh, rw = random.randint(3, 4), random.randint(3, 4)
    h = rows * rh + rows + 1
    w = cols * rw + cols + 1
    g = np.zeros((h, w), dtype=int)
    g[:: rh + 1, :] = 5
    g[:, :: rw + 1] = 5
    palette = [random.choice([1, 2, 3, 4, 6, 7, 8, 9]) for _ in range(rows * cols)]
    k = 0
    for i in range(rows):
        for j in range(cols):
            r0 = i * (rh + 1) + 1
            c0 = j * (rw + 1) + 1
            mode = random.choice(["top", "left", "inner"])
            if mode == "top":
                rr, cc = r0, random.randint(c0, c0 + rw - 1)
            elif mode == "left":
                rr, cc = random.randint(r0 + 1, r0 + rh - 1), c0
            else:
                rr, cc = random.randint(r0 + 1, r0 + rh - 1), random.randint(c0 + 1, c0 + rw - 1)
            g[rr, cc] = palette[k]
            k += 1
    return g
