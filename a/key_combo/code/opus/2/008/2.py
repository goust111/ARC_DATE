import random
from collections import deque

import numpy as np


def _component_count(g, color):
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if g[r, c] != color or seen[r, c]:
                continue
            count += 1
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                cr, cc = q.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and g[nr, nc] == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return count


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    h, w = g.shape
    sep = next(r for r in range(h) if np.all(g[r] == 5))
    upper = g[:sep]
    colors = [int(c) for c in np.unique(upper) if c not in (0, 5)]
    counts = {c: _component_count(upper, c) for c in colors}
    out = np.zeros_like(g)
    for r in range(sep + 1, h):
        for c in range(w):
            color = int(g[r, c])
            if color in counts:
                out[r, c:min(w, c + counts[color])] = 5
    return out


def _can_place(g, cells, color):
    h, w = g.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w) or g[r, c] != 0:
            return False
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and g[nr, nc] == color:
                return False
    return True


def generate_input() -> np.ndarray:
    h, w = random.choice([(16, 12), (18, 12), (18, 14)])
    sep = random.randint(8, 10)
    g = np.zeros((h, w), dtype=int)
    g[sep] = 5
    colors = random.sample([1, 2, 3], random.randint(2, 3))
    counts = random.sample(range(1, 6), len(colors))
    shapes = [
        [(0, 0), (0, 1), (1, 0)],
        [(0, 0), (1, 0), (1, 1), (2, 0)],
        [(0, 1), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (0, 1), (1, 1), (1, 2)],
    ]
    for color, n in zip(colors, counts):
        placed = 0
        tries = 0
        while placed < n and tries < 500:
            tries += 1
            shape = random.choice(shapes)
            r = random.randint(0, sep - 4)
            c = random.randint(0, w - 4)
            cells = [(r + dr, c + dc) for dr, dc in shape]
            if _can_place(g, cells, color) and all(g[rr, cc] == 0 for rr, cc in cells):
                for rr, cc in cells:
                    g[rr, cc] = color
                placed += 1
    rows = random.sample(range(sep + 2, h), len(colors))
    cols = random.sample(range(0, max(1, w - 5)), len(colors))
    for color, r, c in zip(colors, rows, cols):
        g[r, c] = color
    return g
