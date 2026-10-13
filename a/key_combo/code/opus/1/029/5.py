import random

import numpy as np


def _legend(grid):
    row = grid[0]
    end = 0
    while end < len(row) and row[end] != 0 and row[end] != 5:
        end += 1
    items = []
    i = 0
    while i < end:
        color = int(row[i])
        j = i
        while j < end and row[j] == color:
            j += 1
        items.append((color, j - i))
        i = j
    return items


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    h, w = g.shape
    items = _legend(g)
    markers = sorted((r, c) for r, c in zip(*np.where(g == 5)))
    for (color, radius), (mr, mc) in zip(items, markers):
        for r in range(mr - radius + 1, mr + radius):
            for c in range(mc - radius + 1, mc + radius):
                if not (0 <= r < h and 0 <= c < w):
                    continue
                k = max(abs(r - mr), abs(c - mc))
                if k < radius:
                    g[r, c] = color if k % 2 == 0 else 0
    return g


def _far(points, p, dist):
    return all(abs(r - p[0]) + abs(c - p[1]) >= dist for r, c in points)


def generate_input() -> np.ndarray:
    h = w = random.randint(18, 22)
    g = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 3)
    counts = random.sample([1, 2, 3], 3)
    x = 0
    for color, count in zip(colors, counts):
        g[0, x : x + count] = color
        x += count
    pts = []
    tries = 0
    while len(pts) < 3 and tries < 500:
        tries += 1
        p = (random.randint(5, h - 6), random.randint(5, w - 6))
        if _far(pts, p, 10):
            pts.append(p)
    if len(pts) < 3:
        pts = [(5, 5), (5, w - 6), (h - 6, w // 2)]
    for r, c in pts:
        g[r, c] = 5
    return g
