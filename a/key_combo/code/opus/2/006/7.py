import random
from collections import deque

import numpy as np


def _components(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == 0 or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                y, x = q.popleft()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and int(grid[ny, nx]) == color:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            rows = [p[0] for p in cells]
            cols = [p[1] for p in cells]
            comps.append(
                {
                    "color": color,
                    "cells": cells,
                    "area": len(cells),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    comps = sorted(_components(grid), key=lambda c: c["area"], reverse=True)
    seed_obj = comps[0]
    fill_color = comps[1]["color"]
    r1, c1, r2, c2 = seed_obj["bbox"]
    seed = np.zeros((r2 - r1 + 1, c2 - c1 + 1), dtype=int)
    for r, c in seed_obj["cells"]:
        seed[r - r1, c - c1] = seed_obj["color"]
    sh, sw = seed.shape
    out = np.zeros((sh * sh, sw * sw), dtype=int)
    for r in range(sh):
        for c in range(sw):
            block = (slice(r * sh, (r + 1) * sh), slice(c * sw, (c + 1) * sw))
            if seed[r, c] == seed_obj["color"]:
                out[block] = seed
            else:
                out[block] = fill_color
    return out


def _random_polyomino(area):
    cells = {(0, 0)}
    while len(cells) < area:
        y, x = random.choice(tuple(cells))
        dy, dx = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        cells.add((y + dy, x + dx))
        min_y = min(p[0] for p in cells)
        min_x = min(p[1] for p in cells)
        cells = {(r - min_y, c - min_x) for r, c in cells}
        if max(p[0] for p in cells) >= 5 or max(p[1] for p in cells) >= 5:
            return _random_polyomino(area)
    return sorted(cells)


def generate_input() -> np.ndarray:
    palette = [1, 2, 3, 4, 7]
    for _ in range(500):
        grid = np.zeros((14, 14), dtype=int)
        n = random.randint(3, 5)
        areas = sorted(random.sample(range(3, 13), n), reverse=True)
        colors = random.sample(palette, n)
        occupied = set()
        ok = True
        for area, color in zip(areas, colors):
            shape = _random_polyomino(area)
            max_r = max(p[0] for p in shape)
            max_c = max(p[1] for p in shape)
            placed = False
            for _j in range(100):
                top = random.randint(0, 13 - max_r)
                left = random.randint(0, 13 - max_c)
                cells = {(top + r, left + c) for r, c in shape}
                buffered = set(cells)
                for y, x in list(cells):
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        buffered.add((y + dy, x + dx))
                if occupied.isdisjoint(buffered):
                    for y, x in cells:
                        grid[y, x] = color
                    occupied.update(buffered)
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok:
            return grid
    return generate_input()
