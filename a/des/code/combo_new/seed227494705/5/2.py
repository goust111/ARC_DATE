import random
import numpy as np


BG = 7


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _norm(cells):
    r0, c0, _, _ = _bbox(cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def main(grid: np.ndarray) -> np.ndarray:
    """Transfer color from the upper member of each equal-shape pair.

    Input structure: On an orange background, each foreground color forms one
    4-connected sprite. Sprites are arranged in pairs with identical normalized
    masks and different colors.
    Rule: Group sprites by normalized shape. In each two-sprite group, the
    sprite whose bounding box starts higher is the master; erase it, and recolor
    the lower paired sprite in-place with the master's color.
    Key insight: Shape matching establishes the pair, then vertical position
    establishes the one-way color transfer direction.
    """
    out = grid.copy()
    objects = []
    for col in [int(x) for x in np.unique(grid) if x != BG]:
        for comp in _components(grid == col):
            objects.append({"color": col, "cells": comp, "bbox": _bbox(comp), "key": _norm(comp)})
    groups = {}
    for obj in objects:
        groups.setdefault(obj["key"], []).append(obj)
    for pair in groups.values():
        if len(pair) != 2:
            continue
        pair.sort(key=lambda o: (o["bbox"][0], o["bbox"][1]))
        master, slave = pair
        for r, c in master["cells"]:
            out[r, c] = BG
        for r, c in slave["cells"]:
            out[r, c] = master["color"]
    return out


def generate_input() -> np.ndarray:
    h = w = random.choice([22, 24, 26])
    grid = np.full((h, w), BG, dtype=int)
    shapes = [
        [(0, 0), (1, 0), (1, 1)],
        [(0, 1), (1, 1), (2, 0), (2, 1)],
        [(0, 0), (0, 1), (1, 1), (1, 2)],
        [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)],
    ]
    colors = random.sample([0, 1, 2, 3, 4, 5, 6, 8, 9], 8)
    slots = [(2, 2), (10, 3), (2, 9), (12, 10), (4, 15), (14, 16), (7, 6), (16, 6)]
    for idx, shape in enumerate(random.sample(shapes, random.randint(2, 4))):
        upper = slots[2 * idx]
        lower = slots[2 * idx + 1]
        for col, (r0, c0) in [(colors[2 * idx], upper), (colors[2 * idx + 1], lower)]:
            for dr, dc in shape:
                grid[r0 + dr, c0 + dc] = col
    return grid
