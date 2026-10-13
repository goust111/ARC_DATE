import random
from collections import deque
import numpy as np


def _components8(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] == 0:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                x, y = q.popleft()
                cells.append((x, y))
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < h and 0 <= ny < w and not seen[nx, ny] and grid[nx, ny] != 0:
                            seen[nx, ny] = True
                            q.append((nx, ny))
            comps.append(cells)
    return comps


def _sprites(grid):
    ans = []
    for cells in _components8(grid):
        colors = {}
        for r, c in cells:
            colors.setdefault(int(grid[r, c]), []).append((r, c))
        if len(colors) == 2 and 1 in [len(v) for v in colors.values()] and len(cells) > 3:
            anchor_color = next(k for k, v in colors.items() if len(v) == 1)
            body_color = next(k for k in colors if k != anchor_color)
            ar, ac = colors[anchor_color][0]
            offsets = [(r - ar, c - ac) for r, c in colors[body_color]] + [(0, 0)]
            ans.append({"anchor": anchor_color, "body": body_color, "size": len(colors[body_color]), "offsets": offsets, "cells": cells})
    return ans


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    sprites = _sprites(g)
    sprite_cells = {p for s in sprites for p in s["cells"]}
    out = np.zeros_like(g)
    for sp in sorted(sprites, key=lambda s: s["size"]):
        markers = []
        for r in range(g.shape[0]):
            for c in range(g.shape[1]):
                if (r, c) in sprite_cells or g[r, c] != sp["anchor"]:
                    continue
                if sum(0 <= r + dr < g.shape[0] and 0 <= c + dc < g.shape[1] and g[r + dr, c + dc] != 0 for dr in (-1, 0, 1) for dc in (-1, 0, 1)) == 1:
                    markers.append((r, c))
        for mr, mc in markers:
            for dr, dc in sp["offsets"]:
                rr, cc = mr + dr, mc + dc
                if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                    out[rr, cc] = sp["body"]
    return out


def _put_sprite(g, top, left, shape, body, anchor):
    ar, ac = shape[0]
    for r, c in shape:
        g[top + r, left + c] = body
    g[top + ar, left + ac] = anchor
    return top + ar, left + ac


def generate_input() -> np.ndarray:
    g = np.zeros((18, 18), dtype=int)
    shapes = [
        [(1, 1), (0, 1), (1, 2), (2, 1)],
        [(1, 1), (0, 0), (1, 0), (1, 2), (2, 2)],
        [(1, 1), (0, 1), (1, 0), (2, 0), (2, 1), (2, 2)],
        [(1, 1), (0, 2), (1, 0), (1, 2), (2, 1), (2, 2), (3, 2)],
    ]
    n = random.randint(3, 4)
    body_cols = random.sample([1, 2, 3, 4, 6, 7, 8, 9], n)
    anchor_cols = random.sample([1, 2, 3, 4, 6, 7, 8, 9], n)
    while set(body_cols) & set(anchor_cols):
        anchor_cols = random.sample([1, 2, 3, 4, 6, 7, 8, 9], n)
    for i in range(n):
        _put_sprite(g, 1 + i * 4, 1, shapes[i], body_cols[i], anchor_cols[i])
    positions = [
        (4, 10), (4, 13), (4, 16), (7, 10), (7, 13), (7, 16),
        (10, 10), (10, 13), (10, 16), (13, 10), (13, 13), (13, 16),
        (16, 10), (16, 13), (16, 16),
    ]
    random.shuffle(positions)
    k = 0
    for i in range(n):
        for _ in range(random.randint(1, 3)):
            r, c = positions[k]; k += 1
            g[r, c] = anchor_cols[i]
    for r, c in positions[k:k + random.randint(0, 2)]:
        g[r, c] = random.choice([x for x in [1, 2, 3, 4, 6, 7, 8, 9] if x not in anchor_cols])
    return g
