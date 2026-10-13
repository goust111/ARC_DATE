import random
from collections import deque

import numpy as np


BG = 0


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            out.append(comp)
    return out


def _parse_legend(row):
    mapping = {}
    c = 0
    w = len(row)
    while c < w:
        if row[c] == BG:
            c += 1
            continue
        combo = []
        while c < w and row[c] != BG:
            combo.append(int(row[c]))
            c += 1
        if c < w - 1:
            c += 1
            if c < w and row[c] != BG:
                mapping[tuple(sorted(set(combo)))] = int(row[c])
                c += 1
        while c < w and row[c] == BG:
            c += 1
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    legend = _parse_legend(grid[-1])
    scene = grid[:-1]
    out = grid.copy()
    h, w = scene.shape
    comps = _components(scene == BG)
    for comp in comps:
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp):
            continue
        boundary = set()
        for r, c in comp:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and scene[nr, nc] != BG:
                    boundary.add(int(scene[nr, nc]))
        color = legend[tuple(sorted(boundary))]
        for r, c in comp:
            out[r, c] = color
    return out


def generate_input() -> np.ndarray:
    h, w = 22, 28
    grid = np.zeros((h, w), dtype=int)
    combos = []
    result_colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 6)
    border_colors = [1, 2, 3, 4, 6, 7, 8, 9]
    rooms = []
    for _ in range(random.randint(2, 3)):
        rooms.append(
            (
                random.randint(2, 12),
                random.randint(2, 16),
                random.randint(4, 6),
                random.randint(5, 7),
                tuple(sorted(random.sample(border_colors, random.randint(2, 3)))),
            )
        )
    used = np.zeros((h - 1, w), dtype=bool)
    chosen = []
    for r0, c0, rh, rw, combo in rooms:
        r1, c1 = r0 + rh - 1, c0 + rw - 1
        if r1 >= h - 1 or c1 >= w:
            continue
        if np.any(used[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2]):
            continue
        used[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2] = True
        colors = list(combo)
        random.shuffle(colors)
        top = colors[0]
        bottom = colors[min(1, len(colors) - 1)]
        side = colors[-1]
        grid[r0, c0 : c1 + 1] = top
        grid[r1, c0 : c1 + 1] = bottom
        grid[r0 : r1 + 1, c0] = side
        grid[r0 : r1 + 1, c1] = side
        chosen.append(combo)
    mapping = {}
    pos = 0
    for i, combo in enumerate(sorted(set(chosen))):
        res = result_colors[i]
        mapping[combo] = res
        for val in combo:
            grid[-1, pos] = val
            pos += 1
        pos += 1
        grid[-1, pos] = res
        pos += 3
    return grid
