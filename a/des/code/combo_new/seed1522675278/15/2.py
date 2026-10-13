import random
from collections import defaultdict, deque

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


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    comps = _components(grid != BG)
    groups = defaultdict(list)
    for comp in comps:
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, c0 = min(rows), min(cols)
        shape = tuple(sorted((r - r0, c - c0) for r, c in comp))
        groups[shape].append(comp)
    out = grid.copy()
    for pair in groups.values():
        pair = sorted(pair, key=lambda comp: (min(r for r, _ in comp), min(c for _, c in comp)))
        for i in range(0, len(pair), 2):
            a, b = pair[i], pair[i + 1]
            ca = int(grid[a[0]])
            cb = int(grid[b[0]])
            for r, c in a:
                out[r, c] = cb
            for r, c in b:
                out[r, c] = ca
    return out


def _random_shape():
    h = random.randint(2, 4)
    w = random.randint(2, 4)
    target = random.randint(3, h * w - 1)
    cells = {(0, 0)}
    while len(cells) < target:
        r, c = random.choice(list(cells))
        nr, nc = r + random.choice([-1, 0, 1]), c + random.choice([-1, 0, 1])
        if 0 <= nr < h and 0 <= nc < w and abs(nr - r) + abs(nc - c) == 1:
            cells.add((nr, nc))
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, c0 = min(rows), min(cols)
    shape = tuple(sorted((r - r0, c - c0) for r, c in cells))
    return shape


def _make_input_once():
    grid = np.zeros((22, 22), dtype=int)
    occupied = np.zeros_like(grid, dtype=bool)
    colors = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], random.randint(4, 6))
    pairs = len(colors) // 2
    used_shapes = set()
    for i in range(pairs):
        for _ in range(100):
            shape = _random_shape()
            if shape not in used_shapes:
                used_shapes.add(shape)
                break
        else:
            return None
        h = max(r for r, _ in shape) + 1
        w = max(c for _, c in shape) + 1
        arr = np.zeros((h, w), dtype=int)
        for r, c in shape:
            arr[r, c] = 1
        for color in colors[2 * i : 2 * i + 2]:
            placed = False
            for _ in range(300):
                r = random.randint(1, 21 - h)
                c = random.randint(1, 21 - w)
                rr0, rr1 = max(0, r - 1), min(22, r + h + 1)
                cc0, cc1 = max(0, c - 1), min(22, c + w + 1)
                if np.any(occupied[rr0:rr1, cc0:cc1]):
                    continue
                grid[r : r + h, c : c + w] = np.where(arr != 0, color, grid[r : r + h, c : c + w])
                occupied[rr0:rr1, cc0:cc1] = True
                placed = True
                break
            if not placed:
                return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _make_input_once()
        if grid is None:
            continue
        try:
            out = main(grid)
        except Exception:
            continue
        if isinstance(out, np.ndarray) and out.ndim == 2 and max(out.shape) <= 30:
            return grid
    raise RuntimeError("failed to generate valid input")
