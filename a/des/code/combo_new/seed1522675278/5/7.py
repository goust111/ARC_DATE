import random
from collections import deque

import numpy as np


BG = 0
SEAM = 5


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cur = []
            while q:
                x, y = q.popleft()
                cur.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            out.append(cur)
    return out


def _profiles(comp):
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, c0 = min(rows), min(cols)
    h = max(rows) - r0 + 1
    by_row = [[] for _ in range(h)]
    for r, c in comp:
        by_row[r - r0].append(c)
    mins = [min(x) for x in by_row]
    maxs = [max(x) for x in by_row]
    left_base = min(sorted(mins)[len(mins) // 2], min(mins))
    right_base = max(sorted(maxs)[len(maxs) // 2], max(maxs))
    left = tuple(m - left_base for m in mins)
    right = tuple(m - right_base for m in maxs)
    width = right_base - left_base + 1
    return left, right, width, h, r0, c0


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    pieces = []
    for color in [c for c in np.unique(grid) if c != BG]:
        for comp in _components(grid == color):
            pieces.append((color, comp, _profiles(comp)))
    if not pieces:
        return grid
    start = min(pieces, key=lambda x: sum(abs(v) for v in x[2][0]))
    ordered = [start]
    remaining = [p for p in pieces if p != start]
    while remaining:
        prev = ordered[-1]
        pr = prev[2][1]
        nxt = min(remaining, key=lambda x: sum(abs(a + b) for a, b in zip(pr, x[2][0])))
        ordered.append(nxt)
        remaining.remove(nxt)
    height = max(x[2][3] for x in ordered)
    width = sum(x[2][2] for x in ordered) + len(ordered) - 1
    out = np.zeros((height, width), dtype=int)
    x = 0
    for idx, (color, comp, prof) in enumerate(ordered):
        left, right, basew, h, r0, c0 = prof
        mask = np.zeros((h, max(c for _, c in comp) - c0 + 1), dtype=int)
        for r, c in comp:
            mask[r - r0, c - c0] = color
        out[: mask.shape[0], x : x + mask.shape[1]] = np.maximum(out[: mask.shape[0], x : x + mask.shape[1]], mask)
        x += basew
        if idx < len(ordered) - 1:
            out[:h, x] = SEAM
            x += 1
    return out


def generate_input() -> np.ndarray:
    h = random.randint(12, 18)
    w = random.randint(18, 26)
    grid = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 2, 3], random.randint(2, 3))
    x = 1
    for i, color in enumerate(colors):
        ph = random.randint(3, 4)
        pw = random.randint(3, 4)
        top = random.randint(1, h - ph - 1)
        left = x
        grid[top : top + ph, left : left + pw] = color
        mid = top + ph // 2
        if i < len(colors) - 1:
            grid[mid, left + pw] = color
        if i > 0:
            grid[mid, left] = BG
        x += pw + 3
    return grid
