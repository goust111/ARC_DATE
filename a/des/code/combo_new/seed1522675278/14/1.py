import random
from collections import deque

import numpy as np


BG = 0
OUTLINE = 1


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


def _inside_regions(comp):
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    box = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in comp:
        box[r - r0, c - c0] = True
    h, w = box.shape
    outside = np.zeros_like(box, dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not box[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not box[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not box[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))
    inside = (~box) & (~outside)
    return [(r0, c0, reg) for reg in _components(inside)]


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    legend = [int(v) for v in grid[-1] if v != BG]
    out = grid.copy()
    for comp in _components(grid[:-1] == OUTLINE):
        regions = _inside_regions(comp)
        count = len(regions)
        if count >= len(legend):
            continue
        fill = legend[count]
        for r0, c0, reg in regions:
            for r, c in reg:
                out[r0 + r, c0 + c] = fill
    return out


def _shape_with_regions(regions):
    h = random.randint(5, 7)
    w = random.randint(2 * regions + 3, 2 * regions + 5)
    arr = np.zeros((h, w), dtype=int)
    arr[[0, -1], :] = OUTLINE
    arr[:, [0, -1]] = OUTLINE
    for i in range(1, regions):
        col = 1 + 2 * i
        arr[:, col] = OUTLINE
    return arr


def generate_input() -> np.ndarray:
    grid = np.zeros((22, 24), dtype=int)
    legend = random.sample([2, 3, 4, 5, 6, 7, 8, 9], 4)
    grid[-1, :4] = legend
    occupied = np.zeros_like(grid, dtype=bool)
    occupied[-2:, :] = True
    for regions in random.sample([1, 2, 3], random.randint(2, 3)):
        shape = _shape_with_regions(regions)
        sh, sw = shape.shape
        for _ in range(200):
            r = random.randint(1, 19 - sh)
            c = random.randint(1, 23 - sw)
            rr0, rr1 = max(0, r - 1), min(22, r + sh + 1)
            cc0, cc1 = max(0, c - 1), min(24, c + sw + 1)
            if np.any(occupied[rr0:rr1, cc0:cc1]):
                continue
            grid[r : r + sh, c : c + sw] = np.where(shape != BG, shape, grid[r : r + sh, c : c + sw])
            occupied[rr0:rr1, cc0:cc1] = True
            break
    return grid
