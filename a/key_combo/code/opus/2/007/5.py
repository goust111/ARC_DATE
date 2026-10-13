import random
from collections import deque
import numpy as np


def _same_under_flip(block, ref):
    return np.array_equal(block, ref) or np.array_equal(block, np.fliplr(ref)) or np.array_equal(block, np.flipud(ref))


def _extract_tile(grid):
    h, w = grid.shape
    candidates = []
    for p in range(1, w):
        if w % p == 0 and w // p >= 2:
            ref = grid[:, :p]
            if all(_same_under_flip(grid[:, i:i + p], ref) for i in range(0, w, p)):
                candidates.append((h * p, grid[:, :p]))
    for p in range(1, h):
        if h % p == 0 and h // p >= 2:
            ref = grid[:p, :]
            if all(_same_under_flip(grid[i:i + p, :], ref) for i in range(0, h, p)):
                candidates.append((p * w, grid[:p, :]))
    if not candidates:
        return grid.copy()
    return min(candidates, key=lambda x: x[0])[1].copy()


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    comps = []
    for r, c in np.argwhere(mask):
        if seen[r, c]:
            continue
        q = deque([(int(r), int(c))])
        seen[r, c] = True
        comp = []
        while q:
            x, y = q.popleft()
            comp.append((x, y))
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= nx < mask.shape[0] and 0 <= ny < mask.shape[1] and mask[nx, ny] and not seen[nx, ny]:
                    seen[nx, ny] = True
                    q.append((nx, ny))
        comps.append(comp)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    tile = _extract_tile(grid)
    out = tile.copy()
    for comp in _components(tile == 5):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if r in (r0, r1) and c in (c0, c1):
                    out[r, c] = 1
                elif r in (r0, r1) or c in (c0, c1):
                    out[r, c] = 4
                else:
                    out[r, c] = 2
    return out


def generate_input() -> np.ndarray:
    for _ in range(1000):
        th, tw = random.choice([(8, 9), (9, 8), (10, 7), (7, 10)])
        tile = np.zeros((th, tw), dtype=int)
        rects = []
        for _i in range(random.randint(1, 3)):
            for _j in range(200):
                rh, rw = random.choice([(2, 3), (3, 2), (3, 4), (4, 3)])
                if rh >= th or rw >= tw:
                    continue
                r0 = random.randint(0, th - rh)
                c0 = random.randint(0, tw - rw)
                rect = (r0, r0 + rh - 1, c0, c0 + rw - 1)
                if all(rect[1] + 1 < a or a1 + 1 < rect[0] or rect[3] + 1 < b or b1 + 1 < rect[2] for a, a1, b, b1 in rects):
                    rects.append(rect)
                    tile[rect[0]:rect[1] + 1, rect[2]:rect[3] + 1] = 5
                    break
        if not rects:
            continue
        axis = random.choice(["h", "v"])
        reps = random.randint(2, 3)
        blocks = []
        for i in range(reps):
            block = tile.copy()
            if i and random.random() < 0.5:
                block = np.fliplr(block) if axis == "h" else np.flipud(block)
            blocks.append(block)
        grid = np.concatenate(blocks, axis=1 if axis == "h" else 0)
        if max(grid.shape) <= 30 and np.array_equal(_extract_tile(grid), tile):
            return grid
    raise RuntimeError("failed to generate input")
