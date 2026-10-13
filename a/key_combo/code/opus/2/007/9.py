import random
from collections import deque
import numpy as np


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
    h, w = grid.shape
    heights = []
    for c in range(w):
        ht = 0
        for r in range(h - 1, -1, -1):
            if grid[r, c] == 3:
                ht += 1
            else:
                break
        heights.append(ht)

    ref_pts = np.argwhere(grid == 7)
    rr0, cc0 = ref_pts.min(axis=0)
    rr1, cc1 = ref_pts.max(axis=0)
    ref = grid[rr0:rr1 + 1, cc0:cc1 + 1] == 7

    out = grid.copy()
    out[grid == 7] = 0
    out[grid == 1] = 0
    for comp in _components(grid == 1):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        center_r = (min(rows) + max(rows)) // 2
        center_c = (min(cols) + max(cols)) // 2
        rot = np.rot90(ref, -heights[center_c] % 4)
        top = center_r - rot.shape[0] // 2
        left = center_c - rot.shape[1] // 2
        for dr, dc in np.argwhere(rot):
            out[top + int(dr), left + int(dc)] = 1
    return out


def generate_input() -> np.ndarray:
    shape = np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool)
    for _ in range(2000):
        h = w = 16
        grid = np.zeros((h, w), dtype=int)
        heights = [random.randint(1, 8) for _ in range(w)]
        centers = random.sample(range(3, 14), random.randint(3, 4))
        rots = [random.randint(0, 3) for _ in centers]
        for c, rot in zip(centers, rots):
            choices = [v for v in range(1, 10) if v % 4 == rot]
            heights[c] = random.choice(choices)
        for c, ht in enumerate(heights):
            grid[h - ht:, c] = 3
        grid[0:3, 0:3][shape] = 7
        occupied = set(map(tuple, np.argwhere(grid != 0)))
        ok = True
        for c, rot in zip(centers, rots):
            mask = np.rot90(shape, -rot)
            placed = False
            for _j in range(200):
                lo, hi = 4, h - heights[c] - 2
                if lo > hi:
                    break
                cr = random.randint(lo, hi)
                top = cr - 1
                left = c - 1
                pts = [(top + int(r), left + int(col)) for r, col in np.argwhere(mask)]
                box = [(top + r, left + col) for r in range(3) for col in range(3)]
                if all(0 <= r < h and 0 <= col < w and (r, col) not in occupied for r, col in box for _ in [0]):
                    for r, col in pts:
                        grid[r, col] = 1
                    occupied.update(box)
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok:
            return grid
    raise RuntimeError("failed to generate input")
