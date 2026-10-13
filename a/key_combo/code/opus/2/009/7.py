import random
from collections import deque
import numpy as np


def _components_color(grid, color):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for sr, sc in zip(*np.where(grid == color)):
        if seen[sr, sc]:
            continue
        q = deque([(int(sr), int(sc))])
        seen[sr, sc] = True
        cells = []
        while q:
            r, c = q.popleft()
            cells.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and grid[rr, cc] == color and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        comps.append(cells)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    anchors = np.argwhere(grid == 2)
    ar, ac = map(int, anchors[0])
    offsets = [(int(r - ar), int(c - ac)) for r, c in zip(*np.where(grid == 1))]
    stamps = []
    for comp in _components_color(grid, 3):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
        full = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
        missing = list(full - set(comp))[0]
        dr = -1 if missing[0] == r0 else 1
        dc = -1 if missing[1] == c0 else 1
        cr = sum(rows) // 3
        cc = sum(cols) // 3
        stamps.append((cr, cc, dr, dc))

    out = np.zeros_like(grid)
    for cr, cc, dr, dc in sorted(stamps):
        tr, tc = cr + dr, cc + dc
        for orow, ocol in offsets:
            rr, cc2 = tr + orow, tc + ocol
            if 0 <= rr < out.shape[0] and 0 <= cc2 < out.shape[1]:
                out[rr, cc2] = 1
    return out


def _sprite_offsets():
    return [(-1, 0), (0, -1), (0, 1), (1, 0), (1, 1), (2, 0)]


def generate_input() -> np.ndarray:
    rng = random.Random()
    tri_by_missing = {
        (-1, -1): [(0, 1), (1, 0), (1, 1)],
        (-1, 1): [(0, 0), (1, 0), (1, 1)],
        (1, -1): [(0, 0), (0, 1), (1, 1)],
        (1, 1): [(0, 0), (0, 1), (1, 0)],
    }
    for _ in range(1000):
        h = w = 18
        grid = np.zeros((h, w), dtype=int)
        ar, ac = rng.randint(3, 5), rng.randint(3, 5)
        grid[ar, ac] = 2
        offsets = _sprite_offsets()
        for dr, dc in offsets:
            grid[ar + dr, ac + dc] = 1
        used_stamp = set()
        dirs = rng.sample(list(tri_by_missing), rng.randint(3, 4))
        ok = True
        for miss in dirs:
            placed = False
            for _try in range(300):
                top = rng.randint(2, h - 4)
                left = rng.randint(8, w - 4)
                cells = [(top + r, left + c) for r, c in tri_by_missing[miss]]
                if any(grid[r, c] != 0 for r, c in cells):
                    continue
                if any(abs(r - rr) <= 2 and abs(c - cc) <= 2 for r, c in cells for rr, cc in zip(*np.where(grid == 3))):
                    continue
                cr = sum(r for r, _ in cells) // 3
                cc = sum(c for _, c in cells) // 3
                tr, tc = cr + miss[0], cc + miss[1]
                stamp = {(tr + dr, tc + dc) for dr, dc in offsets}
                if any(not (0 <= r < h and 0 <= c < w) for r, c in stamp) or used_stamp & stamp:
                    continue
                for r, c in cells:
                    grid[r, c] = 3
                used_stamp |= stamp
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return grid
    return generate_input()
