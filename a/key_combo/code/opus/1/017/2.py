import random

import numpy as np


DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    out = np.array(grid, copy=True)
    gray = np.argwhere(grid == 5)
    if len(gray) == 0:
        return out
    tr, tc = gray.min(axis=0)
    templ = (grid[tr : tr + 3, tc : tc + 3] == 5)
    markers = sorted(map(tuple, np.argwhere(grid == 1)))
    red_locked = set()
    h, w = grid.shape
    for mr, mc in markers:
        for di, (dr, dc) in enumerate(DIRS):
            r, c = mr + dr, mc + dc
            path = []
            while 0 <= r < h and 0 <= c < w and (out[r, c] == 0 or (out[r, c] == 1 and grid[r, c] == 0)):
                path.append((r, c))
                r += dr
                c += dc
            if not (0 <= r < h and 0 <= c < w) or not path:
                continue
            hit = (r, c)
            rot = np.rot90(templ, -di)
            for rr in range(3):
                for cc in range(3):
                    if not rot[rr, cc]:
                        continue
                    nr, nc = hit[0] + rr - 1, hit[1] + cc - 1
                    if 0 <= nr < h and 0 <= nc < w and out[nr, nc] == 0:
                        out[nr, nc] = 2
                        red_locked.add((nr, nc))
            for pr, pc in path:
                if (pr, pc) not in red_locked and out[pr, pc] in (0, 1):
                    out[pr, pc] = 1
    return out


def _clear(grid, cells):
    h, w = grid.shape
    return all(0 <= r < h and 0 <= c < w and grid[r, c] == 0 for r, c in cells)


def generate_input() -> np.ndarray:
    h = w = 18
    masks = [
        [(0, 0), (0, 2), (1, 0), (2, 1)],
        [(0, 0), (1, 1), (1, 2), (2, 0), (2, 2)],
        [(0, 0), (0, 1), (1, 2), (2, 1)],
    ]
    grid = np.zeros((h, w), dtype=int)
    for r, c in random.choice(masks):
        grid[r, c] = 5
    marker_slots = [(6, 6), (6, 12), (12, 6), (12, 12)]
    random.shuffle(marker_slots)
    for mr, mc in marker_slots[: random.randint(2, 4)]:
        if grid[mr, mc] != 0:
            continue
        grid[mr, mc] = 1
        dirs = random.sample(DIRS, 2)
        for dr, dc in dirs:
            dist = random.randint(3, 5)
            hr, hc = mr + dr * dist, mc + dc * dist
            footprint = [(hr + rr, hc + cc) for rr in (-1, 0, 1) for cc in (-1, 0, 1)]
            line = [(mr + dr * k, mc + dc * k) for k in range(1, dist)]
            if _clear(grid, footprint + line) and 2 <= hr < h - 2 and 2 <= hc < w - 2:
                grid[hr, hc] = 3
    return grid

