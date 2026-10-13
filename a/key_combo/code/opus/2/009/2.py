import random
from collections import deque
import numpy as np


def _legend_box(grid):
    pts = np.argwhere(grid == 7)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(grid, mask):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for sr, sc in zip(*np.where(mask)):
        if seen[sr, sc]:
            continue
        q = deque([(int(sr), int(sc))])
        seen[sr, sc] = True
        cells = []
        colors = set()
        while q:
            r, c = q.popleft()
            cells.append((r, c))
            colors.add(int(grid[r, c]))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        comps.append((cells, colors))
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    h, w = grid.shape
    out = np.zeros((h * 2, w * 2), dtype=grid.dtype)
    lr0, lc0, lr1, lc1 = _legend_box(grid)
    legend = np.zeros_like(grid, dtype=bool)
    legend[lr0:lr1 + 1, lc0:lc1 + 1] = True
    out[lr0:lr1 + 1, lc0:lc1 + 1] = grid[lr0:lr1 + 1, lc0:lc1 + 1]
    target_colors = {int(x) for x in grid[lr0:lr1, lc0:lc1].reshape(-1) if x not in (0, 7)}

    for r, c in zip(*np.where((grid != 0) & (~legend))):
        color = int(grid[r, c])
        k = 0
        while r + k < out.shape[0] and c + k < out.shape[1]:
            out[r + k, c + k] = color
            k += 1

    out_legend = np.zeros_like(out, dtype=bool)
    out_legend[lr0:lr1 + 1, lc0:lc1 + 1] = True
    mask = (out != 0) & (~out_legend)
    for cells, colors in _components(out, mask):
        if target_colors <= colors:
            for r, c in cells:
                out[r, c] = 0
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(1000):
        grid = np.zeros((8, 8), dtype=int)
        target = rng.sample([1, 3, 4], 2)
        other = rng.choice([2, 6, 8, 9])
        grid[0, 0] = target[0]
        grid[0, 1] = target[1]
        grid[0, 2] = 7
        grid[1, 0:3] = 7

        br = rng.randint(2, 4)
        bc = rng.randint(3, 5)
        target_shape = [(0, 0, target[0]), (0, 1, target[1]), (1, 1, rng.choice(target))]
        for dr, dc, color in target_shape:
            grid[br + dr, bc + dc] = color

        for _try in range(100):
            rr = rng.randint(2, 7)
            cc = rng.randint(0, 7)
            if grid[rr, cc] == 0 and abs(rr - br) + abs(cc - bc) > 3:
                grid[rr, cc] = other
                break
        return grid
    return generate_input()
