import random
from collections import deque
import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for sr, sc in np.argwhere(mask):
        if seen[sr, sc]:
            continue
        q = deque([(int(sr), int(sc))])
        seen[sr, sc] = True
        comp = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _slots(grid):
    grey = np.argwhere(grid == 5)
    r0, c0 = grey.min(axis=0)
    r1, c1 = grey.max(axis=0)
    sub = grid[r0:r1 + 1, c0:c1 + 1]
    line_rows = [r0 + i for i in range(sub.shape[0]) if np.all(sub[i, :] == 5)]
    line_cols = [c0 + j for j in range(sub.shape[1]) if np.all(sub[:, j] == 5)]
    cells = []
    for a, b in zip(line_rows, line_rows[1:]):
        for x, y in zip(line_cols, line_cols[1:]):
            if b - a > 1 and y - x > 1:
                cells.append((a + 1, b - 1, x + 1, y - 1))
    return (r0, r1, c0, c1), cells


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    (gr0, gr1, gc0, gc1), slots = _slots(grid)
    outside = np.ones(grid.shape, dtype=bool)
    outside[gr0:gr1 + 1, gc0:gc1 + 1] = False
    comps = []
    for comp in _components((grid != 0) & (grid != 5) & outside):
        color = int(grid[comp[0]])
        comps.append((len(comp), color))
    comps.sort()
    for i, slot in enumerate(slots):
        r0, r1, c0, c1 = slot
        if i < len(comps):
            out[r0:r1 + 1, c0:c1 + 1] = comps[i][1]
        else:
            out[r0:r1 + 1, c0:c1 + 1] = 0
    return out


def _draw_lattice(grid, top, left, rows=2, cols=3):
    for rr in range(rows + 1):
        grid[top + 2 * rr, left:left + 2 * cols + 1] = 5
    for cc in range(cols + 1):
        grid[top:top + 2 * rows + 1, left + 2 * cc] = 5


def _put_shape(grid, top, left, color, shape):
    for dr, dc in shape:
        grid[top + dr, left + dc] = color


def generate_input() -> np.ndarray:
    shapes = [
        [(0, 0), (0, 1)],
        [(0, 0), (1, 0), (1, 1)],
        [(0, 0), (1, 0), (2, 0), (2, 1)],
        [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)],
        [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (3, 1)],
    ]
    for _ in range(200):
        grid = np.zeros((16, 16), dtype=int)
        _draw_lattice(grid, 5, 9)
        colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(3, 5))
        used = np.zeros_like(grid, dtype=bool)
        ok = True
        for color, shape in zip(colors, shapes[:len(colors)]):
            for _try in range(80):
                top = random.randint(1, 11)
                left = random.randint(1, 5)
                cells = {(top + dr, left + dc) for dr, dc in shape}
                pad = {(r + dr, c + dc) for r, c in cells for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
                if all(0 <= r < 16 and 0 <= c < 8 and not used[r, c] for r, c in pad):
                    _put_shape(grid, top, left, color, shape)
                    for r, c in pad:
                        if 0 <= r < 16 and 0 <= c < 16:
                            used[r, c] = True
                    break
            else:
                ok = False
        if ok:
            return grid
    return generate_input()
