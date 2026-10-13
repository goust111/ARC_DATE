import random
from collections import deque
import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for sr, sc in zip(*np.where(mask)):
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
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        comps.append(cells)
    return comps


def _bbox(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    out = grid.copy()
    fg_comps = _components((grid != 0) & (grid != 5))
    template_cells = max(fg_comps, key=len)
    tr0, tc0, tr1, tc1 = _bbox(template_cells)
    template = grid[tr0:tr1 + 1, tc0:tc1 + 1]

    for cells in _components(grid == 5):
        r0, c0, r1, c1 = _bbox(cells)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        ih, iw = r1 - r0 - 1, c1 - c0 - 1
        choices = []
        for k in range(4):
            rot = np.rot90(template, k)
            th, tw = rot.shape
            if ih % th == 0 and iw % tw == 0 and ih // th == iw // tw:
                choices.append((k, rot, ih // th))
        if not choices:
            continue
        if len(choices) > 1:
            choices.sort(key=lambda x: (int(x[1][0, 0]), x[0]))
        _, rot, scale = choices[0]
        fill = np.kron(rot, np.ones((scale, scale), dtype=grid.dtype))
        out[r0 + 1:r1, c0 + 1:c1] = fill
    return out


def _draw_rect(grid, top, left, ih, iw):
    bottom, right = top + ih + 1, left + iw + 1
    grid[top, left:right + 1] = 5
    grid[bottom, left:right + 1] = 5
    grid[top:bottom + 1, left] = 5
    grid[top:bottom + 1, right] = 5


def _free(grid, top, left, h, w):
    gh, gw = grid.shape
    if top < 0 or left < 0 or top + h > gh or left + w > gw:
        return False
    area = grid[max(0, top - 1):min(gh, top + h + 1), max(0, left - 1):min(gw, left + w + 1)]
    return np.all(area == 0)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(1000):
        grid = np.zeros((28, 28), dtype=int)
        th, tw = rng.choice([(2, 3), (3, 2)])
        vals = rng.choices([1, 2, 3, 4], k=th * tw)
        template = np.array(vals, dtype=int).reshape(th, tw)
        grid[1:1 + th, 1:1 + tw] = template
        specs = []
        for _i in range(rng.randint(2, 4)):
            k = rng.randrange(4)
            rot = np.rot90(template, k)
            scale = rng.choice([2, 3])
            specs.append((rot.shape[0] * scale, rot.shape[1] * scale))
        for ih, iw in specs:
            placed = False
            for _try in range(300):
                top = rng.randint(1, 26 - ih)
                left = rng.randint(6, 26 - iw)
                if _free(grid, top, left, ih + 2, iw + 2):
                    _draw_rect(grid, top, left, ih, iw)
                    placed = True
                    break
            if not placed:
                break
        else:
            return grid
    return generate_input()
