import random
from collections import deque
import numpy as np


SHAPES = [
    [(0, 0), (1, 0), (1, 1), (2, 1)],
    [(0, 1), (1, 1), (1, 0), (2, 1), (2, 2)],
    [(0, 0), (0, 1), (1, 0), (2, 0)],
    [(0, 0), (1, 0), (1, 1), (1, 2)],
    [(0, 1), (1, 0), (1, 1), (2, 0)],
]


def _components(grid, color):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r, c in zip(*np.where(grid == color)):
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        comp = []
        while q:
            rr, cc = q.popleft()
            comp.append((rr, cc))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == color and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _round_half_up(x):
    return int(np.floor(float(x) + 0.5))


def _rot(dr, dc, r):
    if r == 0:
        return dr, dc
    if r == 1:
        return dc, -dr
    if r == 2:
        return -dr, -dc
    return -dc, dr


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    h, w = grid.shape
    refs = np.argwhere(grid == 5)
    if refs.size == 0:
        return grid
    pr, pc = refs[0]
    res = grid.copy()
    res[pr, pc] = 0
    for color in [int(c) for c in np.unique(grid) if c not in (0, 5)]:
        for comp in _components(grid, color):
            cr = _round_half_up(sum(r for r, _ in comp) / len(comp))
            cc = _round_half_up(sum(c for _, c in comp) / len(comp))
            turns = (abs(cr - int(pr)) + abs(cc - int(pc))) % 4
            for r, c in comp:
                res[r, c] = 0
            for r, c in comp:
                nr, nc = _rot(r - cr, c - cc, turns)
                rr, cc2 = cr + nr, cc + nc
                if 0 <= rr < h and 0 <= cc2 < w:
                    res[rr, cc2] = color
    return res


def generate_input() -> np.ndarray:
    for _ in range(100):
        h = w = random.choice([16, 17, 18])
        g = np.zeros((h, w), dtype=int)
        ref = (h // 2 + random.choice([-1, 0, 1]), w // 2 + random.choice([-1, 0, 1]))
        g[ref] = 5
        colors = random.sample([1, 2, 3, 4, 7, 8], random.randint(4, 6))
        occupied = {(r, c) for r in range(ref[0] - 1, ref[0] + 2) for c in range(ref[1] - 1, ref[1] + 2)}
        residues = set()
        placed = 0
        for color in colors:
            shape = random.choice(SHAPES)
            candidates = []
            max_r = max(r for r, _ in shape)
            max_c = max(c for _, c in shape)
            for ar in range(1, h - max_r - 1):
                for ac in range(1, w - max_c - 1):
                    cells = {(ar + r, ac + c) for r, c in shape}
                    halo = {(r + dr, c + dc) for r, c in cells for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
                    if not (halo & occupied):
                        candidates.append((cells, halo))
            if not candidates:
                continue
            cells, halo = random.choice(candidates)
            for p in cells:
                g[p] = color
            cr = _round_half_up(sum(r for r, _ in cells) / len(cells))
            cc = _round_half_up(sum(c for _, c in cells) / len(cells))
            residues.add((abs(cr - ref[0]) + abs(cc - ref[1])) % 4)
            occupied |= halo
            placed += 1
        if placed >= 4 and len(residues) >= 3 and g[ref] == 5:
            return g
    return g
