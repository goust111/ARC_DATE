import random
from collections import deque

import numpy as np


def _object_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) in (0, 2, 4):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) not in (0, 2, 4):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """Count how many red guide lines each gray anchor sees below it, shift that object upward by the count, and drop a gray ray from the moved anchor."""
    out = grid.copy()
    baseline = next(r for r in range(grid.shape[0]) if np.all(grid[r] == 4))
    guide_rows = {r for r in range(grid.shape[0]) if np.all(grid[r] == 2)}
    comps = _object_components(grid)
    moved = []
    for comp in comps:
        anchor = next((r, c) for r, c in comp if int(grid[r, c]) == 5)
        shift = sum(anchor[0] < gr < baseline for gr in guide_rows)
        moved.append((comp, anchor, shift))
        for r, c in comp:
            out[r, c] = 0
    for comp, anchor, shift in moved:
        for r, c in comp:
            out[r - shift, c] = int(grid[r, c])
    for _, anchor, shift in moved:
        r, c = anchor[0] - shift, anchor[1]
        rr = r + 1
        while rr < baseline and int(out[rr, c]) not in (2, 4):
            if out[rr, c] == 0:
                out[rr, c] = 5
            rr += 1
    for gr in guide_rows:
        out[gr, :] = 2
    out[baseline, :] = 4
    return out


def _grow_blob(rng: random.Random, size: int) -> list[tuple[int, int]]:
    cells = [(0, 0)]
    while len(cells) < size:
        r, c = rng.choice(cells)
        nbrs = [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
        rng.shuffle(nbrs)
        for nbr in nbrs:
            if nbr not in cells and -1 <= nbr[0] <= 1 and -1 <= nbr[1] <= 1:
                cells.append(nbr)
                break
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return [(r - min_r, c - min_c) for r, c in cells]


def generate_input() -> np.ndarray:
    rng = random.Random()
    while True:
        h = rng.randint(11, 14)
        w = rng.randint(13, 16)
        grid = np.zeros((h, w), dtype=int)
        baseline = h - 1
        grid[baseline, :] = 4
        line_count = rng.randint(2, 4)
        guide_rows = sorted(rng.sample(range(1, baseline - 1), k=line_count))
        for r in guide_rows:
            grid[r, :] = 2

        objects = []
        colors = rng.sample([1, 3, 7, 9], k=rng.randint(3, 4))
        ok = True
        for color in colors:
            shape = _grow_blob(rng, rng.randint(3, 5))
            anchor = rng.choice(shape)
            max_r = max(r for r, _ in shape)
            max_c = max(c for _, c in shape)
            positions = [(r, c) for r in range(0, baseline - max_r) for c in range(0, w - max_c)]
            rng.shuffle(positions)
            placed = None
            for sr, sc in positions:
                cells = [(sr + r, sc + c) for r, c in shape]
                if any(rr in guide_rows or rr == baseline for rr, _ in cells):
                    continue
                shift = sum((sr + anchor[0]) < gr < baseline for gr in guide_rows)
                moved = {(rr - shift, cc) for rr, cc in cells}
                if any(rr < 0 for rr, _ in moved):
                    continue
                if any(grid[rr, cc] != 0 for rr, cc in cells):
                    continue
                if any((rr, cc) in other for other in objects for rr, cc in moved):
                    continue
                if any(rr in guide_rows or rr == baseline for rr, _ in moved):
                    continue
                placed = (cells, moved, (sr + anchor[0], sc + anchor[1]))
                break
            if placed is None:
                ok = False
                break
            cells, moved, anchor_abs = placed
            objects.append(set(moved))
            for rr, cc in cells:
                grid[rr, cc] = color
            grid[anchor_abs] = 5
        if ok:
            return grid
