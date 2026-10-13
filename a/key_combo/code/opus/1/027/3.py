import random
from collections import deque
import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
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


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    h, w = grid.shape
    wall = 2
    chambers = []
    chamber_cells = set()
    for comp in _components(grid != wall):
        touches_edge = any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp)
        if not touches_edge and all(grid[r, c] == 0 for r, c in comp):
            chambers.append(comp)
            chamber_cells.update(comp)

    samples = []
    for r, c in np.argwhere((grid != 0) & (grid != wall)):
        p = (int(r), int(c))
        if p not in chamber_cells:
            samples.append((p[0], p[1], int(grid[p])))

    period = 1
    cycle = {}
    for p in range(1, 7):
        candidate = {}
        ok = True
        for r, c, color in samples:
            residue = (r + c) % p
            if residue in candidate and candidate[residue] != color:
                ok = False
                break
            candidate[residue] = color
        if ok and len(candidate) == p:
            period = p
            cycle = candidate
            break

    out = grid.copy()
    for comp in chambers:
        rows = [p[0] for p in comp]
        cols = [p[1] for p in comp]
        residue = (((min(rows) + max(rows)) // 2) + ((min(cols) + max(cols)) // 2)) % period
        color = cycle[residue]
        for r, c in comp:
            out[r, c] = color
    return out


def _draw_rect(grid, top, left, bottom, right, color):
    grid[top, left:right + 1] = color
    grid[bottom, left:right + 1] = color
    grid[top:bottom + 1, left] = color
    grid[top:bottom + 1, right] = color


def generate_input() -> np.ndarray:
    rng = random.Random()
    size = rng.choice([15, 17, 19, 21])
    grid = np.zeros((size, size), dtype=int)
    rects = [
        (3, 3, 7, 7),
        (3, size - 8, 7, size - 4),
        (size - 8, 3, size - 4, 7),
        (size - 8, size - 8, size - 4, size - 4),
    ]
    if size >= 19:
        rects.append((size // 2 - 2, size // 2 - 2, size // 2 + 2, size // 2 + 2))
    rng.shuffle(rects)
    for rect in rects[:rng.randint(4, min(5, len(rects)))]:
        _draw_rect(grid, *rect, 2)

    period = rng.choice([2, 3])
    colors = [1, 3, 4, 6, 8, 9]
    rng.shuffle(colors)
    cycle = colors[:period]
    blocked = set()
    for top, left, bottom, right in rects:
        blocked.update((r, c) for r in range(top, bottom + 1) for c in range(left, right + 1))
    outside = [(r, c) for r in range(size) for c in range(size) if (r, c) not in blocked and grid[r, c] == 0]
    for residue in range(period):
        cells = [(r, c) for r, c in outside if (r + c) % period == residue]
        for r, c in rng.sample(cells, min(3, len(cells))):
            grid[r, c] = cycle[residue]
    return grid
