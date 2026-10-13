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
        color = mask[sr, sc]
        q = deque([(int(sr), int(sc))])
        seen[sr, sc] = True
        comp = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] == color and color != 0 and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _has_hole(arr, comp):
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
    sub = arr[r0:r1 + 1, c0:c1 + 1]
    blocked = sub != 0
    seen = np.zeros_like(blocked, dtype=bool)
    q = deque()
    for r in range(sub.shape[0]):
        for c in (0, sub.shape[1] - 1):
            if not blocked[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(sub.shape[1]):
        for r in (0, sub.shape[0] - 1):
            if not blocked[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < sub.shape[0] and 0 <= nc < sub.shape[1] and not blocked[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return np.any((~blocked) & (~seen))


def main(grid: np.ndarray) -> np.ndarray:
    totals = {int(c): int(np.sum(grid == c)) for c in set(grid.ravel()) if c != 0}
    minority = min(totals, key=totals.get)
    work = grid.copy()
    work[work == minority] = 0
    comps = _components(work)
    out = work.copy()
    transformed = []
    for comp in comps:
        color = int(work[comp[0]])
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
        for r, c in comp:
            out[r, c] = 0
        new_comp = []
        if _has_hole(work, comp):
            for r, c in comp:
                nc = c0 + c1 - c
                out[r, nc] = color
                new_comp.append((r, nc))
        else:
            for r, c in comp:
                out[r, c] = color
                new_comp.append((r, c))
        transformed.append(new_comp)
    final = out.copy()
    for comp in transformed:
        for r, c in comp:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr or dc:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < final.shape[0] and 0 <= nc < final.shape[1] and final[nr, nc] == 0:
                            final[nr, nc] = 2
    return final


def _place(grid, top, left, color, shape):
    for r, c in shape:
        grid[top + r, left + c] = color


def generate_input() -> np.ndarray:
    hole = [(0, 1), (0, 2), (0, 3), (1, 0), (1, 3), (2, 0), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3), (1, 4)]
    solid_shapes = [
        [(0, 0), (1, 0), (1, 1)],
        [(0, 1), (1, 0), (1, 1), (2, 1)],
        [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)],
        [(0, 0), (1, 0), (2, 0), (2, 1)],
    ]
    for _ in range(100):
        grid = np.zeros((14, 14), dtype=int)
        placements = [(1, 1), (1, 8), (7, 1), (8, 8), (5, 5), (10, 4)]
        colors = [1, 3, 4, 5, 8, 1]
        random.shuffle(placements)
        _place(grid, placements[0][0], placements[0][1], colors[0], hole)
        for i, shape in enumerate(solid_shapes[:4], start=1):
            _place(grid, placements[i][0], placements[i][1], colors[i], shape)
        _place(grid, placements[5][0], placements[5][1], colors[5], [(0, 0), (0, 1)])
        totals = {c: int(np.sum(grid == c)) for c in set(grid.ravel()) if c}
        vals = list(totals.values())
        if vals.count(min(vals)) == 1 and any(c != min(totals, key=totals.get) for c in totals):
            return grid
    return generate_input()
