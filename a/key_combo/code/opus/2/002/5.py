import itertools
import random
from collections import deque

import numpy as np


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    comps = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        comp = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(comp):
    rs = [p[0] for p in comp]
    cs = [p[1] for p in comp]
    return min(rs), max(rs) + 1, min(cs), max(cs) + 1


def _corner(grid):
    pts = set(map(tuple, np.argwhere(grid == 6)))
    for r in range(grid.shape[0] - 1):
        for c in range(grid.shape[1] - 1):
            cells = {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)}
            if len(cells & pts) == 3:
                return next(iter(cells - pts))
    return (0, 0)


def _holes(arr):
    mask = arr == 0
    seen = np.zeros(mask.shape, dtype=bool)
    holes = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        comp = []
        border = False
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            if r in (0, arr.shape[0] - 1) or c in (0, arr.shape[1] - 1):
                border = True
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < arr.shape[0] and 0 <= nc < arr.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        if not border:
            holes.append(comp)
    return holes


def _find_template(grid):
    mask = (grid != 0) & (grid != 6) & (grid != 7)
    comps = _components(mask)
    if not comps:
        return np.zeros((1, 1), dtype=int), None
    comp = max(comps, key=len)
    r0, r1, c0, c1 = _bbox(comp)
    return grid[r0:r1, c0:c1].copy(), (r0, r1, c0, c1)


def _assemble(pieces):
    best = None
    best_score = -1
    for order in itertools.permutations(pieces):
        if len({p.shape[0] for p in order}) != 1:
            continue
        arr = np.concatenate(order, axis=1)
        if not (np.all(arr[0] == 7) and np.all(arr[-1] == 7) and np.all(arr[:, 0] == 7) and np.all(arr[:, -1] == 7)):
            continue
        score = len(_holes(arr))
        if score > best_score:
            best = arr
            best_score = score
    if best is not None:
        return best
    return max(pieces, key=lambda p: p.size)


def main(grid: np.ndarray) -> np.ndarray:
    template, _ = _find_template(grid)
    pieces = []
    for comp in _components(grid == 7):
        r0, r1, c0, c1 = _bbox(comp)
        pieces.append(np.where(grid[r0:r1, c0:c1] == 7, 7, 0))
    assembled = _assemble(pieces)
    out = np.zeros_like(grid)
    for hole in _holes(assembled):
        r0, r1, c0, c1 = _bbox(hole)
        if (r1 - r0, c1 - c0) == template.shape:
            assembled[r0:r1, c0:c1] = np.where(template != 0, template, assembled[r0:r1, c0:c1])
    tr, tc = _corner(grid)
    h, w = assembled.shape
    out[tr : tr + h, tc : tc + w] = assembled[: max(0, min(h, out.shape[0] - tr)), : max(0, min(w, out.shape[1] - tc))]
    return out


def _place(grid, arr, forbidden):
    h, w = arr.shape
    spots = [(r, c) for r in range(grid.shape[0] - h + 1) for c in range(grid.shape[1] - w + 1)]
    random.shuffle(spots)
    for r, c in spots:
        area = {(rr, cc) for rr in range(r - 1, r + h + 1) for cc in range(c - 1, c + w + 1)}
        if any(0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1] and (rr, cc) in forbidden for rr, cc in area):
            continue
        grid[r : r + h, c : c + w] = np.where(arr != 0, arr, grid[r : r + h, c : c + w])
        forbidden.update((rr, cc) for rr in range(r, r + h) for cc in range(c, c + w))
        return True
    return False


def generate_input() -> np.ndarray:
    grid = np.zeros((18, 18), dtype=int)
    final = np.full((6, 8), 7, dtype=int)
    final[2:4, 4:6] = 0
    splits = [final[:, :3], final[:, 3:5], final[:, 5:]]
    forbidden = set()
    for piece in random.sample(splits, len(splits)):
        _place(grid, piece, forbidden)

    a, b = random.sample([1, 2, 3, 4, 8, 9], 2)
    template = np.array([[a, b], [b, a]], dtype=int)
    _place(grid, template, forbidden)

    corner = (random.randint(0, 11), random.randint(0, 9))
    r, c = corner
    for rr, cc in [(r + 1, c), (r, c + 1), (r + 1, c + 1)]:
        grid[rr, cc] = 6
    return grid
