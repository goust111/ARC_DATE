import random
from collections import deque
import numpy as np


def _frame_boxes(grid):
    h, w = grid.shape
    boxes = []
    runs = []
    for r in range(h):
        c = 0
        while c < w:
            if grid[r, c] != 5:
                c += 1
                continue
            start = c
            while c < w and grid[r, c] == 5:
                c += 1
            if c - start >= 6:
                runs.append((r, start, c - 1))
    for i, (r1, c1, c2) in enumerate(runs):
        for r2, d1, d2 in runs[i + 1:]:
            if r2 - r1 < 5 or (c1, c2) != (d1, d2):
                continue
            if (np.all(grid[r1:r2 + 1, c1] == 5) and np.all(grid[r1:r2 + 1, c2] == 5) and
                    np.any(grid[r1 + 1:r2, c1 + 1:c2] == 7)):
                boxes.append((r1, c1, r2, c2))
    boxes.sort(key=lambda b: -((b[2] - b[0] + 1) * (b[3] - b[1] + 1)))
    kept = []
    for b in boxes:
        area = set((r, c) for r in range(b[0], b[2] + 1) for c in range(b[1], b[3] + 1))
        if not any(area & a for _, a in kept):
            kept.append((b, area))
    return [b for b, _ in kept]


def _regions(grid, box):
    r1, c1, r2, c2 = box
    seen = set()
    n = 0
    for r in range(r1 + 1, r2):
        for c in range(c1 + 1, c2):
            if grid[r, c] == 5 or (r, c) in seen:
                continue
            n += 1
            q = deque([(r, c)])
            seen.add((r, c))
            while q:
                rr, cc = q.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if r1 < nr < r2 and c1 < nc < c2 and grid[nr, nc] != 5 and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        q.append((nr, nc))
    return n


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    res = grid.copy()
    for r1, c1, r2, c2 in _frame_boxes(grid):
        n = _regions(grid, (r1, c1, r2, c2))
        motif = [(r, c) for r, c in zip(*np.where(grid[r1 + 1:r2, c1 + 1:c2] == 7))]
        motif = [(r + r1 + 1, c + c1 + 1) for r, c in motif]
        for r, c in motif:
            if n % 2:
                res[r, c1 + c2 - c] = 7
            else:
                res[r1 + r2 - r, c] = 7
        for r in range(r1 + 1, r2):
            for c in range(c1 + 1, c2):
                if res[r, c] == 5:
                    res[r, c] = 0
    return res


def _draw_frame(g, r, c, s):
    g[r, c:c + s] = 5
    g[r + s - 1, c:c + s] = 5
    g[r:r + s, c] = 5
    g[r:r + s, c + s - 1] = 5


def _motif_cells(allowed, count):
    comps = []
    remaining = set(allowed)
    while remaining:
        start = remaining.pop()
        comp = {start}
        stack = [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    comp.add(p)
                    stack.append(p)
        comps.append(comp)
    allowed = max(comps, key=len)
    count = min(count, len(allowed))
    start = random.choice(tuple(allowed))
    cells = {start}
    while len(cells) < count:
        base = random.choice(tuple(cells))
        nbrs = [(base[0] + dr, base[1] + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        nbrs = [p for p in nbrs if p in allowed and p not in cells]
        if nbrs:
            cells.add(random.choice(nbrs))
        elif len(cells) == 1:
            start = random.choice(tuple(allowed))
            cells = {start}
        else:
            frontier = [p for cell in cells for p in ((cell[0] + 1, cell[1]), (cell[0] - 1, cell[1]),
                        (cell[0], cell[1] + 1), (cell[0], cell[1] - 1)) if p in allowed and p not in cells]
            if frontier:
                cells.add(random.choice(frontier))
    return cells


def generate_input() -> np.ndarray:
    g = np.zeros((20, 20), dtype=int)
    slots = [(1, 1), (1, 11), (11, 1)]
    random.shuffle(slots)
    counts = random.sample([2, 3, 4], random.randint(2, 3))
    for n, (r, c) in zip(counts, slots):
        s = random.choice([7, 8])
        _draw_frame(g, r, c, s)
        if n == 2:
            x = c + s // 2
            g[r + 1:r + s - 1, x] = 5
        elif n == 3:
            xs = [c + 2, c + s - 3]
            for x in sorted(set(xs)):
                g[r + 1:r + s - 1, x] = 5
        else:
            x = c + s // 2
            y = r + s // 2
            g[r + 1:r + s - 1, x] = 5
            g[y, c + 1:c + s - 1] = 5
        if n % 2:
            allowed = {(rr, cc) for rr in range(r + 1, r + s - 1) for cc in range(c + 1, c + s // 2) if g[rr, cc] == 0}
        else:
            allowed = {(rr, cc) for rr in range(r + 1, r + s // 2) for cc in range(c + 1, c + s - 1) if g[rr, cc] == 0}
        for rr, cc in _motif_cells(allowed, random.randint(3, min(6, len(allowed)))):
            g[rr, cc] = 7
    return g
