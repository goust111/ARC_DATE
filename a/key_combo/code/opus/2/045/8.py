import random
from collections import deque

import numpy as np

BG = 0
BLUE = 1
RED = 2
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                x, y = q.popleft()
                cells.append((x, y))
                for dr, dc in DIRS4:
                    nx, ny = x + dr, y + dc
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(cells)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int, copy=False)
    h, w = g.shape

    lu = np.fliplr(g)
    rv = np.flipud(g)
    rd = np.flipud(np.fliplr(g))

    out = np.zeros((2 * h, 2 * w), dtype=int)
    out[:h, :w] = g
    out[:h, w:] = lu
    out[h:, :w] = rv
    out[h:, w:] = rd

    mask = out == BLUE
    comps = _components(mask)

    def _bbox(cells):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        return min(rs), max(rs), min(cs), max(cs)

    items = []
    for comp in comps:
        area = len(comp)
        r0, r1, c0, c1 = _bbox(comp)
        items.append((area, r0, c0, comp))

    items.sort(key=lambda x: (-x[0], x[1], x[2]))

    for rank, (_area, _r0, _c0, comp) in enumerate(items):
        if rank % 2 == 0:
            color = RED
        else:
            color = BLUE
        for r, c in comp:
            out[r, c] = color

    return out


def _draw_component(grid: np.ndarray, start, target_size: int):
    h, w = grid.shape
    cells = [start]
    occupied = {start}
    frontier = [start]

    while len(cells) < target_size:
        random.shuffle(frontier)
        if not frontier:
            return
        x, y = frontier[0]
        nbrs = []
        for dr, dc in DIRS4:
            nx, ny = x + dr, y + dc
            if 0 <= nx < h and 0 <= ny < w and (nx, ny) not in occupied:
                nbrs.append((nx, ny))
        if not nbrs:
            frontier.pop(0)
            continue
        nx, ny = random.choice(nbrs)
        cells.append((nx, ny))
        occupied.add((nx, ny))
        frontier.append((nx, ny))

    for r, c in cells:
        grid[r, c] = BLUE


def generate_input() -> np.ndarray:
    h, w = 5, 4
    g = np.zeros((h, w), dtype=int)

    total = random.randint(3, 5)
    if total <= 2:
        total = 3
    comp_count = random.randint(2, 3)
    sizes = [1] * comp_count
    remain = total - comp_count
    for i in range(comp_count):
        if remain <= 0:
            break
        add = random.randint(0, remain)
        sizes[i] += add
        remain -= add

    used = set()
    for i in range(comp_count):
        size = max(1, sizes[i])
        if i == 0:
            # enforce border contact
            border = [(0, c) for c in range(w)] + [(h - 1, c) for c in range(w)] + [(r, 0) for r in range(h)] + [(r, w - 1) for r in range(h)]
            border = [p for p in border if p not in used]
            start = random.choice(border)
        else:
            free = [(r, c) for r in range(h) for c in range(w) if (r, c) not in used]
            if not free:
                break
            start = random.choice(free)
        _draw_component(g, start, size)
        for r in range(h):
            for c in range(w):
                if g[r, c] == BLUE:
                    used.add((r, c))

    return g
