import random
from collections import Counter, deque
import numpy as np


def _components_color(grid, color):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                x, y = q.popleft()
                cells.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and not seen[nx, ny] and grid[nx, ny] == color:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(cells)
    return comps


def _interior(grid, outline):
    rs = [r for r, _ in outline]
    cs = [c for _, c in outline]
    r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
    block = set(outline)
    q = deque()
    seen = set()
    for r in range(r0, r1 + 1):
        for c in (c0, c1):
            if (r, c) not in block:
                q.append((r, c)); seen.add((r, c))
    for c in range(c0, c1 + 1):
        for r in (r0, r1):
            if (r, c) not in block and (r, c) not in seen:
                q.append((r, c)); seen.add((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if r0 <= nr <= r1 and c0 <= nc <= c1 and (nr, nc) not in block and (nr, nc) not in seen:
                seen.add((nr, nc)); q.append((nr, nc))
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if (r, c) not in block and (r, c) not in seen]


def _symmetric(outline):
    rs = [r for r, _ in outline]
    cs = [c for _, c in outline]
    c0, c1 = min(cs), max(cs)
    cells = set(outline)
    return all((r, c0 + c1 - c) in cells for r, c in cells)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    out = g.copy()
    for outline in _components_color(g, 5):
        if len(outline) < 8:
            continue
        inside = _interior(g, outline)
        marks = Counter(int(g[r, c]) for r, c in inside if g[r, c] not in (0, 5))
        if not marks:
            continue
        fill = max(marks, key=lambda k: (marks[k], -k)) if _symmetric(outline) else min(marks, key=lambda k: (marks[k], k))
        for r, c in inside:
            out[r, c] = fill
    return out


def _rect(g, top, left, ht, wd, asym=False):
    g[top, left:left + wd] = 5
    g[top + ht - 1, left:left + wd] = 5
    g[top:top + ht, left] = 5
    g[top:top + ht, left + wd - 1] = 5
    if asym:
        g[top + 1, left + wd - 1] = 0
        g[top + 1, left + wd - 2] = 5


def generate_input() -> np.ndarray:
    g = np.zeros((16, 16), dtype=int)
    slots = [(1, 1), (1, 9), (9, 1), (9, 9)]
    random.shuffle(slots)
    n = random.randint(3, 4)
    asyms = [False, True] + [random.choice([False, True]) for _ in range(n - 2)]
    for i in range(n):
        top, left = slots[i]
        ht = random.choice([5, 6])
        wd = random.choice([5, 6])
        _rect(g, top, left, ht, wd, asyms[i])
        cells = [(r, c) for r in range(top + 1, top + ht - 1) for c in range(left + 1, left + wd - 1) if g[r, c] == 0]
        colors = random.sample([1, 2, 3, 4], 2)
        random.shuffle(cells)
        g[cells[0]] = colors[0]
        g[cells[1]] = colors[0]
        g[cells[2]] = colors[1]
    return g
