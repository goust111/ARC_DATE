import random
from collections import deque
import numpy as np

AZURE = 8
RED = 2
GREEN = 3
GREY = 5


def components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not seen[r, c]:
                q = deque([(r, c)])
                seen[r, c] = True
                cells = []
                while q:
                    x, y = q.popleft()
                    cells.append((x, y))
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                            seen[nx, ny] = True
                            q.append((nx, ny))
                comps.append(cells)
    return comps


def turn_count(cells):
    s = set(cells)
    nbrs = {p: [q for q in ((p[0] + 1, p[1]), (p[0] - 1, p[1]), (p[0], p[1] + 1), (p[0], p[1] - 1)) if q in s] for p in s}
    ends = [p for p, ns in nbrs.items() if len(ns) == 1]
    if not ends:
        return 0
    path = [ends[0]]
    prev = None
    cur = ends[0]
    visited = {cur}
    while True:
        nxts = [p for p in nbrs[cur] if p != prev]
        if not nxts:
            break
        nxt = nxts[0]
        if nxt in visited:
            break
        path.append(nxt)
        prev, cur = cur, nxt
        visited.add(cur)
    if len(path) < 3:
        return 0
    dirs = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    return sum(d1 != d2 for d1, d2 in zip(dirs, dirs[1:]))


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    out = grid.copy()
    for cells in components(grid == GREY):
        colors = set()
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    v = grid[nr, nc]
                    if v not in (AZURE, GREY):
                        colors.add(v)
        if len(colors) != 1:
            continue
        color = next(iter(colors))
        final = RED if turn_count(cells) % 2 else GREEN
        out[grid == color] = final
        for r, c in cells:
            out[r, c] = final
    return out


def path_between(a, b, bends):
    pts = [a]
    cur = a
    for nxt in bends + [b]:
        if cur[0] == nxt[0]:
            step = 1 if nxt[1] > cur[1] else -1
            for c in range(cur[1] + step, nxt[1] + step, step):
                pts.append((cur[0], c))
        else:
            step = 1 if nxt[0] > cur[0] else -1
            for r in range(cur[0] + step, nxt[0] + step, step):
                pts.append((r, cur[1]))
        cur = nxt
    return pts


def generate_input() -> np.ndarray:
    h, w = random.randint(18, 24), random.randint(18, 24)
    grid = np.full((h, w), AZURE, dtype=int)
    palette = random.sample([1, 4, 6, 7, 9], random.randint(2, 4))
    rows = sorted(random.sample(range(2, h - 4), len(palette)))
    for idx, color in enumerate(palette):
        r1 = rows[idx]
        r2 = min(h - 3, r1 + random.randint(4, 7))
        c1 = random.randint(1, w // 3)
        c2 = random.randint(2 * w // 3, w - 2)
        len1 = random.randint(3, 5)
        len2 = random.randint(3, 5)
        horizontal = random.random() < 0.5
        if horizontal:
            grid[r1, c1 : c1 + len1] = color
            grid[r2, c2 - len2 + 1 : c2 + 1] = color
            a = (r1, c1 + len1 - 1)
            b = (r2, c2 - len2 + 1)
        else:
            grid[r1 : r1 + len1, c1] = color
            grid[r2 - len2 + 1 : r2 + 1, c2] = color
            a = (r1 + len1 - 1, c1)
            b = (r2 - len2 + 1, c2)
        style = random.choice([0, 1, 2])
        if style == 0:
            bends = [(a[0], b[1])]
        elif style == 1 and abs(a[0] - b[0]) > 2:
            mid = random.randint(min(a[0], b[0]) + 1, max(a[0], b[0]) - 1)
            bends = [(mid, a[1]), (mid, b[1])]
        elif style == 2 and abs(a[1] - b[1]) > 2:
            mid = random.randint(min(a[1], b[1]) + 1, max(a[1], b[1]) - 1)
            bends = [(a[0], mid), (b[0], mid)]
        else:
            bends = [(b[0], a[1])]
        for r, c in path_between(a, b, bends)[1:-1]:
            if 0 <= r < h and 0 <= c < w:
                grid[r, c] = GREY
    return grid
