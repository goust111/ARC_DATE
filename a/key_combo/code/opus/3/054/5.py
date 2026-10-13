import random
from collections import Counter, deque

import numpy as np


def _solid_square(grid, color):
    pts = np.argwhere(grid == color)
    r0, c0 = pts.min(axis=0)
    return int(r0), int(c0)


def _fits(passable, r, c):
    h, w = passable.shape
    return 0 <= r <= h - 3 and 0 <= c <= w - 3 and bool(np.all(passable[r : r + 3, c : c + 3]))


def _distances(passable, target):
    q = deque([target])
    dist = {target: 0}
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):
            nr, nc = r + dr, c + dc
            if (nr, nc) not in dist and _fits(passable, nr, nc):
                dist[(nr, nc)] = dist[(r, c)] + 1
                q.append((nr, nc))
    return dist


def main(grid: np.ndarray) -> np.ndarray:
    """Slide the yellow 3x3 block through the channel, collect path markers, and recolor the stopped block by the top count."""
    g = np.array(grid, dtype=int)
    out = g.copy()
    start = _solid_square(g, 4)
    target = _solid_square(g, 9)
    passable = np.isin(g, [1, 2, 3, 4, 5, 9])
    dist = _distances(passable, target)
    pos = start
    if pos not in dist:
        reach = _distances(passable, start)
        pos = min(reach, key=lambda p: (abs(p[0] - target[0]) + abs(p[1] - target[1]), reach[p]))
        path = [pos]
    else:
        path = [pos]
        while pos != target:
            r, c = pos
            choices = []
            for rank, (dr, dc) in enumerate(((-1, 0), (0, 1), (1, 0), (0, -1))):
                np5 = (r + dr, c + dc)
                if np5 in dist and dist[np5] < dist[pos]:
                    choices.append((dist[np5], rank, np5))
            if not choices:
                break
            pos = min(choices)[2]
            path.append(pos)

    counts = Counter()
    collected = set()
    for r, c in path[1:]:
        for rr in range(r, r + 3):
            for cc in range(c, c + 3):
                if int(g[rr, cc]) in (1, 2, 3) and (rr, cc) not in collected:
                    collected.add((rr, cc))
                    counts[int(g[rr, cc])] += 1
                    out[rr, cc] = 0
    out[start[0] : start[0] + 3, start[1] : start[1] + 3] = 5
    color = max([1, 2, 3], key=lambda k: (counts[k], -k))
    out[pos[0] : pos[0] + 3, pos[1] : pos[1] + 3] = color
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((15, 15), dtype=int)
    path = [(1, c) for c in range(1, 9)] + [(r, 8) for r in range(2, 9)]
    cells = set()
    for r, c in path:
        for rr in range(r, r + 3):
            for cc in range(c, c + 3):
                cells.add((rr, cc))
    for r, c in cells:
        grid[r, c] = 5
    grid[1:4, 1:4] = 4
    grid[8:11, 8:11] = 9
    candidates = [(r, c) for r, c in cells if grid[r, c] == 5 and not (8 <= r < 11 and 8 <= c < 11)]
    random.shuffle(candidates)
    counts = random.choice([{2: 5, 3: 3, 1: 2}, {1: 5, 2: 3, 3: 2}, {3: 5, 1: 3, 2: 2}])
    i = 0
    for color, n in counts.items():
        for _ in range(n):
            r, c = candidates[i]
            i += 1
            grid[r, c] = color
    return grid
