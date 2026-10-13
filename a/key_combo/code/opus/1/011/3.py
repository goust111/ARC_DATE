import heapq
import random
from collections import deque

import numpy as np


def _components_of(grid, vals):
    h, w = grid.shape
    vals = set(vals)
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] not in vals or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and not seen[nx, ny] and grid[nx, ny] in vals:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(comp)
    return comps


def _cup_info(comp, grid):
    rows = [p[0] for p in comp]
    cols = [p[1] for p in comp]
    r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
    pts = set(comp)
    direction = None
    tail = None
    if sum((r0, c) in pts for c in range(c0, c1 + 1)) == 1:
        tail = next((r0, c) for c in range(c0, c1 + 1) if (r0, c) in pts)
        r0 += 1
        direction = (-1, 0)
    elif sum((r1, c) in pts for c in range(c0, c1 + 1)) == 1:
        tail = next((r1, c) for c in range(c0, c1 + 1) if (r1, c) in pts)
        r1 -= 1
        direction = (1, 0)
    elif sum((r, c0) in pts for r in range(r0, r1 + 1)) == 1:
        tail = next((r, c0) for r in range(r0, r1 + 1) if (r, c0) in pts)
        c0 += 1
        direction = (0, -1)
    elif sum((r, c1) in pts for r in range(r0, r1 + 1)) == 1:
        tail = next((r, c1) for r in range(r0, r1 + 1) if (r, c1) in pts)
        c1 -= 1
        direction = (0, 1)
    if direction is None:
        direction = (-1, 0)
        tail = (r0, (c0 + c1) // 2)
    interior = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    reds = [(r, c) for r, c in interior if grid[r, c] == 2]
    return {"comp": set(comp), "body": (r0, r1, c0, c1), "tail": tail, "dir": direction, "reds": reds}


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    out = grid.copy()
    cup_comps = _components_of(grid, [1, 7])
    cups = [_cup_info(comp, grid) for comp in cup_comps]
    id_at = {}
    for i, cup in enumerate(cups):
        for p in cup["comp"]:
            id_at[p] = i
    edges = {}
    edge_cells = {}
    for gcomp in _components_of(grid, [3]):
        touched = set()
        for r, c in gcomp:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                p = (r + dr, c + dc)
                if p in id_at:
                    touched.add(id_at[p])
        if len(touched) == 2:
            a, b = sorted(touched)
            edges.setdefault(a, []).append(b)
            edges.setdefault(b, []).append(a)
            edge_cells[frozenset((a, b))] = set(gcomp)
    terminals = [i for i, cup in enumerate(cups) if grid[next(iter(cup["comp"]))] == 7]
    if len(terminals) != 2:
        return out
    def weight(a, b):
        return len(cups[a]["reds"]) + len(cups[b]["reds"])
    start, goal = terminals
    pq = [(0, start, [start])]
    best = {start: 0}
    final = [start]
    while pq:
        dist, node, path = heapq.heappop(pq)
        if node == goal:
            final = path
            break
        if dist != best[node]:
            continue
        for nb in sorted(edges.get(node, [])):
            nd = dist + weight(node, nb)
            if nd < best.get(nb, 10**9):
                best[nb] = nd
                heapq.heappush(pq, (nd, nb, path + [nb]))
    path_set = set(final)
    for i, cup in enumerate(cups):
        cnt = len(cup["reds"])
        for p in cup["reds"]:
            out[p] = 0
        if i in path_set and i not in terminals:
            tr, tc = cup["tail"]
            dr, dc = cup["dir"]
            r, c = tr + dr, tc + dc
            for _ in range(cnt):
                if 0 <= r < out.shape[0] and 0 <= c < out.shape[1] and out[r, c] == 0:
                    out[r, c] = 2
                r += dr
                c += dc
        elif i in terminals:
            for p in cup["reds"]:
                out[p] = 2
    for a, b in zip(final, final[1:]):
        for p in edge_cells.get(frozenset((a, b)), ()):
            out[p] = 5
    return out


def _draw_cup(grid, top, left, color, tail_dir, red_count):
    r0, c0 = top, left
    for c in range(c0, c0 + 5):
        grid[r0, c] = color
        grid[r0 + 4, c] = color
    for r in range(r0, r0 + 5):
        grid[r, c0] = color
        grid[r, c0 + 4] = color
    mid = (r0 + 2, c0 + 2)
    tr, tc = mid[0] + tail_dir[0] * 3, mid[1] + tail_dir[1] * 3
    grid[tr, tc] = color
    cells = [(r, c) for r in range(r0 + 1, r0 + 4) for c in range(c0 + 1, c0 + 4)]
    for r, c in random.sample(cells, red_count):
        grid[r, c] = 2


def _line(grid, a, b):
    r, c = a
    br, bc = b
    while (r, c) != (br, bc):
        grid[r, c] = 3
        r += (br > r) - (br < r)
        c += (bc > c) - (bc < c)
    grid[br, bc] = 3


def generate_input() -> np.ndarray:
    grid = np.zeros((25, 25), dtype=int)
    cups = {
        "A": (2, 2, 7, (-1, 0)),
        "B": (2, 16, 1, (-1, 0)),
        "C": (16, 2, 1, (1, 0)),
        "D": (16, 16, 7, (1, 0)),
    }
    bcnt, ccnt = random.sample(range(1, 6), 2)
    counts = {"A": random.randint(1, 4), "B": bcnt, "C": ccnt, "D": random.randint(1, 4)}
    for k, (r, c, color, td) in cups.items():
        _draw_cup(grid, r, c, color, td, counts[k])
    _line(grid, (4, 7), (4, 15))
    _line(grid, (7, 4), (15, 4))
    _line(grid, (7, 18), (15, 18))
    _line(grid, (18, 7), (18, 15))
    return grid
