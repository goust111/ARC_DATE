import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
RED = 2


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            out.append(comp)
    return out


def _inner_cells(box):
    r0, r1, c0, c1 = box
    return [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    containers = []
    for comp in _components(grid == GREY):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 >= 3 and c1 - c0 >= 3:
            containers.append((r0, r1, c0, c1))
    containers.sort(key=lambda b: (b[2], b[0]))
    graph = {i: [] for i in range(len(containers))}
    for i in range(len(containers) - 1):
        graph[i].append(i + 1)
        graph[i + 1].append(i)
    contents = []
    start = None
    for i, box in enumerate(containers):
        vals = [int(grid[r, c]) for r, c in _inner_cells(box) if grid[r, c] not in (BG, GREY, RED)]
        contents.append(vals)
        if len(graph[i]) == 1 and vals:
            start = i
    if start is None:
        start = 0
    order = [start]
    prev = None
    cur = start
    while True:
        nxts = [n for n in graph[cur] if n != prev]
        if not nxts:
            break
        nxt = nxts[0]
        order.append(nxt)
        prev, cur = cur, nxt
    new_contents = [[] for _ in containers]
    for idx, container_idx in enumerate(order):
        if idx == 0:
            new_contents[container_idx] = []
        else:
            new_contents[container_idx] = contents[order[idx - 1]]
    out = grid.copy()
    for i, box in enumerate(containers):
        cells = _inner_cells(box)
        for r, c in cells:
            out[r, c] = BG
        for (r, c), val in zip(cells, new_contents[i]):
            out[r, c] = val
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((24, 24), dtype=int)
    boxes = [(3, 8, 2, 7), (3, 8, 10, 15), (3, 8, 18, 23)]
    for r0, r1, c0, c1 in boxes:
        grid[r0, c0 : c1 + 1] = GREY
        grid[r1, c0 : c1 + 1] = GREY
        grid[r0 : r1 + 1, c0] = GREY
        grid[r0 : r1 + 1, c1] = GREY
    for c in range(7, 11):
        grid[5, c] = RED
    for c in range(15, 19):
        grid[5, c] = RED
    colors = random.sample([1, 3, 4, 6, 7, 8, 9], random.randint(3, 6))
    cells = _inner_cells(boxes[0])
    random.shuffle(cells)
    for (r, c), color in zip(cells, colors):
        grid[r, c] = color
    if random.choice([True, False]):
        cells = _inner_cells(boxes[1])
        for (r, c), color in zip(cells, random.sample(colors, min(len(cells), random.randint(1, 3)))):
            grid[r, c] = color
    return grid
