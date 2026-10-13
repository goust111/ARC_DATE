import random
from collections import deque

import numpy as np


def _component(mask, start):
    q = deque([start])
    seen = {start}
    comp = []
    while q:
        r, c = q.popleft()
        comp.append((r, c))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            p = (r + dr, c + dc)
            if 0 <= p[0] < mask.shape[0] and 0 <= p[1] < mask.shape[1] and mask[p] and p not in seen:
                seen.add(p)
                q.append(p)
    return comp


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    out = grid.copy()
    hub = tuple(np.argwhere(grid == 5)[0])
    items = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        start = (hub[0] + dr, hub[1] + dc)
        if not (0 <= start[0] < grid.shape[0] and 0 <= start[1] < grid.shape[1]):
            continue
        color = int(grid[start])
        if color in (0, 5):
            continue
        path = [start]
        cur = start
        while True:
            nxt = (cur[0] + 2 * dr, cur[1] + 2 * dc)
            if 0 <= nxt[0] < grid.shape[0] and 0 <= nxt[1] < grid.shape[1] and grid[nxt] == color:
                path.append(nxt)
                cur = nxt
            else:
                break
        path_set = set(path)
        mask = grid == color
        sprites = []
        seen = set(path_set)
        for p in map(tuple, np.argwhere(mask)):
            if p in seen:
                continue
            comp = _component(mask, p)
            seen.update(comp)
            if any(abs(cur[0] - r) + abs(cur[1] - c) == 1 for r, c in comp):
                sprites.append(comp)
        if not sprites:
            continue
        sprite = sprites[0]
        cx = (min(c for _, c in sprite) + max(c for _, c in sprite)) / 2
        items.append({"color": color, "path": path, "sprite": sprite, "length": len(path), "cx": cx})
    length_order = sorted(items, key=lambda x: x["length"])
    pos_order = sorted(items, key=lambda x: x["cx"])
    for target, source in zip(pos_order, length_order):
        new_color = source["color"]
        for p in target["path"] + target["sprite"]:
            out[p] = new_color
    out[hub] = 5
    return out


def _draw_path(grid, hub, direction, length, color, sprite):
    dr, dc = direction
    cur = (hub[0] + dr, hub[1] + dc)
    for _ in range(length):
        grid[cur] = color
        last = cur
        cur = (cur[0] + 2 * dr, cur[1] + 2 * dc)
    for r, c in sprite(last):
        grid[r, c] = color


def generate_input() -> np.ndarray:
    grid = np.zeros((16, 16), dtype=int)
    hub = (random.choice([7, 8]), random.choice([7, 8]))
    grid[hub] = 5
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 4)
    specs = [
        ((0, -1), 3, colors[0], lambda p: [(p[0], p[1] - 1), (p[0] - 1, p[1] - 1), (p[0], p[1] - 2), (p[0] + 1, p[1] - 1)]),
        ((-1, 0), 1, colors[1], lambda p: [(p[0] - 1, p[1]), (p[0] - 2, p[1]), (p[0] - 2, p[1] - 1), (p[0] - 2, p[1] + 1)]),
        ((0, 1), 2, colors[2], lambda p: [(p[0], p[1] + 1), (p[0] - 1, p[1] + 1), (p[0], p[1] + 2), (p[0] + 1, p[1] + 1)]),
    ]
    if random.random() < 0.5:
        specs.append(((1, 0), 3, colors[3], lambda p: [(p[0], p[1] + 1), (p[0] - 1, p[1] + 1), (p[0], p[1] + 2), (p[0] - 1, p[1] + 2)]))
    for direction, length, color, sprite in specs:
        _draw_path(grid, hub, direction, length, color, sprite)
    return grid
