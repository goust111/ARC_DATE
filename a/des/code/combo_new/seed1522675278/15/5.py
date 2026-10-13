import random
from collections import deque

import numpy as np


BG = 0
WALL = 3


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


def _object_info(comp):
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    box = {(r - r0, c - c0) for r, c in comp}
    for r, c in comp:
        rr, cc = r - r0, c - c0
        if rr == 0 and 0 < cc < c1 - c0 and (r + 1, c) in comp:
            return "U", (r0 + 1, r1, c0, c1)
        if rr == r1 - r0 and 0 < cc < c1 - c0 and (r - 1, c) in comp:
            return "D", (r0, r1 - 1, c0, c1)
        if cc == 0 and 0 < rr < r1 - r0 and (r, c + 1) in comp:
            return "L", (r0, r1, c0 + 1, c1)
        if cc == c1 - c0 and 0 < rr < r1 - r0 and (r, c - 1) in comp:
            return "R", (r0, r1, c0, c1 - 1)
    return "R", (r0, r1, c0, c1 - 1)


def _dist(body_box, direction, occupied, shape):
    h, w = shape
    r0, r1, c0, c1 = body_box
    dr, dc = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}[direction]
    step = 0
    while True:
        step += 1
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                nr, nc = r + dr * step, c + dc * step
                if nr < 0 or nr >= h or nc < 0 or nc >= w or occupied[nr, nc]:
                    return step - 1


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    objects = []
    wall_cells = grid == WALL
    for comp in _components((grid != BG) & (grid != WALL)):
        color = int(grid[comp[0]])
        direction, box = _object_info(comp)
        objects.append({"color": color, "comp": comp, "direction": direction, "box": box})
    out = np.zeros_like(grid)
    out[wall_cells] = WALL
    stopped = wall_cells.copy()
    pending = objects[:]
    while pending:
        dists = []
        for obj in pending:
            dists.append((_dist(obj["box"], obj["direction"], stopped, grid.shape), obj["box"][0], obj["box"][2], obj))
        _, _, _, obj = min(dists)
        pending.remove(obj)
        dr, dc = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}[obj["direction"]]
        dist = _dist(obj["box"], obj["direction"], stopped, grid.shape)
        rows = [r for r, _ in obj["comp"]]
        cols = [c for _, c in obj["comp"]]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        body_box = obj["box"]
        br0, br1, bc0, bc1 = body_box
        for r in range(br0 + dr * dist, br1 + dr * dist + 1):
            for c in range(bc0 + dc * dist, bc1 + dc * dist + 1):
                out[r, c] = obj["color"]
                stopped[r, c] = True
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((24, 24), dtype=int)
    if random.choice([True, False]):
        r = random.randint(7, 16)
        c0 = random.randint(5, 10)
        grid[r, c0 : c0 + random.randint(5, 8)] = WALL
    else:
        c = random.randint(7, 16)
        r0 = random.randint(5, 10)
        grid[r0 : r0 + random.randint(5, 8), c] = WALL
    colors = random.sample([1, 2, 4, 5, 6, 7, 8, 9], random.randint(2, 4))
    occupied = grid != BG
    for color in colors:
        bh, bw = random.randint(2, 3), random.randint(2, 3)
        body = np.full((bh, bw), color, dtype=int)
        direction = random.choice(["U", "D", "L", "R"])
        if direction == "U":
            arr = np.zeros((bh + 1, bw), dtype=int)
            arr[1:, :] = body
            arr[0, bw // 2] = color
        elif direction == "D":
            arr = np.zeros((bh + 1, bw), dtype=int)
            arr[:-1, :] = body
            arr[-1, bw // 2] = color
        elif direction == "L":
            arr = np.zeros((bh, bw + 1), dtype=int)
            arr[:, 1:] = body
            arr[bh // 2, 0] = color
        else:
            arr = np.zeros((bh, bw + 1), dtype=int)
            arr[:, :-1] = body
            arr[bh // 2, -1] = color
        ah, aw = arr.shape
        for _ in range(300):
            r = random.randint(1, 23 - ah)
            c = random.randint(1, 23 - aw)
            rr0, rr1 = max(0, r - 1), min(24, r + ah + 1)
            cc0, cc1 = max(0, c - 1), min(24, c + aw + 1)
            if np.any(occupied[rr0:rr1, cc0:cc1]):
                continue
            grid[r : r + ah, c : c + aw] = np.where(arr != BG, arr, grid[r : r + ah, c : c + aw])
            occupied[rr0:rr1, cc0:cc1] = True
            break
    return grid
