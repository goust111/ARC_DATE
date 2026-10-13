import random
from collections import defaultdict, deque

import numpy as np


BG = 0
GREY = 5
ORANGE = 7
MAROON = 9


DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _mirror_components(grid):
    h, w = grid.shape
    seen = set()
    mirrors = {}
    for r in range(h):
        for c in range(w):
            if grid[r, c] != GREY or (r, c) in seen:
                continue
            q = deque([(r, c)])
            seen.add((r, c))
            cells = []
            while q:
                x, y = q.popleft()
                cells.append((x, y))
                for dx, dy in DIRS:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and grid[nx, ny] == GREY and (nx, ny) not in seen:
                        seen.add((nx, ny))
                        q.append((nx, ny))
            rows = {x for x, _ in cells}
            cols = {y for _, y in cells}
            if len(rows) == 1:
                orient = "h"
                cells = sorted(cells, key=lambda p: p[1])
                endpoints = (cells[0], cells[-1])
            elif len(cols) == 1:
                orient = "v"
                cells = sorted(cells, key=lambda p: p[0])
                endpoints = (cells[0], cells[-1])
            else:
                orient = "x"
                endpoints = (cells[0], cells[-1])
            for cell in cells:
                mirrors[cell] = (orient, endpoints, cells)
    return mirrors


def _reflect(hit, direction, mirror_info):
    orient, endpoints, _ = mirror_info
    dr, dc = direction
    if orient == "h" and dr != 0:
        left, right = endpoints
        if abs(hit[1] - left[1]) <= abs(hit[1] - right[1]):
            return left, (0, -1)
        return right, (0, 1)
    if orient == "v" and dc != 0:
        top, bottom = endpoints
        if abs(hit[0] - top[0]) <= abs(hit[0] - bottom[0]):
            return top, (-1, 0)
        return bottom, (1, 0)
    return None


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    h, w = grid.shape
    out = grid.copy()
    mirrors = _mirror_components(grid)
    towers = [(r, c, int(grid[r, c])) for r in range(h) for c in range(w) if grid[r, c] not in (BG, GREY)]
    tower_cells = {(r, c) for r, c, _ in towers}

    active = []
    for r, c, color in towers:
        for d in DIRS:
            active.append({"pos": (r, c), "dir": d, "color": color, "seen": {((r, c), d, color)}})

    max_steps = h * w * 8
    for _ in range(max_steps):
        if not active:
            break
        proposals = defaultdict(list)
        next_active = []
        for ray in active:
            r, c = ray["pos"]
            dr, dc = ray["dir"]
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in tower_cells:
                continue
            if grid[nr, nc] == GREY:
                reflected = _reflect((nr, nc), (dr, dc), mirrors.get((nr, nc)))
                if reflected is None:
                    continue
                start, ndir = reflected
                state = (start, ndir, ORANGE)
                if state in ray["seen"]:
                    continue
                seen = set(ray["seen"])
                seen.add(state)
                next_active.append({"pos": start, "dir": ndir, "color": ORANGE, "seen": seen})
                continue
            if out[nr, nc] != BG:
                continue
            proposals[(nr, nc)].append(ray)

        for cell, rays in proposals.items():
            if len(rays) > 1:
                out[cell] = MAROON
                continue
            ray = rays[0]
            out[cell] = ray["color"]
            state = (cell, ray["dir"], ray["color"])
            if state in ray["seen"]:
                continue
            seen = set(ray["seen"])
            seen.add(state)
            next_active.append({"pos": cell, "dir": ray["dir"], "color": ray["color"], "seen": seen})
        active = next_active
    return out


def _can_place(grid, cells):
    h, w = grid.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if grid[r, c] != BG:
            return False
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == GREY:
                return False
    return True


def generate_input() -> np.ndarray:
    h = random.randint(13, 22)
    w = random.randint(13, 22)
    grid = np.zeros((h, w), dtype=int)

    mirror_count = random.randint(2, 5)
    for _ in range(mirror_count):
        for _attempt in range(100):
            length = random.randint(2, 4)
            horizontal = random.choice([True, False])
            if horizontal:
                r = random.randint(2, h - 3)
                c = random.randint(2, w - length - 2)
                cells = [(r, c + i) for i in range(length)]
            else:
                r = random.randint(2, h - length - 2)
                c = random.randint(2, w - 3)
                cells = [(r + i, c) for i in range(length)]
            if _can_place(grid, cells):
                for cell in cells:
                    grid[cell] = GREY
                break

    colors = [1, 2, 3, 4, 6, 8]
    random.shuffle(colors)
    tower_count = random.randint(2, min(4, len(colors)))
    placed = 0
    for color in colors[:tower_count]:
        for _attempt in range(200):
            r = random.randint(1, h - 2)
            c = random.randint(1, w - 2)
            if grid[r, c] != BG:
                continue
            if any(grid[r + dr, c + dc] == GREY for dr, dc in DIRS):
                continue
            grid[r, c] = color
            placed += 1
            break

    if placed < 2:
        return generate_input()
    return grid
