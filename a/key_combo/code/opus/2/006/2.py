import random
from collections import deque

import numpy as np


def _rg_components(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) not in (2, 3):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                y, x = q.popleft()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and int(grid[ny, nx]) in (2, 3):
                        seen[ny, nx] = True
                        q.append((ny, nx))
            comps.append(cells)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    h, w = grid.shape
    dominoes = []
    for cells in _rg_components(grid):
        if len(cells) != 2:
            continue
        vals = [int(grid[p]) for p in cells]
        if sorted(vals) != [2, 3]:
            continue
        red = cells[vals.index(2)]
        green = cells[vals.index(3)]
        if abs(red[0] - green[0]) + abs(red[1] - green[1]) == 1:
            dominoes.append((red, green))

    for red, _green in dominoes:
        out[red] = 0
    for red, green in dominoes:
        dy = red[0] - green[0]
        dx = red[1] - green[1]
        y, x = red
        while 0 <= y < h and 0 <= x < w and int(out[y, x]) == 0:
            out[y, x] = 8
            ny, nx = y + dy, x + dx
            if not (0 <= ny < h and 0 <= nx < w) or int(out[ny, nx]) != 0:
                break
            y, x = ny, nx
    return out


def _clear_for_domino(grid, red, green, direction):
    h, w = grid.shape
    for p in (red, green):
        y, x = p
        if not (0 <= y < h and 0 <= x < w) or grid[y, x] != 0:
            return False
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if (ny, nx) not in (red, green) and 0 <= ny < h and 0 <= nx < w and grid[ny, nx] in (2, 3):
                return False
    dy, dx = direction
    y, x = red
    for _ in range(3):
        if not (0 <= y < h and 0 <= x < w) or grid[y, x] != 0:
            return False
        y += dy
        x += dx
    return True


def generate_input() -> np.ndarray:
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for _ in range(800):
        h = random.randint(11, 16)
        w = random.randint(12, 18)
        grid = np.zeros((h, w), dtype=int)
        used_rays = set()
        count = random.randint(2, 4)
        placed = 0
        for _j in range(150):
            if placed == count:
                break
            dy, dx = random.choice(dirs)
            red = (random.randint(1, h - 2), random.randint(1, w - 2))
            green = (red[0] - dy, red[1] - dx)
            if not _clear_for_domino(grid, red, green, (dy, dx)):
                continue
            ray = []
            y, x = red
            while 0 <= y < h and 0 <= x < w and grid[y, x] == 0:
                ray.append((y, x))
                y += dy
                x += dx
            if len(ray) < 3 or any(p in used_rays for p in ray):
                continue
            grid[red] = 2
            grid[green] = 3
            used_rays.update(ray)
            if 0 <= y < h and 0 <= x < w and grid[y, x] == 0 and random.random() < 0.8:
                grid[y, x] = 5
            placed += 1
        if placed != count:
            continue
        for _j in range(random.randint(3, 7)):
            for _k in range(100):
                y, x = random.randrange(h), random.randrange(w)
                if grid[y, x] != 0 or (y, x) in used_rays:
                    continue
                color = random.choice([2, 3, 5])
                if color in (2, 3):
                    bad = False
                    for dy, dx in dirs:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and grid[ny, nx] in (2, 3):
                            bad = True
                    if bad:
                        continue
                grid[y, x] = color
                break
        if len(_rg_components(grid)) >= count:
            return grid
    return generate_input()
