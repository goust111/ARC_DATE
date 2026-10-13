import random
from collections import deque
import numpy as np


def _orange_components(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for sr, sc in zip(*np.where(grid == 7)):
        if seen[sr, sc]:
            continue
        q = deque([(int(sr), int(sc))])
        seen[sr, sc] = True
        cells = []
        while q:
            r, c = q.popleft()
            cells.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and grid[rr, cc] == 7 and not seen[rr, cc]:
                        seen[rr, cc] = True
                        q.append((rr, cc))
        comps.append(cells)
    return comps


def _divider(grid):
    h, w = grid.shape
    for r in range(h):
        if np.all(grid[r, :] == 5):
            return ("h", r)
    for c in range(w):
        if np.all(grid[:, c] == 5):
            return ("v", c)
    raise ValueError("missing divider")


def _reflect_dir(r, c, dr, dc, h, w):
    if dc != 0:
        return (-1, 0) if r >= h - 1 - r else (1, 0)
    return (0, -1) if c >= w - 1 - c else (0, 1)


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    h, w = grid.shape
    div_axis, div_pos = _divider(grid)
    markers = [(int(r), int(c), int(grid[r, c])) for r, c in zip(*np.where((grid != 0) & (grid != 5) & (grid != 7)))]
    comps = _orange_components(grid)
    sprites = []
    for mr, mc, color in markers:
        near = None
        best = 99
        for comp in comps:
            dist = min(abs(mr - r) + abs(mc - c) for r, c in comp)
            if dist < best:
                best, near = dist, comp
        mean_r = sum(r for r, _ in near) / len(near) - mr
        mean_c = sum(c for _, c in near) / len(near) - mc
        if abs(mean_r) >= abs(mean_c):
            dr, dc = (-1 if mean_r > 0 else 1), 0
        else:
            dr, dc = 0, (-1 if mean_c > 0 else 1)
        distance = abs((mr if div_axis == "h" else mc) - div_pos)
        sprites.append((distance, mr, mc, color, dr, dc))

    out = grid.copy()
    for _, mr, mc, color, dr, dc in sorted(sprites):
        r, c = mr, mc
        step = 0
        while True:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                break
            if grid[nr, nc] == 5:
                dr, dc = _reflect_dir(r, c, dr, dc, h, w)
                continue
            r, c = nr, nc
            step += 1
            if step % 2 == 1:
                out[r, c] = color
    out[grid == 5] = 5
    out[grid == 7] = 7
    for r, c, color in markers:
        out[r, c] = color
    return out


def _body_for_dir(ray):
    base = [(1, -1), (1, 0), (1, 1), (2, -1), (2, 0), (2, 1), (3, 0)]
    if ray == (-1, 0):
        return base
    if ray == (1, 0):
        return [(-r, c) for r, c in base]
    if ray == (0, -1):
        return [(c, r) for r, c in base]
    return [(c, -r) for r, c in base]


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(1000):
        h = w = 18
        grid = np.zeros((h, w), dtype=int)
        vertical = rng.choice([True, False])
        div = rng.randint(7, 10)
        if vertical:
            grid[:, div] = 5
            marker_pool = [(rng.randint(3, 14), rng.choice([2, 4, 6, 13])) for _ in range(4)]
            rays = [rng.choice([(0, 1), (0, -1), (-1, 0), (1, 0)]) for _ in marker_pool]
        else:
            grid[div, :] = 5
            marker_pool = [(rng.choice([2, 4, 13, 15]), rng.randint(3, 14)) for _ in range(4)]
            rays = [rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)]) for _ in marker_pool]
        colors = [1, 2, 3, 4]
        ok = True
        for (mr, mc), ray, color in zip(marker_pool, rays, colors):
            if grid[mr, mc] != 0:
                ok = False
                break
            body = _body_for_dir(ray)
            cells = [(mr + dr, mc + dc) for dr, dc in body]
            if any(not (0 <= r < h and 0 <= c < w) or grid[r, c] != 0 for r, c in cells):
                ok = False
                break
            grid[mr, mc] = color
            for r, c in cells:
                grid[r, c] = 7
        if ok:
            return grid
    return generate_input()
