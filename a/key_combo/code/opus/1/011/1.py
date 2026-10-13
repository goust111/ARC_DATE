import random
from collections import deque

import numpy as np


def _components(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 0 or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and not seen[nx, ny] and grid[nx, ny] != 0:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            out.append(comp)
    return out


def _symmetry_type(comp):
    rows = [p[0] for p in comp]
    cols = [p[1] for p in comp]
    r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
    pts = {(r - r0, c - c0) for r, c in comp}
    hh, ww = r1 - r0 + 1, c1 - c0 + 1
    if {(hh - 1 - r, c) for r, c in pts} == pts:
        return (0, 1)
    if {(r, ww - 1 - c) for r, c in pts} == pts:
        return (1, 0)
    return (1, 1)


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    out = grid.copy()
    comps = _components(grid)
    objects = []
    for comp in comps:
        if len(comp) == 1:
            out[comp[0]] = 0
            continue
        color = int(grid[comp[0]])
        rows = [p[0] for p in comp]
        cols = [p[1] for p in comp]
        objects.append(
            {
                "coords": set(comp),
                "color": color,
                "center": ((min(rows) + max(rows)) // 2, (min(cols) + max(cols)) // 2),
                "dir": _symmetry_type(comp),
            }
        )
    all_obj = set().union(*(o["coords"] for o in objects)) if objects else set()
    ray_owner = {}
    rays = []
    h, w = grid.shape
    for idx, obj in enumerate(objects):
        r, c = obj["center"]
        dr, dc = obj["dir"]
        path = []
        r += dr
        c += dc
        while 0 <= r < h and 0 <= c < w:
            p = (r, c)
            if p in all_obj and p not in obj["coords"]:
                break
            if p not in obj["coords"]:
                path.append(p)
                ray_owner.setdefault(p, set()).add(idx)
            r += dr
            c += dc
        rays.append(path)
    intersections = {p for p, owners in ray_owner.items() if len(owners) >= 2}
    hit_objects = set()
    for p in intersections:
        hit_objects.update(ray_owner[p])
    for path in rays:
        for p in path:
            out[p] = 5
    for p in intersections:
        out[p] = 4
    for idx in hit_objects:
        for p in objects[idx]["coords"]:
            out[p] = 8
    return out


def generate_input() -> np.ndarray:
    h = w = random.randint(17, 20)
    grid = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 2, 3, 6, 7], 5)
    dr = random.randint(2, 3)
    dc = random.randint(2, 3)
    hr = random.randint(dr + 4, min(dr + 6, h - 5))
    hc = dc + 1
    vc = dc + (hr - dr)
    vr = dr + 1
    shapes = [
        ((dr, dc), [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, 0)]),
        ((hr, hc), [(-1, 0), (0, 0), (1, 0), (0, 1)]),
        ((vr, vc), [(0, -1), (0, 0), (0, 1), (1, 0)]),
    ]
    extra_center = (random.randint(12, h - 3), random.randint(2, 5))
    shapes.append((extra_center, random.choice([
        [(-1, 0), (0, 0), (1, 0), (0, -1)],
        [(0, -1), (0, 0), (0, 1), (-1, 0)],
        [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, 0)],
    ])))
    if random.random() < 0.5:
        shapes.append(((random.randint(11, h - 3), random.randint(12, w - 3)), [(0, -1), (0, 0), (0, 1), (1, 0)]))
    for color, (center, coords) in zip(colors, shapes):
        cr, cc = center
        for rr, cc2 in coords:
            grid[cr + rr, cc + cc2] = color
    for _ in range(random.randint(0, 3)):
        for _attempt in range(100):
            r, c = random.randrange(h), random.randrange(w)
            if grid[r, c] != 0:
                continue
            if all(not (0 <= r + dr2 < h and 0 <= c + dc2 < w and grid[r + dr2, c + dc2] != 0)
                   for dr2, dc2 in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                grid[r, c] = random.choice([1, 2, 3, 6, 7])
                break
    return grid
