import random
from collections import defaultdict, deque

import numpy as np


DIRS = {(-1, -1), (-1, 1), (1, -1), (1, 1)}
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(grid, color):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != color or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dx, dy in DIR4:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and not seen[nx, ny] and grid[nx, ny] == color:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(comp)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """Extend diagonal rays from marked 2x2 squares with distance-priority conflicts.

    Input structure: Distinct colored 2x2 squares each have one gray marker just
    outside one diagonal corner.
    Rule: The marker chooses the square corner and the outward diagonal direction.
    Rays grow synchronously, stop before square bodies or occupied older ray
    cells, and same-step conflicts are won by the ray with smaller distance and
    then by the source square's top-left coordinate. Gray markers are removed.
    Key insight: The marker is a direction handle; ray intersections are resolved
    geometrically rather than by color or drawing order.
    """
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    squares = []
    body_cells = set()
    for color in [int(x) for x in np.unique(grid) if x not in (0, 5)]:
        for comp in _components(grid, color):
            if len(comp) != 4:
                continue
            rs = [r for r, _ in comp]
            cs = [c for _, c in comp]
            if max(rs) - min(rs) == 1 and max(cs) - min(cs) == 1:
                box = (min(rs), min(cs), color)
                squares.append(box)
                body_cells.update(comp)

    rays = []
    gray = set(zip(*np.where(grid == 5)))
    for r, c, color in squares:
        options = [
            ((r - 1, c - 1), (r, c), (-1, -1)),
            ((r - 1, c + 2), (r, c + 1), (-1, 1)),
            ((r + 2, c - 1), (r + 1, c), (1, -1)),
            ((r + 2, c + 2), (r + 1, c + 1), (1, 1)),
        ]
        for marker, corner, direction in options:
            if marker in gray:
                rays.append({"pos": corner, "corner": corner, "dir": direction,
                             "color": color, "source": (r, c), "alive": True})
                break

    out = grid.copy()
    out[out == 5] = 0
    occupied = set(body_cells)
    while any(ray["alive"] for ray in rays):
        arrivals = defaultdict(list)
        for idx, ray in enumerate(rays):
            if not ray["alive"]:
                continue
            nr = ray["pos"][0] + ray["dir"][0]
            nc = ray["pos"][1] + ray["dir"][1]
            if not (0 <= nr < h and 0 <= nc < w) or (nr, nc) in body_cells:
                ray["alive"] = False
            elif (nr, nc) in occupied:
                ray["alive"] = False
            else:
                arrivals[(nr, nc)].append(idx)
        if not arrivals:
            break
        winners = {}
        for cell, idxs in arrivals.items():
            def key(idx):
                ray = rays[idx]
                dist = max(abs(cell[0] - ray["corner"][0]), abs(cell[1] - ray["corner"][1]))
                return (dist, ray["source"][0], ray["source"][1])
            winner = min(idxs, key=key)
            winners[cell] = winner
            for idx in idxs:
                if idx != winner:
                    rays[idx]["alive"] = False
        for cell, idx in winners.items():
            ray = rays[idx]
            out[cell] = ray["color"]
            occupied.add(cell)
            ray["pos"] = cell
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((15, 15), dtype=int)
    specs = [
        (2, 2, 1, (1, 1)),
        (2, 11, 2, (1, -1)),
        (11, 2, 3, (-1, 1)),
        (11, 11, 4, (-1, -1)),
    ]
    if random.random() < 0.4:
        specs.append((6, 1, 6, (1, 1)))
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], len(specs))
    for (r, c, _, direction), color in zip(specs, colors):
        grid[r:r + 2, c:c + 2] = color
        dr, dc = direction
        marker_r = r + (2 if dr == 1 else -1)
        marker_c = c + (2 if dc == 1 else -1)
        if 0 <= marker_r < 15 and 0 <= marker_c < 15:
            grid[marker_r, marker_c] = 5
    return grid

