import random
import numpy as np


BG = 0
GREY = 5


def _components(mask: np.ndarray):
    seen = set()
    out = []
    H, W = mask.shape
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or (r, c) in seen:
                continue
            stack = [(r, c)]
            seen.add((r, c))
            comp = []
            while stack:
                x, y = stack.pop()
                comp.append((x, y))
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < H and 0 <= ny < W and mask[nx, ny] and (nx, ny) not in seen:
                        seen.add((nx, ny))
                        stack.append((nx, ny))
            out.append(comp)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey path segment-by-segment while traversing from its marked start endpoint.

    Input structure:
    - Grey 4-connected paths lie on a black background.
    - Exactly one endpoint of each path touches a colored start marker.
    - Extra colored single pixels beside the path act as dye stations.

    Rule:
    - Start at the endpoint next to the start marker and traverse the path in order.
    - Paint path pixels using the current color, initially the start-marker color.
    - When a path pixel is adjacent to a dye station, switch the current color after
      that pixel so the next segment uses the station color.
    - Keep all markers and stations unchanged.

    Key insight:
    - The path itself is neutral; the output depends on traversal direction plus the
      ordered sequence of side-adjacent color stations encountered along that walk.
    """
    out = grid.copy()
    path_mask = grid == GREY
    stations = {(r, c): int(grid[r, c]) for r, c in np.argwhere((grid != BG) & (grid != GREY))}
    for comp in _components(path_mask):
        comp_set = set(comp)
        neigh = {}
        ends = []
        for p in comp:
            ns = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                q = (p[0] + dx, p[1] + dy)
                if q in comp_set:
                    ns.append(q)
            neigh[p] = ns
            if len(ns) == 1:
                ends.append(p)
        start = None
        color = None
        for p in ends:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                q = (p[0] + dx, p[1] + dy)
                if q in stations:
                    start = p
                    color = stations[q]
        if start is None:
            continue
        prev = None
        cur = start
        walked = set()
        while True:
            if cur in walked:
                break
            walked.add(cur)
            out[cur] = color
            next_color = None
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                q = (cur[0] + dx, cur[1] + dy)
                if q in stations:
                    next_color = stations[q]
            nxts = [p for p in neigh[cur] if p != prev]
            if not nxts:
                break
            prev, cur = cur, nxts[0]
            if next_color is not None:
                color = next_color
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((24, 24), dtype=int)
    colors = [1, 2, 3, 4, 6, 7, 8, 9]
    for _ in range(random.randint(2, 4)):
        r = random.randint(2, 20)
        c = random.randint(2, 20)
        length = random.randint(8, 14)
        path = [(r, c)]
        direction = random.choice([(0, 1), (1, 0)])
        for _ in range(length - 1):
            if random.random() < 0.25:
                direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
            nr = min(max(1, path[-1][0] + direction[0]), 22)
            nc = min(max(1, path[-1][1] + direction[1]), 22)
            if (nr, nc) == path[-1] or (nr, nc) in path:
                continue
            path.append((nr, nc))
        if any(grid[p] != BG for p in path):
            continue
        for p in path:
            grid[p] = GREY
        start_color = random.choice(colors)
        sr, sc = path[0]
        marker_spots = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            q = (sr + dx, sc + dy)
            if 0 <= q[0] < 24 and 0 <= q[1] < 24 and grid[q] == BG:
                marker_spots.append(q)
        if not marker_spots:
            for p in path:
                grid[p] = BG
            continue
        mr, mc = random.choice(marker_spots)
        grid[mr, mc] = start_color
        inner = path[1:-1]
        if inner:
            for p in random.sample(inner, random.randint(1, min(3, len(inner)))):
                for dx, dy in random.sample([(-1, 0), (1, 0), (0, -1), (0, 1)], 4):
                    q = (p[0] + dx, p[1] + dy)
                    if 0 <= q[0] < 24 and 0 <= q[1] < 24 and grid[q] == BG:
                        grid[q] = random.choice(colors)
                        break
    return grid
