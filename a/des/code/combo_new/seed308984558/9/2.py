import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
ORDER = [(0, 0), (0, 1), (1, 0), (1, 1)]


def _neighbors(r, c):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        yield r + dr, c + dc


def _distance_order(path_cells, start):
    q = deque([start])
    dist = {start: 0}
    while q:
        cur = q.popleft()
        for nxt in _neighbors(*cur):
            if nxt in path_cells and nxt not in dist:
                dist[nxt] = dist[cur] + 1
                q.append(nxt)
    return [p for p, _ in sorted(dist.items(), key=lambda item: item[1])]


def _component_passable(grid, start, blocked):
    h, w = grid.shape
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and (nr, nc) not in blocked and grid[nr, nc] != BLACK:
                seen.add((nr, nc))
                q.append((nr, nc))
    return seen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide pipe seeds into their station end and encode the count inside each station.

    Input structure:
    - Grey one-cell-wide pipes form non-branching orthogonal paths.
    - Each path ends at a 2x2 grey station block touching exactly one pipe endpoint.
    - Colored seed pixels lie on the pipe path; seeds on the same path share one color.

    Rule:
    - For each station, order the attached pipe cells from the station-adjacent end
      outward along the path.
    - Move all seeds of that path onto the first k ordered pipe cells nearest the
      station, preserving their shared color.
    - Recolor the first k cells of the station block in row-major order to the same
      color, leaving the remaining station cells grey.
    - All original seed positions elsewhere on the path revert to grey.

    Key insight:
    - The station defines the privileged path endpoint; the task combines a geometric
      slide toward that endpoint with a second-stage count display inside the station.
    """
    h, w = grid.shape
    out = grid.copy()
    handled = set()
    for r in range(h - 1):
        for c in range(w - 1):
            if not np.all(grid[r : r + 2, c : c + 2] == GREY):
                continue
            block = {(r + dr, c + dc) for dr in range(2) for dc in range(2)}
            adj = [p for cell in block for p in _neighbors(*cell) if 0 <= p[0] < h and 0 <= p[1] < w and p not in block and grid[p] == GREY]
            adj = list(dict.fromkeys(adj))
            if len(adj) != 1 or (r, c) in handled:
                continue
            handled.add((r, c))
            start = adj[0]
            path_cells = _component_passable(grid, start, block)
            order = _distance_order(path_cells, start)
            seed_cells = [p for p in order if grid[p] not in (BLACK, GREY)]
            color = int(grid[seed_cells[0]]) if seed_cells else GREY
            count = len(seed_cells)
            for p in order:
                out[p] = GREY
            for p in order[:count]:
                out[p] = color
            for idx, (dr, dc) in enumerate(ORDER):
                out[r + dr, c + dc] = color if idx < count else GREY
    return out


def generate_input() -> np.ndarray:
    while True:
        h = random.randint(18, 26)
        w = random.randint(18, 26)
        grid = np.zeros((h, w), dtype=int)
        ok = True
        stations = random.randint(2, 4)
        for color in random.sample(COLORS, stations):
            side = random.choice(["left", "right", "top", "bottom"])
            if side in ("left", "right"):
                r = random.randint(2, h - 4)
                c = 1 if side == "left" else w - 3
                grid[r : r + 2, c : c + 2] = GREY
                start = (r + random.randint(0, 1), c + 2) if side == "left" else (r + random.randint(0, 1), c - 1)
                direction = (0, 1) if side == "left" else (0, -1)
            else:
                r = 1 if side == "top" else h - 3
                c = random.randint(2, w - 4)
                grid[r : r + 2, c : c + 2] = GREY
                start = (r + 2, c + random.randint(0, 1)) if side == "top" else (r - 1, c + random.randint(0, 1))
                direction = (1, 0) if side == "top" else (-1, 0)
            path = [start]
            cur = start
            cur_dir = direction
            for _ in range(random.randint(5, 8)):
                nr, nc = cur[0] + cur_dir[0], cur[1] + cur_dir[1]
                if not (1 <= nr < h - 1 and 1 <= nc < w - 1) or grid[nr, nc] != BLACK or (nr, nc) in path:
                    break
                cur = (nr, nc)
                path.append(cur)
                if random.random() < 0.25:
                    cur_dir = (cur_dir[1], -cur_dir[0])
            if len(path) < 4:
                ok = False
                break
            for r0, c0 in path:
                grid[r0, c0] = GREY
            seeds = random.sample(path[1:], random.randint(1, min(4, len(path) - 1)))
            for r0, c0 in seeds:
                grid[r0, c0] = color
        if ok:
            return grid
