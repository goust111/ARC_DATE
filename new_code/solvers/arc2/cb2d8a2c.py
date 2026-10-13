import heapq
from collections import deque

import numpy as np

from arc_utils import cc4 as _cc4
from arc_utils import DIR4_URDL as _DIRS4

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Extend a green path from one grid edge to the opposite, routing around red bars at a distance dictated by their blue pixel count.

    Input structure:
    - bg = azure (8). Red/blue bars (color 2/1) act as obstacles.
    - A single green (3) pixel sits on a grid border — the path start.

    Rule:
    - Each bar's keep-out radius equals its number of blue pixels; the path must stay farther than that from every bar cell.
    - The green pixel extends inward toward the opposite edge, turning at right angles to navigate around forbidden zones.
    - After the path is drawn, all blue pixels become red.

    Key insight: Blue pixels on a bar encode its danger radius — more blue means a wider no-go zone the path must respect.
    """
    g = np.array(grid, copy=True)
    h, w = g.shape

    BG = 8
    GREEN = 3
    BLUE = 1
    RED = 2

    dirs = _DIRS4  # U, R, D, L

    # --- find start (green pixel on border) ---
    start_pts = list(zip(*np.where(g == GREEN)))
    if not start_pts:
        out = g.copy()
        out[out == BLUE] = RED
        return out

    start_pts = [(int(r), int(c)) for r, c in start_pts]
    border_pts = [p for p in start_pts if p[0] in (0, h - 1) or p[1] in (0, w - 1)]
    sr, sc = border_pts[0] if border_pts else start_pts[0]

    # Inward initial direction + opposite-edge target.
    if sr == 0:
        init_dir = 2  # down
        goal_kind, goal_val = "row", h - 1
    elif sr == h - 1:
        init_dir = 0  # up
        goal_kind, goal_val = "row", 0
    elif sc == 0:
        init_dir = 1  # right
        goal_kind, goal_val = "col", w - 1
    elif sc == w - 1:
        init_dir = 3  # left
        goal_kind, goal_val = "col", 0
    else:
        init_dir = 2
        goal_kind, goal_val = "row", h - 1

    def is_goal(r: int, c: int) -> bool:
        return (r == goal_val) if goal_kind == "row" else (c == goal_val)

    # --- find bar components (colors {1,2}) ---
    bar_mask = (g == BLUE) | (g == RED)
    bar_comps = _cc4(bar_mask)

    # --- build forbidden region: for each bar, all cells with Manhattan distance < (blue_count+1) ---
    forbidden = np.zeros((h, w), dtype=bool)
    for comp in bar_comps:
        blue_count = sum(1 for r, c in comp if g[r, c] == BLUE)
        min_dist = blue_count + 1
        radius = min_dist - 1  # forbid distance <= radius
        if radius < 0:
            continue

        dist = np.full((h, w), -1, dtype=np.int16)
        q = deque()
        for r, c in comp:
            dist[r, c] = 0
            q.append((r, c))

        while q:
            r, c = q.popleft()
            d = int(dist[r, c])
            if d > radius:
                continue
            forbidden[r, c] = True
            if d == radius:
                continue
            for dr, dc in dirs:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and dist[rr, cc] == -1:
                    dist[rr, cc] = d + 1
                    q.append((rr, cc))

    blocked = bar_mask | forbidden
    blocked[sr, sc] = False

    # --- Dijkstra: minimize (steps, turns), with stable ties via (r,c,dir) ---
    INF = (10**9, 10**9)
    best_cost: dict[tuple[int, int, int], tuple[int, int]] = {(sr, sc, init_dir): (0, 0)}
    prev: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    heap: list[tuple[tuple[int, int], int, int, int]] = [((0, 0), sr, sc, init_dir)]

    goal_state: tuple[int, int, int] | None = None
    while heap:
        (steps, turns), r, c, dprev = heapq.heappop(heap)
        if (steps, turns) != best_cost.get((r, c, dprev), INF):
            continue
        if is_goal(r, c) and (r, c) != (sr, sc):
            goal_state = (r, c, dprev)
            break
        for nd, (dr, dc) in enumerate(dirs):
            rr, cc = r + dr, c + dc
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            if blocked[rr, cc]:
                continue
            ncost = (steps + 1, turns + (0 if nd == dprev else 1))
            key = (rr, cc, nd)
            if ncost < best_cost.get(key, INF):
                best_cost[key] = ncost
                prev[key] = (r, c, dprev)
                heapq.heappush(heap, (ncost, rr, cc, nd))

    # --- paint path + convert blue -> red ---
    out = g.copy()
    if goal_state is not None:
        path: list[tuple[int, int]] = []
        st = goal_state
        while True:
            r, c, _ = st
            path.append((r, c))
            if (r, c) == (sr, sc):
                break
            st = prev[st]
        path.reverse()
        for r, c in path:
            if out[r, c] == BG:
                out[r, c] = GREEN

    out[out == BLUE] = RED
    return out
