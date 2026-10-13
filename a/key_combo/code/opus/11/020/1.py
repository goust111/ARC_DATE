from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

MAX_GENERATION_ATTEMPTS = 600


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _initial_direction(h: int, w: int, r: int, c: int) -> tuple[int, int]:
    if r == 0:
        dr = 1
    elif r == h - 1:
        dr = -1
    else:
        dr = 1 if 2 * r < h - 1 else -1

    if c == 0:
        dc = 1
    elif c == w - 1:
        dc = -1
    else:
        dc = 1 if 2 * c < w - 1 else -1
    return dr, dc


def _find_start(grid: np.ndarray) -> tuple[int, int]:
    starts = np.argwhere(grid == BLUE)
    if starts.shape[0] != 1:
        raise ValueError("expected exactly one blue start pixel")
    return int(starts[0, 0]), int(starts[0, 1])


def _simulate_path(grid: np.ndarray) -> tuple[list[tuple[int, int]], int, int]:
    h, w = grid.shape
    r, c = _find_start(grid)
    dr, dc = _initial_direction(h, w, r, c)

    path: list[tuple[int, int]] = []
    wall_hits = 0
    boundary_hits = 0
    seen_states: set[tuple[int, int, int, int]] = set()

    while 0 <= r < h and 0 <= c < w:
        path.append((r, c))
        state = (r, c, dr, dc)
        if state in seen_states:
            raise ValueError("path entered a loop")
        seen_states.add(state)

        nr = r + dr
        nc = c + dc
        if nr < 0 or nr >= h:
            break

        if nc < 0 or nc >= w:
            boundary_hits += 1
            dc = -dc
            nc = c + dc

        if grid[nr, nc] == RED:
            wall_hits += 1
            r, c = nr, nc
            dc = -dc
        else:
            r, c = nr, nc

        if len(path) > h * w * 4:
            raise ValueError("path too long")

    return path, wall_hits, boundary_hits


def _label_regions(
    grid: np.ndarray, path: list[tuple[int, int]]
) -> tuple[np.ndarray, int, dict[int, set[int]], np.ndarray]:
    h, w = grid.shape
    separator = np.zeros((h, w), dtype=bool)
    for r, c in path:
        separator[r, c] = True
    separator |= grid == RED

    labels = np.zeros((h, w), dtype=int)
    region_id = 0
    for r in range(h):
        for c in range(w):
            if grid[r, c] != BLACK or labels[r, c] != 0:
                continue
            region_id += 1
            queue = deque([(r, c)])
            labels[r, c] = region_id
            while queue:
                rr, cc = queue.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if (
                        _in_bounds((h, w), nr, nc)
                        and grid[nr, nc] == BLACK
                        and labels[nr, nc] == 0
                    ):
                        labels[nr, nc] = region_id
                        queue.append((nr, nc))

    graph: dict[int, set[int]] = defaultdict(set)
    for r in range(h):
        for c in range(w):
            if not separator[r, c]:
                continue
            adjacent_regions: list[int] = []
            seen: set[int] = set()
            for nr, nc in _neighbors4(r, c):
                if not _in_bounds((h, w), nr, nc):
                    continue
                rid = int(labels[nr, nc])
                if rid > 0 and rid not in seen:
                    seen.add(rid)
                    adjacent_regions.append(rid)
            for i, a in enumerate(adjacent_regions):
                for b in adjacent_regions[i + 1 :]:
                    graph[a].add(b)
                    graph[b].add(a)

    return labels, region_id, graph, separator


def _lower_left_region(labels: np.ndarray) -> int:
    h, w = labels.shape
    if labels[h - 1, 0] > 0:
        return int(labels[h - 1, 0])
    for r in range(h - 1, -1, -1):
        for c in range(w):
            if labels[r, c] > 0:
                return int(labels[r, c])
    return 0


def _region_parity(
    labels: np.ndarray, n_regions: int, graph: dict[int, set[int]]
) -> dict[int, int] | None:
    parity: dict[int, int] = {}

    def bfs(seed: int, seed_color: int) -> bool:
        parity[seed] = seed_color
        queue = deque([seed])
        while queue:
            cur = queue.popleft()
            for nxt in graph.get(cur, ()):
                nxt_color = 1 - parity[cur]
                if nxt in parity:
                    if parity[nxt] != nxt_color:
                        return False
                else:
                    parity[nxt] = nxt_color
                    queue.append(nxt)
        return True

    start = _lower_left_region(labels)
    if start > 0 and not bfs(start, 0):
        return None

    for rid in range(1, n_regions + 1):
        if rid not in parity and not bfs(rid, 0):
            return None
    return parity


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace a bouncing diagonal path, use it together with the red wall as separators, and alternately color the split background regions.

    Input structure:
    - The grid is mostly black background with one unique blue border pixel and one red 4-connected one-cell-thick wall polyline.
    - The blue pixel sits on the outer border and is not 4-adjacent to the red wall.
    - The red wall may be straight or bent, and the blue path can bounce on both side borders and red wall cells.

    Rule:
    - Launch the blue pixel diagonally toward the grid interior.
    - The vertical direction stays fixed until the path exits through the top or bottom; the horizontal direction flips when the next step would leave the left/right border, and also flips after stepping onto a red wall cell.
    - Treat every visited blue-path cell together with every red wall cell as separator lines, split the remaining black cells into 4-connected regions, then color the lower-left region grey and alternate grey/green across separator adjacencies.
    - Keep the red wall red, keep non-red path cells blue, and leave only the recolored regions changed.

    Key insight:
    - The final coloring cannot be decided from the input wall alone: the blue separator is dynamic, so you must simulate the whole bounce path first and only then build the region adjacency graph for alternating coloring.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("grid must be 2D")

    path, _wall_hits, _boundary_hits = _simulate_path(g)
    labels, n_regions, graph, _separator = _label_regions(g, path)
    parity = _region_parity(labels, n_regions, graph)
    if parity is None:
        raise ValueError("region graph is not bipartite")

    out = g.copy()
    out[g == BLACK] = BLACK
    for rid, color_id in parity.items():
        out[labels == rid] = GREY if color_id == 0 else GREEN

    for r, c in path:
        if g[r, c] != RED:
            out[r, c] = BLUE
    out[g == RED] = RED
    return out


def _border_start(h: int, w: int) -> tuple[int, int]:
    side = random.choice(["top", "bottom", "left", "right"])
    if side == "top":
        return 0, random.randint(1, w - 2)
    if side == "bottom":
        return h - 1, random.randint(1, w - 1)
    if side == "left":
        return random.randint(0, h - 2), 0
    return random.randint(0, h - 1), w - 1


def _straight_wall_candidates(
    h: int, w: int, pivot: tuple[int, int], pathset: set[tuple[int, int]]
):
    r, c = pivot
    axes = ["h", "v"]
    random.shuffle(axes)
    for axis in axes:
        directions = [(0, 1), (0, -1)] if axis == "h" else [(1, 0), (-1, 0)]
        arms: list[list[tuple[int, int]]] = []
        for dr, dc in directions:
            arm: list[tuple[int, int]] = []
            rr, cc = r, c
            while True:
                nr, nc = rr + dr, cc + dc
                if not _in_bounds((h, w), nr, nc) or (nr, nc) in pathset:
                    break
                arm.append((nr, nc))
                rr, cc = nr, nc
            arms.append(arm)

        total = 1 + len(arms[0]) + len(arms[1])
        if total < 5:
            continue

        for length in range(min(12, total), 4, -1):
            lo = max(0, length - 1 - len(arms[1]))
            hi = min(len(arms[0]), length - 1)
            if lo > hi:
                continue
            for _ in range(4):
                take_a = random.randint(lo, hi)
                take_b = length - 1 - take_a
                yield arms[1][:take_b][::-1] + [(r, c)] + arms[0][:take_a]


def _bent_wall_candidates(
    h: int, w: int, pivot: tuple[int, int], pathset: set[tuple[int, int]]
):
    r, c = pivot
    options = [
        ((0, 1), (1, 0)),
        ((0, 1), (-1, 0)),
        ((0, -1), (1, 0)),
        ((0, -1), (-1, 0)),
        ((1, 0), (0, 1)),
        ((1, 0), (0, -1)),
        ((-1, 0), (0, 1)),
        ((-1, 0), (0, -1)),
    ]
    random.shuffle(options)

    for (dr1, dc1), (dr2, dc2) in options:
        branch: list[tuple[int, int]] = []
        rr, cc = r, c
        while _in_bounds((h, w), rr + dr1, cc + dc1) and (rr + dr1, cc + dc1) not in pathset:
            rr += dr1
            cc += dc1
            branch.append((rr, cc))
        if len(branch) < 2:
            continue

        mids = list(range(1, len(branch) + 1))
        random.shuffle(mids)
        for mid in mids[:4]:
            corner = branch[mid - 1]
            tail: list[tuple[int, int]] = []
            rr, cc = corner
            while (
                _in_bounds((h, w), rr + dr2, cc + dc2)
                and (rr + dr2, cc + dc2) not in pathset
                and (rr + dr2, cc + dc2) != (r, c)
            ):
                rr += dr2
                cc += dc2
                tail.append((rr, cc))

            total = 1 + mid + len(tail)
            if total < 5:
                continue

            for length in range(min(12, total), 4, -1):
                tail_len = min(len(tail), length - 1 - mid)
                first_len = length - 1 - tail_len
                if first_len > mid:
                    continue
                yield [(r, c)] + branch[:first_len] + tail[:tail_len]


def _build_wall(h: int, w: int, base_path: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    pathset = set(base_path)
    pivots = [
        p
        for p in base_path[2:-2]
        if 1 <= p[0] < h - 1 and 1 <= p[1] < w - 1 and p != (h - 1, 0)
    ]
    random.shuffle(pivots)

    for pivot in pivots[:20]:
        builders = [_straight_wall_candidates, _bent_wall_candidates]
        random.shuffle(builders)
        for builder in builders:
            for wall in builder(h, w, pivot, pathset):
                if len(wall) == len(set(wall)):
                    return wall
    return None


def _valid_input(grid: np.ndarray) -> bool:
    try:
        path, wall_hits, _boundary_hits = _simulate_path(grid)
    except ValueError:
        return False

    h, w = grid.shape
    if wall_hits < 1:
        return False
    if (h - 1, 0) in path:
        return False

    labels, n_regions, graph, separator = _label_regions(grid, path)
    if n_regions < 2 or labels[h - 1, 0] == 0:
        return False

    sizes = [int(np.count_nonzero(labels == rid)) for rid in range(1, n_regions + 1)]
    if min(sizes) < 2:
        return False
    if int(np.count_nonzero(separator)) > (h * w) // 2:
        return False

    parity = _region_parity(labels, n_regions, graph)
    if parity is None or len(parity) != n_regions:
        return False

    try:
        out = main(grid)
    except ValueError:
        return False
    colors = set(map(int, np.unique(out)))
    return GREY in colors and GREEN in colors


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(11, 15)
    w = random.randint(11, 15)
    start = _border_start(h, w)
    if start == (h - 1, 0):
        return None

    grid = np.zeros((h, w), dtype=int)
    grid[start] = BLUE

    try:
        base_path, _wall_hits, _boundary_hits = _simulate_path(grid)
    except ValueError:
        return None
    if len(base_path) < 8:
        return None

    wall = _build_wall(h, w, base_path)
    if wall is None:
        return None
    if any(abs(start[0] - r) + abs(start[1] - c) == 1 for r, c in wall):
        return None

    for r, c in wall:
        if grid[r, c] != BLACK:
            return None
        grid[r, c] = RED

    return grid if _valid_input(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
