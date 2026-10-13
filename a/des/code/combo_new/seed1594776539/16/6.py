import random
from collections import defaultdict

import numpy as np


BG = 0
WALL = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _region_from_marker(maze: np.ndarray, start: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = maze.shape
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if maze[nr, nc] == WALL or (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            stack.append((nr, nc))
    return seen


def _remove_region_boundary(maze: np.ndarray, start: tuple[int, int]) -> int:
    region = _region_from_marker(maze, start)
    boundary: set[tuple[int, int]] = set()
    h, w = maze.shape
    for r, c in region:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and maze[nr, nc] == WALL:
                boundary.add((nr, nc))
    for r, c in boundary:
        maze[r, c] = BG
    return len(boundary)


def _solve_with_order(grid: np.ndarray, order: list[int]) -> np.ndarray:
    g = np.asarray(grid, dtype=int).copy()
    maze = g[:, 1:]
    markers = {}
    for r, c in np.argwhere((maze != BG) & (maze != WALL)):
        color = int(maze[r, c])
        assert color not in markers, "each interior marker color must appear exactly once"
        markers[color] = (int(r), int(c))

    assert order, "expected at least one order color"
    assert len(order) == len(set(order)), "left-column order colors must be unique"
    assert set(order) == set(markers), "left-column colors must match interior marker colors"

    for color in order:
        _remove_region_boundary(maze, markers[color])

    g[:, 0] = BG
    return g


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove each marked region's surrounding wall ring in the color order encoded on the left edge.

    Input structure:
    - Black cells are the maze corridors, while grey cells form a connected 1-pixel wall network in the main field.
    - The leftmost column is a legend column containing 2-4 single-pixel colored order markers from top to bottom.
    - Matching single-pixel interior markers of those colors lie inside distinct corridor regions of the maze.

    Rule:
    - Read the non-black, non-grey cells in the leftmost column from top to bottom as the execution order.
    - Ignore that legend column when reasoning about the maze. For each color in order, find the matching interior marker and flood-fill its current non-grey region through black cells and already-present colored markers.
    - Turn every grey wall pixel that is 4-adjacent to that current region into black, then continue with the updated maze; keep the interior markers unchanged and erase the left legend column to black.

    Key insight:
    - The border legend does not name destinations directly; it chooses the operation order, and earlier wall removals enlarge later markers' regions so the same local action can remove a much larger outer boundary.
    """
    g = np.asarray(grid, dtype=int)
    order = [int(v) for v in g[:, 0] if int(v) not in (BG, WALL)]
    return _solve_with_order(g, order)


class _DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True


def _room_id(rr: int, cc: int, room_cols: int) -> int:
    return rr * room_cols + cc


def _room_slice(rr: int, cc: int, cell_h: int, cell_w: int) -> tuple[slice, slice]:
    r0 = rr * (cell_h + 1) + 1
    c0 = cc * (cell_w + 1) + 1
    return slice(r0, r0 + cell_h), slice(c0, c0 + cell_w)


def _build_base_maze(room_rows: int, room_cols: int, cell_h: int, cell_w: int) -> np.ndarray:
    h = room_rows * (cell_h + 1) + 1
    w = room_cols * (cell_w + 1) + 1
    maze = np.full((h, w), WALL, dtype=int)
    for rr in range(room_rows):
        for cc in range(room_cols):
            rs, cs = _room_slice(rr, cc, cell_h, cell_w)
            maze[rs, cs] = BG
    return maze


def _open_door(maze: np.ndarray, kind: str, rr: int, cc: int, cell_h: int, cell_w: int, rng: random.Random):
    if kind == "v":
        col = (cc + 1) * (cell_w + 1)
        row = rr * (cell_h + 1) + 1 + rng.randrange(cell_h)
        maze[row, col] = BG
        return
    row = (rr + 1) * (cell_h + 1)
    col = cc * (cell_w + 1) + 1 + rng.randrange(cell_w)
    maze[row, col] = BG


def _connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return True
    start = tuple(map(int, pts[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _room_groups(dsu: _DSU, room_rows: int, room_cols: int) -> dict[int, list[tuple[int, int]]]:
    groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for rr in range(room_rows):
        for cc in range(room_cols):
            groups[dsu.find(_room_id(rr, cc, room_cols))].append((rr, cc))
    return dict(groups)


def _region_graph(dsu: _DSU, room_rows: int, room_cols: int) -> dict[int, set[int]]:
    graph: dict[int, set[int]] = {root: set() for root in _room_groups(dsu, room_rows, room_cols)}
    for rr in range(room_rows):
        for cc in range(room_cols):
            a = dsu.find(_room_id(rr, cc, room_cols))
            if cc + 1 < room_cols:
                b = dsu.find(_room_id(rr, cc + 1, room_cols))
                if a != b:
                    graph[a].add(b)
                    graph[b].add(a)
            if rr + 1 < room_rows:
                b = dsu.find(_room_id(rr + 1, cc, room_cols))
                if a != b:
                    graph[a].add(b)
                    graph[b].add(a)
    return graph


def _find_path(graph: dict[int, set[int]], length: int, rng: random.Random) -> list[int] | None:
    starts = list(graph)
    rng.shuffle(starts)

    def dfs(node: int, path: list[int], used: set[int]) -> list[int] | None:
        if len(path) == length:
            return path[:]
        nxts = list(graph[node] - used)
        rng.shuffle(nxts)
        for nxt in nxts:
            used.add(nxt)
            path.append(nxt)
            found = dfs(nxt, path, used)
            if found is not None:
                return found
            path.pop()
            used.remove(nxt)
        return None

    for start in starts:
        found = dfs(start, [start], {start})
        if found is not None:
            return found
    return None


def _legend_rows(height: int, count: int, rng: random.Random) -> list[int]:
    rows: list[int] = []
    for _ in range(100):
        row = rng.randrange(height)
        if all(abs(row - other) >= 2 for other in rows):
            rows.append(row)
            if len(rows) == count:
                return sorted(rows)
    rows = rng.sample(range(height), count)
    rows.sort()
    return rows


def _place_marker(maze: np.ndarray, room: tuple[int, int], color: int, cell_h: int, cell_w: int, rng: random.Random):
    rr, cc = room
    rs, cs = _room_slice(rr, cc, cell_h, cell_w)
    r = rs.start + rng.randrange(cell_h)
    c = cs.start + rng.randrange(cell_w)
    maze[r, c] = color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    room_rows = rng.randint(3, 5)
    room_cols = rng.randint(3, 5)
    cell_h = rng.randint(2, 3)
    cell_w = rng.randint(2, 3)
    marker_count = rng.randint(2, 4)

    nrooms = room_rows * room_cols
    if nrooms < marker_count + 2:
        return None

    target_regions = rng.randint(max(marker_count + 1, 4), min(nrooms, marker_count + 5))
    merge_target = nrooms - target_regions

    edges: list[tuple[str, int, int, int, int]] = []
    for rr in range(room_rows):
        for cc in range(room_cols):
            a = _room_id(rr, cc, room_cols)
            if cc + 1 < room_cols:
                edges.append(("v", rr, cc, a, _room_id(rr, cc + 1, room_cols)))
            if rr + 1 < room_rows:
                edges.append(("h", rr, cc, a, _room_id(rr + 1, cc, room_cols)))
    rng.shuffle(edges)

    dsu = _DSU(nrooms)
    openings: list[tuple[str, int, int]] = []
    merges = 0
    for kind, rr, cc, a, b in edges:
        if merges == merge_target:
            break
        if dsu.union(a, b):
            openings.append((kind, rr, cc))
            merges += 1
    if merges != merge_target:
        return None

    groups = _room_groups(dsu, room_rows, room_cols)
    if len(groups) != target_regions:
        return None

    graph = _region_graph(dsu, room_rows, room_cols)
    path = _find_path(graph, marker_count, rng)
    if path is None:
        return None

    maze = _build_base_maze(room_rows, room_cols, cell_h, cell_w)
    for kind, rr, cc in openings:
        _open_door(maze, kind, rr, cc, cell_h, cell_w, rng)
    if not _connected(maze == WALL):
        return None

    colors = rng.sample(COLOR_POOL, marker_count)
    if rng.random() < 0.5:
        path.reverse()

    for color, root in zip(colors, path):
        room = rng.choice(groups[root])
        _place_marker(maze, room, color, cell_h, cell_w, rng)

    grid = np.zeros((maze.shape[0], maze.shape[1] + 1), dtype=int)
    grid[:, 1:] = maze

    legend_rows = _legend_rows(grid.shape[0], marker_count, rng)
    for row, color in zip(legend_rows, colors):
        grid[row, 0] = color

    out = main(grid)
    if not np.any(out[:, 1:] != grid[:, 1:]):
        return None

    reverse = _solve_with_order(grid, list(reversed(colors)))
    if np.array_equal(out, reverse):
        return None

    if np.count_nonzero(out[:, 1:] == BG) <= np.count_nonzero(grid[:, 1:] == BG):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid maze-wall-removal input")
