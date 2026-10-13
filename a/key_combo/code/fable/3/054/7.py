from __future__ import annotations

import random
from collections import deque

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9


PALETTE = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.GREY, Color.PINK, Color.ORANGE, Color.MAROON]
MAX_GENERATION_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _find_wall_color(grid: np.ndarray) -> int:
    colors = [int(c) for c in np.unique(grid) if int(c) != Color.BLACK]
    assert colors, "expected at least one non-black color"
    if Color.TEAL in colors:
        return Color.TEAL
    return max(colors, key=lambda c: int(np.count_nonzero(grid == c)))


def _door_orientation(grid: np.ndarray, r: int, c: int, wall_color: int) -> str | None:
    if int(grid[r, c]) == wall_color:
        return None
    vertical = (
        int(grid[r - 1, c]) == wall_color
        and int(grid[r + 1, c]) == wall_color
        and int(grid[r, c - 1]) != wall_color
        and int(grid[r, c + 1]) != wall_color
    )
    horizontal = (
        int(grid[r, c - 1]) == wall_color
        and int(grid[r, c + 1]) == wall_color
        and int(grid[r - 1, c]) != wall_color
        and int(grid[r + 1, c]) != wall_color
    )
    if vertical == horizontal:
        return None
    return "v" if vertical else "h"


def _find_doors(grid: np.ndarray, wall_color: int) -> dict[tuple[int, int], str]:
    h, w = grid.shape
    doors: dict[tuple[int, int], str] = {}
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            orientation = _door_orientation(grid, r, c, wall_color)
            if orientation is not None:
                doors[(r, c)] = orientation
    return doors


def _extract_rooms(
    grid: np.ndarray, wall_color: int, doors: dict[tuple[int, int], str]
) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    door_mask = np.zeros_like(grid, dtype=bool)
    for r, c in doors:
        door_mask[r, c] = True

    floor = (grid != wall_color) & ~door_mask
    room_id = np.full(grid.shape, -1, dtype=int)
    rooms: list[list[tuple[int, int]]] = []

    for start_r, start_c in np.argwhere(floor):
        start_r = int(start_r)
        start_c = int(start_c)
        if room_id[start_r, start_c] != -1:
            continue
        rid = len(rooms)
        room_cells: list[tuple[int, int]] = []
        stack = [(start_r, start_c)]
        room_id[start_r, start_c] = rid
        while stack:
            r, c = stack.pop()
            room_cells.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if not _in_bounds(grid.shape, nr, nc) or not floor[nr, nc] or room_id[nr, nc] != -1:
                    continue
                room_id[nr, nc] = rid
                stack.append((nr, nc))
        rooms.append(room_cells)

    assert rooms, "expected at least one room"
    return room_id, rooms


def _find_seed_pair(
    grid: np.ndarray,
    room_id: np.ndarray,
    doors: dict[tuple[int, int], str],
    wall_color: int,
) -> tuple[int, int, int]:
    door_mask = np.zeros_like(grid, dtype=bool)
    for r, c in doors:
        door_mask[r, c] = True

    candidates: list[tuple[int, int, int]] = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w - 1):
            left = int(grid[r, c])
            right = int(grid[r, c + 1])
            if left in (Color.BLACK, wall_color) or right in (Color.BLACK, wall_color):
                continue
            if left == right or door_mask[r, c] or door_mask[r, c + 1]:
                continue
            left_room = int(room_id[r, c])
            right_room = int(room_id[r, c + 1])
            if left_room >= 0 and left_room == right_room:
                candidates.append((left, right, left_room))

    assert len(candidates) == 1, "expected one horizontal two-color seed pair"
    return candidates[0]


def _build_open_graph(
    grid: np.ndarray, room_id: np.ndarray, rooms: list[list[tuple[int, int]]], doors: dict[tuple[int, int], str]
) -> list[set[int]]:
    graph = [set() for _ in rooms]
    for (r, c), orientation in doors.items():
        if orientation == "v":
            a = int(room_id[r, c - 1])
            b = int(room_id[r, c + 1])
        else:
            a = int(room_id[r - 1, c])
            b = int(room_id[r + 1, c])
        if a < 0 or b < 0 or a == b:
            continue
        if int(grid[r, c]) == Color.BLACK:
            graph[a].add(b)
            graph[b].add(a)
    return graph


def _bfs_depths(graph: list[set[int]], start: int) -> list[int]:
    depth = [-1] * len(graph)
    depth[start] = 0
    q: deque[int] = deque([start])
    while q:
        cur = q.popleft()
        for nxt in graph[cur]:
            if depth[nxt] != -1:
                continue
            depth[nxt] = depth[cur] + 1
            q.append(nxt)
    return depth


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color each reachable room by BFS parity from the two-pixel seed pair, using doorways as graph edges.

    Input structure:
    - Cyan walls partition the black area into rooms, with one-cell doorway slots on some shared walls.
    - A doorway slot is either open black, blocked by one non-cyan plug pixel, or absent entirely.
    - Exactly one room contains a horizontal pair of distinct non-black seed pixels.

    Rule:
    - Detect doorway-shaped cells and temporarily close them to recover the room components.
    - Treat only open black doorway cells as edges between adjacent rooms; plugged doorway cells stay closed.
    - Starting from the seed room, run BFS on the room graph and repaint every even-depth room with the left seed color and every odd-depth room with the right seed color.

    Key insight:
    - The alternating fill unit is the room graph node, not the individual doorway pixel, so door slots must be interpreted as edges before coloring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    wall_color = _find_wall_color(g)
    doors = _find_doors(g, wall_color)
    room_id, rooms = _extract_rooms(g, wall_color, doors)
    left_color, right_color, seed_room = _find_seed_pair(g, room_id, doors, wall_color)
    graph = _build_open_graph(g, room_id, rooms, doors)
    depths = _bfs_depths(graph, seed_room)

    out = g.copy()
    for rid, room_cells in enumerate(rooms):
        if depths[rid] == -1:
            continue
        fill_color = left_color if depths[rid] % 2 == 0 else right_color
        for r, c in room_cells:
            out[r, c] = fill_color
    return out


class _UnionFind:
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


def _room_boxes(heights: list[int], widths: list[int]) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    row = 1
    for h in heights:
        col = 1
        for w in widths:
            boxes.append((row, col, row + h - 1, col + w - 1))
            col += w + 1
        row += h + 1
    return boxes


def _base_grid(heights: list[int], widths: list[int]) -> np.ndarray:
    h = sum(heights) + len(heights) + 1
    w = sum(widths) + len(widths) + 1
    grid = np.full((h, w), Color.BLACK, dtype=int)

    wall_rows = [0]
    acc = 0
    for size in heights:
        acc += size + 1
        wall_rows.append(acc)
    wall_cols = [0]
    acc = 0
    for size in widths:
        acc += size + 1
        wall_cols.append(acc)

    grid[wall_rows, :] = Color.TEAL
    grid[:, wall_cols] = Color.TEAL
    return grid


def _edge_specs(
    heights: list[int], widths: list[int], boxes: list[tuple[int, int, int, int]], n_rows: int, n_cols: int
) -> list[dict[str, int | str]]:
    edges: list[dict[str, int | str]] = []
    for rr in range(n_rows):
        for cc in range(n_cols):
            idx = rr * n_cols + cc
            r0, c0, r1, c1 = boxes[idx]
            if cc + 1 < n_cols:
                edges.append(
                    {
                        "a": idx,
                        "b": idx + 1,
                        "r": random.randint(r0, r1),
                        "c": c1 + 1,
                        "orientation": "v",
                    }
                )
            if rr + 1 < n_rows:
                edges.append(
                    {
                        "a": idx,
                        "b": idx + n_cols,
                        "r": r1 + 1,
                        "c": random.randint(c0, c1),
                        "orientation": "h",
                    }
                )
    return edges


def _build_forest(n_rooms: int, edges: list[dict[str, int | str]], target_open: int) -> set[int] | None:
    order = list(range(len(edges)))
    random.shuffle(order)
    uf = _UnionFind(n_rooms)
    chosen: set[int] = set()
    for edge_idx in order:
        edge = edges[edge_idx]
        if uf.union(int(edge["a"]), int(edge["b"])):
            chosen.add(edge_idx)
            if len(chosen) == target_open:
                return chosen
    return None


def _graph_from_edges(n_rooms: int, edges: list[dict[str, int | str]], open_edges: set[int]) -> list[set[int]]:
    graph = [set() for _ in range(n_rooms)]
    for edge_idx in open_edges:
        edge = edges[edge_idx]
        a = int(edge["a"])
        b = int(edge["b"])
        graph[a].add(b)
        graph[b].add(a)
    return graph


def _candidate_seed_rooms(graph: list[set[int]], require_unreachable: bool) -> list[int]:
    candidates: list[int] = []
    n_rooms = len(graph)
    for room in range(n_rooms):
        depths = _bfs_depths(graph, room)
        reach = [d for d in depths if d != -1]
        if len(reach) < 2:
            continue
        if require_unreachable and len(reach) == n_rooms:
            continue
        if max(reach) >= 2 or len(reach) == 2:
            candidates.append(room)
    return candidates


def _place_seed_pair(grid: np.ndarray, box: tuple[int, int, int, int], left_color: int, right_color: int) -> None:
    r0, c0, r1, c1 = box
    row = random.randint(r0, r1)
    col = random.randint(c0, c1 - 1)
    grid[row, col] = left_color
    grid[row, col + 1] = right_color


def _gen_attempt() -> np.ndarray | None:
    n_rows = random.randint(2, 4)
    n_cols = random.randint(2, 4)
    n_rooms = n_rows * n_cols
    heights = [random.randint(3, 6) for _ in range(n_rows)]
    widths = [random.randint(3, 6) for _ in range(n_cols)]
    boxes = _room_boxes(heights, widths)
    grid = _base_grid(heights, widths)
    edges = _edge_specs(heights, widths, boxes, n_rows, n_cols)

    want_unreachable = random.random() < 0.55
    if want_unreachable:
        target_open = random.randint(2, n_rooms - 2)
    else:
        target_open = n_rooms - 1

    open_edges = _build_forest(n_rooms, edges, target_open)
    if open_edges is None:
        return None

    graph = _graph_from_edges(n_rooms, edges, open_edges)
    seed_candidates = _candidate_seed_rooms(graph, want_unreachable)
    if not seed_candidates:
        return None
    seed_room = random.choice(seed_candidates)

    closed_edges = [idx for idx in range(len(edges)) if idx not in open_edges]
    want_plug = random.random() < 0.8
    plug_edges: set[int] = set()
    for edge_idx in closed_edges:
        if random.random() < 0.45:
            plug_edges.add(edge_idx)
    if want_plug and closed_edges and not plug_edges:
        plug_edges.add(random.choice(closed_edges))

    for edge_idx in open_edges:
        edge = edges[edge_idx]
        grid[int(edge["r"]), int(edge["c"])] = Color.BLACK
    for edge_idx in plug_edges:
        edge = edges[edge_idx]
        grid[int(edge["r"]), int(edge["c"])] = random.choice(PALETTE)

    left_color, right_color = random.sample(PALETTE, 2)
    _place_seed_pair(grid, boxes[seed_room], left_color, right_color)

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
