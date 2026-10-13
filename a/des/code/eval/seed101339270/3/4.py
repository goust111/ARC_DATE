from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
MAGENTA = 6

FILL_COLORS = (RED, BLUE, YELLOW, MAGENTA)
GEN_RETRIES = 300


def _edge_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return tuple(sorted((a, b)))


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=int)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != -1:
                continue
            label = len(components)
            queue = deque([(r, c)])
            labels[r, c] = label
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = label
                        queue.append((nr, nc))
            components.append(coords)
    return labels, components


def _seal_door_gaps(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, str]]]:
    h, w = grid.shape
    sealed = grid == GREEN
    doors: list[tuple[int, int, str]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if grid[r, c] == GREEN:
                continue
            vertical_gap = (
                grid[r - 1, c] == GREEN
                and grid[r + 1, c] == GREEN
                and grid[r, c - 1] != GREEN
                and grid[r, c + 1] != GREEN
            )
            horizontal_gap = (
                grid[r, c - 1] == GREEN
                and grid[r, c + 1] == GREEN
                and grid[r - 1, c] != GREEN
                and grid[r + 1, c] != GREEN
            )
            if vertical_gap and not horizontal_gap:
                sealed[r, c] = True
                doors.append((r, c, "vertical"))
            elif horizontal_gap and not vertical_gap:
                sealed[r, c] = True
                doors.append((r, c, "horizontal"))
    return sealed, doors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate room colors through one-pixel wall doors.

    Input structure:
    - Black background.
    - A one-pixel-thick green wall network partitions the grid into rectangular rooms.
    - Shared walls may contain one black door pixel.
    - Exactly one room contains a red marker pixel.

    Rule:
    - Recover the room partition by temporarily sealing each door gap.
    - Build the room adjacency graph from door positions.
    - Breadth-first levels from the red room map to colors:
      level 0 -> red, level 1 -> blue, level 2 -> yellow, level 3+ -> magenta.
    - Unreachable rooms stay black. Walls and door gaps stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    red_pts = np.argwhere(g == RED)
    assert len(red_pts) == 1, "expected exactly one red start marker"
    start_r, start_c = map(int, red_pts[0])

    sealed_walls, doors = _seal_door_gaps(g)
    labels, components = _label_components(~sealed_walls)

    h, w = g.shape
    room_labels: set[int] = set()
    for label, coords in enumerate(components):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords):
            continue
        room_labels.add(label)
    assert room_labels, "expected at least one enclosed room"

    start_label = int(labels[start_r, start_c])
    assert start_label in room_labels, "red marker must lie inside a room"

    graph = {label: set() for label in room_labels}
    for r, c, orientation in doors:
        if orientation == "vertical":
            a = int(labels[r, c - 1])
            b = int(labels[r, c + 1])
        else:
            a = int(labels[r - 1, c])
            b = int(labels[r + 1, c])
        if a in room_labels and b in room_labels and a != b:
            graph[a].add(b)
            graph[b].add(a)

    distances = {start_label: 0}
    queue = deque([start_label])
    while queue:
        cur = queue.popleft()
        for nxt in graph[cur]:
            if nxt in distances:
                continue
            distances[nxt] = distances[cur] + 1
            queue.append(nxt)

    out = g.copy()
    for label in room_labels:
        color = BLACK if label not in distances else FILL_COLORS[min(distances[label], 3)]
        for r, c in components[label]:
            out[r, c] = color
    return out


def _graph_neighbors(rows: int, cols: int) -> dict[tuple[int, int], list[tuple[int, int]]]:
    neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r in range(rows):
        for c in range(cols):
            pts: list[tuple[int, int]] = []
            if r > 0:
                pts.append((r - 1, c))
            if r + 1 < rows:
                pts.append((r + 1, c))
            if c > 0:
                pts.append((r, c - 1))
            if c + 1 < cols:
                pts.append((r, c + 1))
            neighbors[(r, c)] = pts
    return neighbors


def _graph_distances(
    start: tuple[int, int],
    nodes: list[tuple[int, int]],
    edges: set[tuple[tuple[int, int], tuple[int, int]]],
) -> dict[tuple[int, int], int]:
    adjacency = {node: set() for node in nodes}
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    distances = {start: 0}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        for nxt in adjacency[cur]:
            if nxt in distances:
                continue
            distances[nxt] = distances[cur] + 1
            queue.append(nxt)
    return distances


def _simple_path(
    start: tuple[int, int],
    length: int,
    neighbors: dict[tuple[int, int], list[tuple[int, int]]],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    used = {start}

    def dfs(node: tuple[int, int], remaining: int) -> list[tuple[int, int]] | None:
        if remaining == 0:
            return [node]
        options = neighbors[node][:]
        rng.shuffle(options)
        for nxt in options:
            if nxt in used:
                continue
            used.add(nxt)
            tail = dfs(nxt, remaining - 1)
            if tail is not None:
                return [node] + tail
            used.remove(nxt)
        return None

    return dfs(start, length)


def _choose_dims(mode: str, rng: random.Random) -> tuple[int, int]:
    candidates: list[tuple[int, int]]
    if mode == "depth3":
        candidates = [(2, 3), (3, 2), (2, 4), (4, 2), (3, 3), (3, 4), (4, 3)]
    elif mode == "partial":
        candidates = [(2, 2), (2, 3), (3, 2), (3, 3), (2, 4), (4, 2), (3, 4), (4, 3), (4, 4)]
    else:
        candidates = [(2, 2), (2, 3), (3, 2), (3, 3), (2, 4), (4, 2), (3, 4), (4, 3)]
    return rng.choice(candidates)


def _build_connected_graph(
    rows: int, cols: int, rng: random.Random
) -> tuple[tuple[int, int], set[tuple[tuple[int, int], tuple[int, int]]], dict[tuple[int, int], int]] | None:
    nodes = [(r, c) for r in range(rows) for c in range(cols)]
    candidates = [
        node
        for node in nodes
        if max(abs(node[0] - r) + abs(node[1] - c) for r in range(rows) for c in range(cols)) <= 3
    ]
    if not candidates:
        return None
    start = rng.choice(candidates)
    neighbors = _graph_neighbors(rows, cols)
    open_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    other_nodes = [node for node in nodes if node != start]
    rng.shuffle(other_nodes)
    for node in other_nodes:
        node_dist = abs(node[0] - start[0]) + abs(node[1] - start[1])
        parents = [
            parent
            for parent in neighbors[node]
            if abs(parent[0] - start[0]) + abs(parent[1] - start[1]) == node_dist - 1
        ]
        if not parents:
            return None
        open_edges.add(_edge_key(node, rng.choice(parents)))

    for node in nodes:
        for nbr in neighbors[node]:
            edge = _edge_key(node, nbr)
            if edge in open_edges:
                continue
            if rng.random() < 0.25:
                open_edges.add(edge)

    distances = _graph_distances(start, nodes, open_edges)
    if len(distances) != len(nodes):
        return None
    if max(distances.values()) < 2 or max(distances.values()) > 3:
        return None
    return start, open_edges, distances


def _build_layered_graph(
    rows: int, cols: int, target_depth: int, rng: random.Random
) -> tuple[tuple[int, int], set[tuple[tuple[int, int], tuple[int, int]]], dict[tuple[int, int], int]] | None:
    nodes = [(r, c) for r in range(rows) for c in range(cols)]
    if len(nodes) < target_depth + 2:
        return None
    neighbors = _graph_neighbors(rows, cols)

    starts = nodes[:]
    rng.shuffle(starts)
    path = None
    start = None
    for candidate in starts:
        trial = _simple_path(candidate, target_depth, neighbors, rng)
        if trial is not None:
            start = candidate
            path = trial
            break
    if path is None or start is None:
        return None

    open_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    distances = {node: depth for depth, node in enumerate(path)}
    for a, b in zip(path, path[1:]):
        open_edges.add(_edge_key(a, b))

    target_reachable = rng.randint(target_depth + 1, len(nodes) - 1)
    while len(distances) < target_reachable:
        candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for parent, depth in distances.items():
            if depth >= target_depth:
                continue
            for child in neighbors[parent]:
                if child not in distances:
                    candidates.append((parent, child))
        if not candidates:
            return None
        parent, child = rng.choice(candidates)
        open_edges.add(_edge_key(parent, child))
        distances[child] = distances[parent] + 1

    unreached = [node for node in nodes if node not in distances]
    for node in unreached:
        for nbr in neighbors[node]:
            if nbr in distances or node > nbr:
                continue
            if rng.random() < 0.3:
                open_edges.add(_edge_key(node, nbr))

    actual = _graph_distances(start, nodes, open_edges)
    if actual != distances:
        return None
    if max(distances.values()) != target_depth:
        return None
    return start, open_edges, distances


def _room_layout(
    rows: int, cols: int, rng: random.Random
) -> tuple[np.ndarray, list[int], list[int], dict[tuple[int, int], tuple[int, int, int, int]]]:
    heights = [rng.randint(2, 5) for _ in range(rows)]
    widths = [rng.randint(2, 5) for _ in range(cols)]
    top = rng.randint(0, 2)
    left = rng.randint(0, 2)
    bottom = rng.randint(0, 2)
    right = rng.randint(0, 2)

    wall_rows = [top]
    for height in heights:
        wall_rows.append(wall_rows[-1] + height + 1)
    wall_cols = [left]
    for width in widths:
        wall_cols.append(wall_cols[-1] + width + 1)

    h = wall_rows[-1] + bottom + 1
    w = wall_cols[-1] + right + 1
    grid = np.full((h, w), BLACK, dtype=int)

    top_wall, bottom_wall = wall_rows[0], wall_rows[-1]
    left_wall, right_wall = wall_cols[0], wall_cols[-1]
    for r in wall_rows:
        grid[r, left_wall : right_wall + 1] = GREEN
    for c in wall_cols:
        grid[top_wall : bottom_wall + 1, c] = GREEN

    boxes: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for r in range(rows):
        for c in range(cols):
            boxes[(r, c)] = (
                wall_rows[r] + 1,
                wall_cols[c] + 1,
                wall_rows[r + 1] - 1,
                wall_cols[c + 1] - 1,
            )
    return grid, wall_rows, wall_cols, boxes


def _carve_doors(
    grid: np.ndarray,
    wall_rows: list[int],
    wall_cols: list[int],
    boxes: dict[tuple[int, int], tuple[int, int, int, int]],
    open_edges: set[tuple[tuple[int, int], tuple[int, int]]],
    rng: random.Random,
) -> None:
    for a, b in open_edges:
        if a[0] == b[0]:
            row = a[0]
            col = min(a[1], b[1])
            wall_c = wall_cols[col + 1]
            r0, _, r1, _ = boxes[(row, col)]
            grid[rng.randint(r0, r1), wall_c] = BLACK
        else:
            row = min(a[0], b[0])
            col = a[1]
            wall_r = wall_rows[row + 1]
            _, c0, _, c1 = boxes[(row, col)]
            grid[wall_r, rng.randint(c0, c1)] = BLACK


def _expected_output(
    grid: np.ndarray,
    boxes: dict[tuple[int, int], tuple[int, int, int, int]],
    distances: dict[tuple[int, int], int],
) -> np.ndarray:
    out = grid.copy()
    for room, (r0, c0, r1, c1) in boxes.items():
        color = BLACK if room not in distances else FILL_COLORS[min(distances[room], 3)]
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    mode = rng.choice(["connected", "partial", "depth3"])
    rows, cols = _choose_dims(mode, rng)

    if mode == "connected":
        graph = _build_connected_graph(rows, cols, rng)
    elif mode == "partial":
        graph = _build_layered_graph(rows, cols, rng.choice([1, 2]), rng)
    else:
        graph = _build_layered_graph(rows, cols, 3, rng)
    if graph is None:
        return None

    start, open_edges, distances = graph
    if len(distances) < 2:
        return None
    if mode == "partial" and len(distances) == rows * cols:
        return None
    if mode == "depth3" and not {1, 2, 3}.issubset(set(distances.values())):
        return None

    grid, wall_rows, wall_cols, boxes = _room_layout(rows, cols, rng)
    _carve_doors(grid, wall_rows, wall_cols, boxes, open_edges, rng)

    r0, c0, r1, c1 = boxes[start]
    grid[rng.randint(r0, r1), rng.randint(c0, c1)] = RED

    expected = _expected_output(grid, boxes, distances)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-door puzzle input")
