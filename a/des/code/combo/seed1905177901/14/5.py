from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREEN = 3
GREY = 5
NODE_COLORS = [1, 2, 4, 6, 7, 8, 9]

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400
MIN_NODES = 3
MAX_NODES = 5


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _line(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]] | None:
    if a[0] == b[0]:
        step = 1 if b[1] >= a[1] else -1
        return [(a[0], c) for c in range(a[1], b[1] + step, step)]
    if a[1] == b[1]:
        step = 1 if b[0] >= a[0] else -1
        return [(r, a[1]) for r in range(a[0], b[0] + step, step)]
    return None


def _candidate_routes(
    start: tuple[int, int], stop: tuple[int, int]
) -> list[tuple[list[tuple[int, int]], int]]:
    routes: list[tuple[list[tuple[int, int]], int]] = []
    straight = _line(start, stop)
    if straight is not None:
        routes.append((straight, 0))
    else:
        bends = [(start[0], stop[1]), (stop[0], start[1])]
        for bend in bends:
            first = _line(start, bend)
            second = _line(bend, stop)
            if first is None or second is None:
                continue
            routes.append((first + second[1:], 1))

    unique: list[tuple[list[tuple[int, int]], int]] = []
    seen = set()
    for path, bends in routes:
        key = tuple(path)
        if key not in seen:
            seen.add(key)
            unique.append((path, bends))
    return unique


def _node_ports(node: dict) -> list[tuple[int, int]]:
    r0, c0 = node["top_left"]
    return [
        (r0 - 1, c0),
        (r0 - 1, c0 + 1),
        (r0 + 2, c0),
        (r0 + 2, c0 + 1),
        (r0, c0 - 1),
        (r0 + 1, c0 - 1),
        (r0, c0 + 2),
        (r0 + 1, c0 + 2),
    ]


def _find_nodes(grid: np.ndarray) -> list[dict]:
    mask = (grid != BLACK) & (grid != GREY)
    comps = _components(mask)
    assert MIN_NODES <= len(comps) <= MAX_NODES, "expected between three and five colored nodes"

    nodes: list[dict] = []
    seen_colors: set[int] = set()
    for coords in comps:
        colors = {int(grid[r, c]) for r, c in coords}
        assert len(colors) == 1, "each non-grey component must be monochrome"
        color = colors.pop()
        assert color not in seen_colors, "node colors must be unique"
        seen_colors.add(color)

        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        assert coords.shape[0] == 4, "nodes must be 2x2 blocks"
        assert (r1 - r0, c1 - c0) == (1, 1), "nodes must be 2x2 blocks"
        assert bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color)), "node block must be solid"
        nodes.append({"color": color, "top_left": (r0, c0), "coords": coords})

    nodes.sort(key=lambda node: (node["top_left"][1], node["top_left"][0]))
    return nodes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert a colored node-and-path network into an adjacency matrix.

    Input structure
    - Black background.
    - Several 2x2 solid colored nodes, each with a distinct color.
    - Grey one-pixel-wide orthogonal paths connect some node pairs.

    Rule
    - Order nodes from left to right, then top to bottom.
    - Trace each 4-connected grey path component and determine which two nodes it touches.
    - Output a KxK grid where diagonal cells keep node colors and connected pairs are green.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    nodes = _find_nodes(g)
    k = len(nodes)
    node_ids = np.full(g.shape, -1, dtype=int)
    for idx, node in enumerate(nodes):
        for r, c in node["coords"]:
            node_ids[int(r), int(c)] = idx

    adjacency = np.zeros((k, k), dtype=bool)
    for path in _components(g == GREY):
        touched: set[int] = set()
        for r, c in path:
            rr, cc = int(r), int(c)
            for dr, dc in DIRS4:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and node_ids[nr, nc] >= 0:
                    touched.add(int(node_ids[nr, nc]))
        assert len(touched) == 2, "each grey path component must connect exactly two nodes"
        i, j = sorted(touched)
        adjacency[i, j] = True
        adjacency[j, i] = True

    out = np.full((k, k), BLACK, dtype=int)
    for idx, node in enumerate(nodes):
        out[idx, idx] = int(node["color"])
    out[adjacency] = GREEN
    for idx, node in enumerate(nodes):
        out[idx, idx] = int(node["color"])
    return out


def _expanded_boxes_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    ar0, ac0 = a[0] - 1, a[1] - 1
    ar1, ac1 = a[0] + 2, a[1] + 2
    br0, bc0 = b[0] - 1, b[1] - 1
    br1, bc1 = b[0] + 2, b[1] + 2
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _sample_nodes(h: int, w: int, k: int) -> list[dict] | None:
    colors = random.sample(NODE_COLORS, k)
    candidates = [(r, c) for r in range(1, h - 2) for c in range(1, w - 2)]
    random.shuffle(candidates)

    positions: list[tuple[int, int]] = []
    for pos in candidates:
        if all(not _expanded_boxes_overlap(pos, other) for other in positions):
            positions.append(pos)
            if len(positions) == k:
                break
    if len(positions) != k:
        return None

    nodes = [
        {"color": color, "top_left": pos}
        for color, pos in zip(colors, positions)
    ]
    nodes.sort(key=lambda node: (node["top_left"][1], node["top_left"][0]))
    return nodes


def _paint_nodes(grid: np.ndarray, nodes: list[dict]) -> np.ndarray:
    node_ids = np.full(grid.shape, -1, dtype=int)
    for idx, node in enumerate(nodes):
        r0, c0 = node["top_left"]
        grid[r0 : r0 + 2, c0 : c0 + 2] = int(node["color"])
        node_ids[r0 : r0 + 2, c0 : c0 + 2] = idx
    return node_ids


def _route_is_clear(
    grid: np.ndarray,
    node_ids: np.ndarray,
    path: list[tuple[int, int]],
    endpoints: tuple[int, int],
) -> bool:
    h, w = grid.shape
    path_set = set(path)
    touched: set[int] = set()

    for r, c in path:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != BLACK:
            return False

    for r, c in path:
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if int(grid[nr, nc]) == GREY and (nr, nc) not in path_set:
                return False
            node_idx = int(node_ids[nr, nc])
            if node_idx >= 0:
                touched.add(node_idx)
                if node_idx not in endpoints:
                    return False

    return touched == set(endpoints)


def _valid_routes(
    grid: np.ndarray,
    nodes: list[dict],
    node_ids: np.ndarray,
    i: int,
    j: int,
) -> list[tuple[list[tuple[int, int]], int]]:
    routes: list[tuple[list[tuple[int, int]], int]] = []
    for start in _node_ports(nodes[i]):
        for stop in _node_ports(nodes[j]):
            for path, bends in _candidate_routes(start, stop):
                if _route_is_clear(grid, node_ids, path, (i, j)):
                    routes.append((path, bends))

    unique: list[tuple[list[tuple[int, int]], int]] = []
    seen = set()
    for path, bends in routes:
        key = tuple(path)
        if key not in seen:
            seen.add(key)
            unique.append((path, bends))
    return unique


def _target_edge_count(k: int) -> int:
    max_edges = k * (k - 1) // 2
    return random.randint(1, min(max_edges - 1, k + 1))


def _route_edges(
    grid: np.ndarray,
    nodes: list[dict],
    node_ids: np.ndarray,
) -> set[tuple[int, int]] | None:
    k = len(nodes)
    target_edges = _target_edge_count(k)
    remaining = [(i, j) for i in range(k) for j in range(i + 1, k)]
    edges: set[tuple[int, int]] = set()

    while len(edges) < target_edges:
        options: list[tuple[tuple[int, int], list[tuple[list[tuple[int, int]], int]]]] = []
        for pair in remaining:
            routes = _valid_routes(grid, nodes, node_ids, pair[0], pair[1])
            if routes:
                options.append((pair, routes))
        if not options:
            return None

        preferred_bends = 1 if random.random() < 0.5 else 0
        preferred_options = []
        for pair, routes in options:
            matching = [route for route in routes if route[1] == preferred_bends]
            if matching:
                preferred_options.append((pair, matching))
        pool = preferred_options if preferred_options else options
        pair, routes = random.choice(pool)
        path, _bends = random.choice(routes)
        for r, c in path:
            grid[r, c] = GREY
        edges.add(pair)
        remaining.remove(pair)

    return edges


def _expected_output(nodes: list[dict], edges: set[tuple[int, int]]) -> np.ndarray:
    k = len(nodes)
    out = np.full((k, k), BLACK, dtype=int)
    for idx, node in enumerate(nodes):
        out[idx, idx] = int(node["color"])
    for i, j in edges:
        out[i, j] = GREEN
        out[j, i] = GREEN
    for idx, node in enumerate(nodes):
        out[idx, idx] = int(node["color"])
    return out


def _generate_candidate() -> np.ndarray | None:
    k = random.randint(MIN_NODES, MAX_NODES)
    h = random.randint(10 + k, 15 + k)
    w = random.randint(10 + k, 15 + k)

    nodes = _sample_nodes(h, w, k)
    if nodes is None:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    node_ids = _paint_nodes(grid, nodes)
    edges = _route_edges(grid, nodes, node_ids)
    if not edges:
        return None

    expected = _expected_output(nodes, edges)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid node-network input")
