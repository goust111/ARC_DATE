from __future__ import annotations

import heapq
import random
from collections import defaultdict, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

TERMINAL_COLOR = RED
STATION_COLOR = BLUE
TRACK_COLOR = GREY
BEAD_COLOR = YELLOW

NODE_LOGICAL_POS = {
    0: (0, 0),  # left terminal
    1: (0, 1),
    2: (0, 2),
    3: (0, 3),  # right terminal
    4: (1, 0),
    5: (1, 1),
    6: (1, 2),
    7: (1, 3),
}
TERMINAL_IDS = (0, 3)
MANDATORY_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (4, 5),
    (5, 6),
    (6, 7),
    (0, 4),
    (3, 7),
)
OPTIONAL_EDGES = ((1, 5), (2, 6))
TARGET_NODE_PATH = (0, 4, 5, 6, 7, 3)
TARGET_EDGE_PATH = tuple(tuple(sorted(edge)) for edge in ((0, 4), (4, 5), (5, 6), (6, 7), (3, 7)))

GENERATE_RETRIES = 400


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        nbrs = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]

    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in nbrs:
                    nr, nc = rr + dr, cc + dc
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _solid_square_components(grid: np.ndarray, color: int) -> list[dict]:
    comps = components_mask(grid == color, connectivity=4)
    out: list[dict] = []
    for coords in comps:
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 2 or h != w or coords.shape[0] != h * w:
            continue
        if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color):
            continue
        out.append(
            {
                "color": int(color),
                "bbox": (r0, c0, r1, c1),
                "size": int(h),
                "coords": [tuple(map(int, rc)) for rc in coords],
            }
        )
    return out


def _analyze_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    terminal_nodes = _solid_square_components(g, TERMINAL_COLOR)
    station_nodes = _solid_square_components(g, STATION_COLOR)
    assert len(terminal_nodes) == 2
    assert len(station_nodes) >= 4

    nodes = terminal_nodes + station_nodes
    nodes = sorted(nodes, key=lambda node: node["bbox"])

    pixel_to_node: dict[tuple[int, int], int] = {}
    for node_id, node in enumerate(nodes):
        node["id"] = node_id
        node["coord_set"] = set(node["coords"])
        for rc in node["coords"]:
            pixel_to_node[rc] = node_id

    red_ids = [node["id"] for node in nodes if node["color"] == TERMINAL_COLOR]
    assert len(red_ids) == 2

    track_mask = (g == TRACK_COLOR) | (g == BEAD_COLOR)
    edge_comps = components_mask(track_mask, connectivity=4)
    assert edge_comps

    edges: dict[int, dict] = {}
    edge_by_nodes: dict[tuple[int, int], int] = {}
    adjacency: dict[int, list[tuple[int, int]]] = defaultdict(list)

    for edge_id, coords in enumerate(edge_comps):
        touched: set[int] = set()
        coord_list = [tuple(map(int, rc)) for rc in coords]
        for r, c in coord_list:
            for nr, nc in neighbors4(r, c):
                node_id = pixel_to_node.get((nr, nc))
                if node_id is not None:
                    touched.add(node_id)
        assert len(touched) == 2
        a, b = sorted(touched)
        key = (a, b)
        assert key not in edge_by_nodes
        weight = int(sum(g[r, c] == BEAD_COLOR for r, c in coord_list))
        edges[edge_id] = {
            "id": edge_id,
            "nodes": key,
            "coords": coord_list,
            "weight": weight,
            "length": len(coord_list),
        }
        edge_by_nodes[key] = edge_id
        adjacency[a].append((b, edge_id))
        adjacency[b].append((a, edge_id))

    return {
        "grid": g,
        "nodes": nodes,
        "terminals": tuple(red_ids),
        "edges": edges,
        "edge_by_nodes": edge_by_nodes,
        "adjacency": adjacency,
    }


def _weighted_shortest(scene: dict) -> tuple[list[int], list[int]]:
    start, goal = scene["terminals"]
    pq = [(0, start)]
    dist = {start: 0}
    prev_node = {start: None}
    prev_edge = {start: None}

    while pq:
        cost, node = heapq.heappop(pq)
        if cost != dist[node]:
            continue
        if node == goal:
            break
        for nxt, edge_id in scene["adjacency"][node]:
            edge_cost = scene["edges"][edge_id]["weight"]
            new_cost = cost + edge_cost
            if new_cost < dist.get(nxt, 10**9):
                dist[nxt] = new_cost
                prev_node[nxt] = node
                prev_edge[nxt] = edge_id
                heapq.heappush(pq, (new_cost, nxt))

    assert goal in dist
    node_path: list[int] = []
    edge_path: list[int] = []
    cur = goal
    while cur is not None:
        node_path.append(cur)
        edge_id = prev_edge[cur]
        if edge_id is not None:
            edge_path.append(edge_id)
        cur = prev_node[cur]
    node_path.reverse()
    edge_path.reverse()
    return node_path, edge_path


def _render_path(grid: np.ndarray, scene: dict, node_path: list[int], edge_path: list[int]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for node_id in node_path:
        node = scene["nodes"][node_id]
        if node["color"] != STATION_COLOR:
            continue
        for r, c in node["coords"]:
            out[r, c] = GREEN
    for edge_id in edge_path:
        for r, c in scene["edges"][edge_id]["coords"]:
            if grid[r, c] == TRACK_COLOR:
                out[r, c] = GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the unique minimum-bead route between the two red terminals.

    Input structure:
    - Black background contains exactly two solid red terminal squares and several solid blue station squares.
    - One-cell-wide tracks run between squares; track pixels are grey except that some track cells are yellow beads.
    - Removing the squares leaves a graph whose nodes are the squares and whose edges are the mixed grey/yellow track segments.

    Rule:
    - Treat every square as a graph node and every connected track segment between two squares as one graph edge.
    - The cost of an edge is the number of yellow bead pixels on that segment, not its geometric length or its edge count.
    - Find the unique minimum-total-cost path between the two red terminals, recolor every blue station on that path to green,
      and recolor only the grey track pixels on that path to green while keeping all yellow beads unchanged.

    Key insight:
    - Several visible routes compete, but the decisive metric is the sum of yellow bead pixels on each track segment.
    """
    scene = _analyze_scene(np.asarray(grid, dtype=int))
    node_path, edge_path = _weighted_shortest(scene)
    return _render_path(np.asarray(grid, dtype=int), scene, node_path, edge_path)


def _all_simple_paths(scene: dict) -> list[dict]:
    start, goal = scene["terminals"]
    paths: list[dict] = []

    def dfs(node: int, visited: set[int], node_path: list[int], edge_path: list[int]) -> None:
        if node == goal:
            weight = sum(scene["edges"][edge_id]["weight"] for edge_id in edge_path)
            length = sum(scene["edges"][edge_id]["length"] for edge_id in edge_path)
            paths.append(
                {
                    "nodes": tuple(node_path),
                    "edges": tuple(edge_path),
                    "weight": int(weight),
                    "geom": int(length),
                    "hops": len(edge_path),
                }
            )
            return
        for nxt, edge_id in scene["adjacency"][node]:
            if nxt in visited:
                continue
            visited.add(nxt)
            node_path.append(nxt)
            edge_path.append(edge_id)
            dfs(nxt, visited, node_path, edge_path)
            edge_path.pop()
            node_path.pop()
            visited.remove(nxt)

    dfs(start, {start}, [start], [])
    return paths


def _transform_pair(inp: np.ndarray, out: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    k = rng.randrange(4)
    flip = bool(rng.randrange(2))
    inp2 = np.rot90(inp, k)
    out2 = np.rot90(out, k)
    if flip:
        inp2 = np.fliplr(inp2)
        out2 = np.fliplr(out2)
    return inp2.copy(), out2.copy()


def _sample_sizes(rng: random.Random) -> dict[int, int]:
    sizes = {node_id: rng.choice([2, 3]) for node_id in NODE_LOGICAL_POS}
    blue_ids = [node_id for node_id in NODE_LOGICAL_POS if node_id not in TERMINAL_IDS]
    if len({sizes[node_id] for node_id in blue_ids}) == 1:
        flip_id = rng.choice(blue_ids)
        sizes[flip_id] = 5 - sizes[flip_id]
    return sizes


def _make_layout(rng: random.Random) -> tuple[dict[int, dict], tuple[int, int]] | None:
    row0 = rng.randint(4, 6)
    row_gap = rng.randint(7, 9)
    rows = [row0, row0 + row_gap]

    c0 = rng.randint(4, 5)
    gaps = [rng.randint(6, 7) for _ in range(3)]
    cols = [c0]
    for gap in gaps:
        cols.append(cols[-1] + gap)

    sizes = _sample_sizes(rng)
    nodes: dict[int, dict] = {}
    max_r = 0
    max_c = 0
    for node_id, (ri, ci) in NODE_LOGICAL_POS.items():
        size = sizes[node_id]
        r0 = rows[ri] - 1
        c0 = cols[ci] - 1
        nodes[node_id] = {
            "row_anchor": rows[ri],
            "col_anchor": cols[ci],
            "r0": r0,
            "c0": c0,
            "size": size,
            "color": TERMINAL_COLOR if node_id in TERMINAL_IDS else STATION_COLOR,
        }
        max_r = max(max_r, r0 + size)
        max_c = max(max_c, c0 + size)
    h = max_r + rng.randint(2, 3)
    w = max_c + rng.randint(2, 3)
    if h > 30 or w > 30:
        return None
    return nodes, (h, w)


def _edge_cells(nodes: dict[int, dict], edge: tuple[int, int]) -> list[tuple[int, int]]:
    a, b = edge
    na = nodes[a]
    nb = nodes[b]
    if na["row_anchor"] == nb["row_anchor"]:
        row = na["row_anchor"]
        left_id, right_id = (a, b) if na["col_anchor"] < nb["col_anchor"] else (b, a)
        left = nodes[left_id]
        right = nodes[right_id]
        start = left["c0"] + left["size"]
        end = right["c0"] - 1
        assert start <= end
        return [(row, c) for c in range(start, end + 1)]

    assert na["col_anchor"] == nb["col_anchor"]
    col = na["col_anchor"]
    upper_id, lower_id = (a, b) if na["row_anchor"] < nb["row_anchor"] else (b, a)
    upper = nodes[upper_id]
    lower = nodes[lower_id]
    start = upper["r0"] + upper["size"]
    end = lower["r0"] - 1
    assert start <= end
    return [(r, col) for r in range(start, end + 1)]


def _enumerate_edge_keys(edge_list: tuple[tuple[int, int], ...], extra_edges: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    return [tuple(sorted(edge)) for edge in edge_list + extra_edges]


def _sample_weights(
    edge_cells: dict[tuple[int, int], list[tuple[int, int]]],
    extra_edges: tuple[tuple[int, int], ...],
    rng: random.Random,
) -> dict[tuple[int, int], int]:
    weights: dict[tuple[int, int], int] = {}

    for edge in [tuple(sorted(x)) for x in ((0, 1), (1, 2), (2, 3))]:
        length = len(edge_cells[edge])
        hi = min(4, length - 1)
        lo = min(2, hi)
        weights[edge] = rng.randint(lo, hi)

    for edge in [tuple(sorted(x)) for x in ((4, 5), (5, 6), (6, 7))]:
        length = len(edge_cells[edge])
        weights[edge] = rng.randint(0, min(1, length - 1))

    for edge in [tuple(sorted(x)) for x in ((0, 4), (3, 7))]:
        length = len(edge_cells[edge])
        weights[edge] = rng.randint(0, min(1, length - 1))

    for edge in [tuple(sorted(x)) for x in extra_edges]:
        length = len(edge_cells[edge])
        lo = 1
        hi = min(3, length - 1)
        weights[edge] = rng.randint(lo, hi)

    target_total = sum(weights[edge] for edge in TARGET_EDGE_PATH)
    if target_total == 0:
        edge = rng.choice(TARGET_EDGE_PATH)
        weights[edge] = 1
    return weights


def _paint_case(
    nodes: dict[int, dict],
    shape: tuple[int, int],
    edge_cells: dict[tuple[int, int], list[tuple[int, int]]],
    weights: dict[tuple[int, int], int],
    rng: random.Random,
) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)

    for node in nodes.values():
        r0 = node["r0"]
        c0 = node["c0"]
        size = node["size"]
        grid[r0 : r0 + size, c0 : c0 + size] = node["color"]

    occupied = np.zeros(shape, dtype=bool)
    occupied[grid != BLACK] = True

    for edge, cells in edge_cells.items():
        for r, c in cells:
            assert not occupied[r, c]
        bead_count = weights[edge]
        bead_cells = set(rng.sample(cells, bead_count)) if bead_count else set()
        for r, c in cells:
            grid[r, c] = BEAD_COLOR if (r, c) in bead_cells else TRACK_COLOR
            occupied[r, c] = True

    return grid


def _expected_from_target(inp: np.ndarray, nodes: dict[int, dict], edge_cells: dict[tuple[int, int], list[tuple[int, int]]]) -> np.ndarray:
    out = inp.copy()
    for node_id in TARGET_NODE_PATH[1:-1]:
        node = nodes[node_id]
        r0 = node["r0"]
        c0 = node["c0"]
        size = node["size"]
        out[r0 : r0 + size, c0 : c0 + size] = GREEN
    for edge in TARGET_EDGE_PATH:
        for r, c in edge_cells[edge]:
            if inp[r, c] == TRACK_COLOR:
                out[r, c] = GREEN
    return out


def _make_attempt(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    layout = _make_layout(rng)
    if layout is None:
        return None
    nodes, shape = layout
    extra_count = rng.choice([1, 2])
    extra_edges = tuple(sorted(tuple(sorted(edge)) for edge in rng.sample(list(OPTIONAL_EDGES), extra_count)))
    graph_edges = _enumerate_edge_keys(MANDATORY_EDGES, extra_edges)

    edge_cells = {edge: _edge_cells(nodes, edge) for edge in graph_edges}
    if min(len(cells) for cells in edge_cells.values()) < 2:
        return None

    for _ in range(60):
        weights = _sample_weights(edge_cells, extra_edges, rng)
        inp = _paint_case(nodes, shape, edge_cells, weights, rng)
        exp = _expected_from_target(inp, nodes, edge_cells)
        inp_t, exp_t = _transform_pair(inp, exp, rng)
        if _validate_case(inp_t, exp_t):
            return inp_t, exp_t
    return None


def _validate_case(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        scene = _analyze_scene(inp)
        paths = _all_simple_paths(scene)
        if len(paths) < 3:
            return False

        by_weight = min(path["weight"] for path in paths)
        weight_best = [path for path in paths if path["weight"] == by_weight]
        if len(weight_best) != 1:
            return False
        best = weight_best[0]

        min_hops = min(path["hops"] for path in paths)
        min_geom = min(path["geom"] for path in paths)
        if best["hops"] <= min_hops:
            return False
        if best["geom"] <= min_geom:
            return False

        rendered = _render_path(inp, scene, list(best["nodes"]), list(best["edges"]))
        if not np.array_equal(rendered, expected):
            return False
        if not np.array_equal(main(inp), expected):
            return False

        green_count = int(np.count_nonzero(expected == GREEN))
        if green_count <= 0:
            return False
        if int(np.count_nonzero(inp == BEAD_COLOR)) < 4:
            return False
        return True
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        rng = random.Random()
        case = _make_attempt(rng)
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid bead-shortest-path puzzle")
