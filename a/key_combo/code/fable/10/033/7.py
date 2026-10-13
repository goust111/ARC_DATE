from __future__ import annotations

from collections import defaultdict, deque
import heapq
import random

import numpy as np


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

PORTAL_COLORS = (GREEN, GREY, PINK, ORANGE, TEAL, MAROON)
RESERVED_COLORS = {BLACK, BLUE, RED, YELLOW}

Coord = tuple[int, int]

GEN_RETRIES = 500
GEN_VARIANTS = ("two_step_trap", "three_step_chain", "backtrack_chain")
GEN_WEIGHTS = (4, 3, 3)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_endpoints(grid: np.ndarray) -> tuple[Coord, Coord]:
    coords = np.argwhere(grid == RED)
    assert coords.shape == (2, 2), "expected exactly two red endpoints"
    return tuple(map(int, coords[0])), tuple(map(int, coords[1]))


def _parse_portals(grid: np.ndarray) -> tuple[dict[Coord, Coord], dict[int, tuple[Coord, Coord]]]:
    colors = sorted(set(map(int, grid.ravel())) - RESERVED_COLORS)
    portal_pairs: dict[int, tuple[Coord, Coord]] = {}
    coord_to_pair: dict[Coord, Coord] = {}
    for color in colors:
        coords = np.argwhere(grid == color)
        assert coords.shape == (2, 2), f"portal color {color} must appear exactly twice"
        a = tuple(map(int, coords[0]))
        b = tuple(map(int, coords[1]))
        portal_pairs[color] = (a, b)
        coord_to_pair[a] = b
        coord_to_pair[b] = a
    return coord_to_pair, portal_pairs


def _graph_neighbors(
    grid: np.ndarray,
    coord_to_pair: dict[Coord, Coord],
    cell: Coord,
    *,
    allow_teleport: bool,
) -> list[Coord]:
    r, c = cell
    out: list[Coord] = []
    seen: set[Coord] = set()
    for nr, nc in _neighbors4(r, c):
        if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) != BLUE and (nr, nc) not in seen:
            seen.add((nr, nc))
            out.append((nr, nc))
    if allow_teleport and cell in coord_to_pair:
        partner = coord_to_pair[cell]
        if partner not in seen:
            out.append(partner)
    return out


def _dist_without_teleport(grid: np.ndarray, start: Coord) -> dict[Coord, int]:
    dist = {start: 0}
    q: deque[Coord] = deque([start])
    while q:
        cell = q.popleft()
        for nxt in _graph_neighbors(grid, {}, cell, allow_teleport=False):
            if nxt not in dist:
                dist[nxt] = dist[cell] + 1
                q.append(nxt)
    return dist


def _unique_shortest_path(grid: np.ndarray, start: Coord, goal: Coord) -> list[Coord]:
    coord_to_pair, _ = _parse_portals(grid)
    dist = {start: 0}
    ways = defaultdict(int)
    ways[start] = 1
    q: deque[Coord] = deque([start])

    while q:
        cell = q.popleft()
        for nxt in _graph_neighbors(grid, coord_to_pair, cell, allow_teleport=True):
            nd = dist[cell] + 1
            if nxt not in dist:
                dist[nxt] = nd
                ways[nxt] = min(2, ways[cell])
                q.append(nxt)
            elif dist[nxt] == nd:
                ways[nxt] = min(2, ways[nxt] + ways[cell])

    assert goal in dist, "endpoints must be connected once portals are available"
    assert ways[goal] == 1, "shortest path must be unique"

    path = [goal]
    cur = goal
    while cur != start:
        preds = [
            prev
            for prev in _graph_neighbors(grid, coord_to_pair, cur, allow_teleport=True)
            if prev in dist and dist[prev] + 1 == dist[cur]
        ]
        assert len(preds) == 1, "unique shortest path should give one predecessor at every step"
        cur = preds[0]
        path.append(cur)
    path.reverse()
    return path


def _teleport_count(path: list[Coord]) -> int:
    total = 0
    for a, b in zip(path, path[1:]):
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            total += 1
    return total


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint yellow the unique globally shortest route between the two red endpoints, where same-color portal pixels allow one-step teleports.

    Input structure:
    - The grid contains black background, blue wall cells, exactly two red endpoint pixels, and several portal colors.
    - Each portal color appears in exactly two single cells; same-color cells form a teleport pair.
    - The blue walls physically separate the two red endpoints into different ordinary 4-connected regions.

    Rule:
    - A move to any 4-neighbor non-wall cell costs 1 step.
    - If the current cell is a portal, jumping to its same-color partner also costs 1 step.
    - Among all routes from one red endpoint to the other, find the unique shortest route under those costs.
    - Keep the whole input unchanged except recolor to yellow only the black cells that lie on that route; portal pixels and red endpoints stay their original colors.

    Key insight:
    - The visible maze is disconnected; true reachability comes from inferring the hidden graph whose extra edges are the same-color portal pairings, and the correct route is the unique global optimum in that augmented graph rather than the locally nearest portal choice.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30, "expected a 2D ARC-sized grid"
    assert YELLOW not in set(map(int, np.unique(g))), "yellow is reserved for the output path"

    start, goal = _find_endpoints(g)
    path = _unique_shortest_path(g, start, goal)

    out = g.copy()
    for r, c in path:
        if (r, c) != start and (r, c) != goal and int(out[r, c]) == BLACK:
            out[r, c] = YELLOW
    return out


def _add_edge(adj: dict[str, list[tuple[str, int, str, int | None]]], u: str, v: str, w: int, kind: str, color: int | None) -> None:
    adj[u].append((v, w, kind, color))
    adj[v].append((u, w, kind, color))


def _build_abstract_graph(meta: dict) -> tuple[dict[str, dict], dict[str, list[tuple[str, int, str, int | None]]]]:
    nodes: dict[str, dict] = {
        "S": dict(meta["start"]),
        "G": dict(meta["goal"]),
    }
    chamber_nodes: dict[int, list[str]] = defaultdict(list)
    chamber_nodes[meta["start"]["chamber"]].append("S")
    chamber_nodes[meta["goal"]["chamber"]].append("G")

    for index, portal in enumerate(meta["portals"]):
        left_id = f"P{index}a"
        right_id = f"P{index}b"
        left_node = {
            "coord": portal["a"],
            "chamber": portal["a_chamber"],
            "color": portal["color"],
        }
        right_node = {
            "coord": portal["b"],
            "chamber": portal["b_chamber"],
            "color": portal["color"],
        }
        nodes[left_id] = left_node
        nodes[right_id] = right_node
        chamber_nodes[left_node["chamber"]].append(left_id)
        chamber_nodes[right_node["chamber"]].append(right_id)

    adj: dict[str, list[tuple[str, int, str, int | None]]] = defaultdict(list)
    for ids in chamber_nodes.values():
        ids.sort(key=lambda node_id: nodes[node_id]["coord"][1])
        for a, b in zip(ids, ids[1:]):
            ca = nodes[a]["coord"][1]
            cb = nodes[b]["coord"][1]
            _add_edge(adj, a, b, abs(cb - ca), "walk", None)

    for index, portal in enumerate(meta["portals"]):
        _add_edge(adj, f"P{index}a", f"P{index}b", 1, "teleport", portal["color"])

    for node_id in nodes:
        adj.setdefault(node_id, [])
    return nodes, adj


def _dijkstra_unique(adj: dict[str, list[tuple[str, int, str, int | None]]], start: str, goal: str) -> tuple[dict[str, int], dict[str, int]]:
    inf = 10**9
    dist = {node: inf for node in adj}
    ways = {node: 0 for node in adj}
    dist[start] = 0
    ways[start] = 1
    heap: list[tuple[int, str]] = [(0, start)]

    while heap:
        d, node = heapq.heappop(heap)
        if d != dist[node]:
            continue
        for nxt, w, _kind, _color in adj[node]:
            nd = d + w
            if nd < dist[nxt]:
                dist[nxt] = nd
                ways[nxt] = ways[node]
                heapq.heappush(heap, (nd, nxt))
            elif nd == dist[nxt]:
                ways[nxt] = min(2, ways[nxt] + ways[node])
    assert dist[goal] < inf, "abstract graph must connect start to goal"
    return dist, ways


def _reconstruct_abstract_path(
    adj: dict[str, list[tuple[str, int, str, int | None]]],
    dist: dict[str, int],
    start: str,
    goal: str,
) -> list[str]:
    path = [goal]
    cur = goal
    while cur != start:
        preds = [
            prev
            for prev, w, _kind, _color in adj[cur]
            if dist.get(prev, 10**9) + w == dist[cur]
        ]
        assert len(preds) == 1, "abstract shortest route must be unique"
        cur = preds[0]
        path.append(cur)
    path.reverse()
    return path


def _enumerate_simple_routes(
    adj: dict[str, list[tuple[str, int, str, int | None]]],
    start: str,
    goal: str,
) -> list[dict]:
    routes: list[dict] = []
    max_visits = len(adj) + 2

    def dfs(node: str, seen: set[str], cost: int, path: list[str], teleports: list[int]) -> None:
        if len(path) > max_visits:
            return
        if node == goal:
            routes.append(
                {
                    "cost": cost,
                    "path": tuple(path),
                    "teleports": tuple(teleports),
                }
            )
            return
        for nxt, w, kind, color in adj[node]:
            if nxt in seen:
                continue
            new_seen = set(seen)
            new_seen.add(nxt)
            new_teleports = list(teleports)
            if kind == "teleport":
                assert color is not None
                new_teleports.append(color)
            dfs(nxt, new_seen, cost + w, path + [nxt], new_teleports)

    dfs(start, {start}, 0, [start], [])
    dedup: dict[tuple[str, ...], dict] = {}
    for route in routes:
        dedup.setdefault(route["path"], route)
    return list(dedup.values())


def _special_positions(meta: dict) -> set[Coord]:
    pts = {meta["start"]["coord"], meta["goal"]["coord"]}
    for portal in meta["portals"]:
        pts.add(portal["a"])
        pts.add(portal["b"])
    return pts


def _check_path_paints_only_black(grid: np.ndarray, out: np.ndarray, path: list[Coord]) -> None:
    assert out.shape == grid.shape
    expected = grid.copy()
    for r, c in path:
        if int(expected[r, c]) == BLACK:
            expected[r, c] = YELLOW
    assert np.array_equal(out, expected), "output must recolor exactly the black cells on the path"


def _validate_generated(grid: np.ndarray, meta: dict) -> bool:
    try:
        g = np.asarray(grid, dtype=int)
        start, goal = _find_endpoints(g)
        assert start == meta["start"]["coord"]
        assert goal == meta["goal"]["coord"]

        coord_to_pair, portal_pairs = _parse_portals(g)
        assert len(portal_pairs) == len(meta["portals"])

        physical_dist = _dist_without_teleport(g, start)
        assert goal not in physical_dist, "walls must fully disconnect the red endpoints without portals"

        cell_path = _unique_shortest_path(g, start, goal)
        assert _teleport_count(cell_path) >= meta["min_teleports"], "route must actually rely on portals"

        out = main(g)
        _check_path_paints_only_black(g, out, cell_path)
        assert int(np.count_nonzero(out == YELLOW)) >= 2, "generated route should visibly recolor multiple cells"

        nodes, adj = _build_abstract_graph(meta)
        dist, ways = _dijkstra_unique(adj, "S", "G")
        assert ways["G"] == 1, "abstract shortest route must be unique"
        abstract_path = _reconstruct_abstract_path(adj, dist, "S", "G")
        assert dist["G"] == len(cell_path) - 1, "abstract and grid path lengths must agree"

        routes = _enumerate_simple_routes(adj, "S", "G")
        routes = [route for route in routes if route["teleports"]]
        assert len(routes) >= 2, "need at least one longer alternative portal route"
        best_cost = min(route["cost"] for route in routes)
        best_routes = [route for route in routes if route["cost"] == best_cost]
        assert len(best_routes) == 1, "best abstract route must be unique"

        special_positions = _special_positions(meta)
        for a, b in zip(cell_path, cell_path[1:]):
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1:
                if a[0] == b[0]:
                    row = a[0]
                    lo, hi = sorted((a[1], b[1]))
                    for col in range(lo + 1, hi):
                        assert (row, col) not in special_positions

        best_portals = best_routes[0]["teleports"]
        assert best_portals, "shortest route must use at least one portal"
        optimal_first = best_portals[0]

        greedy_trap = False
        for color, (a, b) in portal_pairs.items():
            first_endpoint = a if abs(a[0] - start[0]) + abs(a[1] - start[1]) <= abs(b[0] - start[0]) + abs(b[1] - start[1]) else b
            entry_dist = abs(first_endpoint[0] - start[0]) + abs(first_endpoint[1] - start[1])
            for route in routes:
                if route["teleports"][0] == color and route["cost"] > best_cost:
                    optimal_entry = min(
                        abs(endpoint[0] - start[0]) + abs(endpoint[1] - start[1])
                        for endpoint in portal_pairs[optimal_first]
                    )
                    if entry_dist < optimal_entry:
                        greedy_trap = True
        if meta["require_greedy_trap"]:
            assert greedy_trap, "variant requires a nearer first portal that is globally worse"

        assert tuple(abstract_path) == best_routes[0]["path"], "reconstructed abstract route must match the unique best route"
        return True
    except AssertionError:
        return False


def _sample_chamber_widths(rng: random.Random, chamber_count: int, min_needed: list[int]) -> list[int] | None:
    widths = [rng.randint(max(4, need), 5) for need in min_needed]
    if sum(widths) + chamber_count - 1 > 30:
        return None
    return widths


def _sample_rows(rng: random.Random, height: int, chamber_count: int) -> list[int]:
    usable = list(range(1, height - 1))
    if len(usable) >= chamber_count:
        return rng.sample(usable, chamber_count)
    return [rng.choice(usable) for _ in range(chamber_count)]


def _ordered_cols(rng: random.Random, left: int, right: int, count: int) -> list[int]:
    cols = sorted(rng.sample(list(range(left, right + 1)), count))
    return cols


def _build_blank_grid(height: int, widths: list[int]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    total_width = sum(widths) + len(widths) - 1
    grid = np.zeros((height, total_width), dtype=int)
    spans: list[tuple[int, int]] = []
    col = 0
    for index, width in enumerate(widths):
        spans.append((col, col + width - 1))
        col += width
        if index + 1 < len(widths):
            grid[:, col] = BLUE
            col += 1
    return grid, spans


def _make_meta(height: int, widths: list[int], rows: list[int], variant: str, rng: random.Random) -> tuple[np.ndarray, dict] | None:
    chamber_count = 4 if variant != "three_step_chain" else 5
    need = {
        "two_step_trap": [3, 2, 2, 3],
        "three_step_chain": [3, 2, 2, 2, 3],
        "backtrack_chain": [3, 3, 2, 2],
    }[variant]
    if len(widths) != chamber_count:
        return None

    grid, spans = _build_blank_grid(height, widths)
    colors = list(rng.sample(PORTAL_COLORS, {"two_step_trap": 4, "three_step_chain": 5, "backtrack_chain": 4}[variant]))

    if variant == "two_step_trap":
        c0 = _ordered_cols(rng, *spans[0], 3)
        c1 = _ordered_cols(rng, *spans[1], 2)
        c2 = _ordered_cols(rng, *spans[2], 2)
        c3 = _ordered_cols(rng, *spans[3], 3)
        start = {"coord": (rows[0], c0[0]), "chamber": 0}
        goal = {"coord": (rows[3], c3[2]), "chamber": 3}
        portals = [
            {"color": colors[0], "a": (rows[0], c0[2]), "a_chamber": 0, "b": (rows[1], c1[0]), "b_chamber": 1},
            {"color": colors[1], "a": (rows[1], c1[1]), "a_chamber": 1, "b": (rows[3], c3[1]), "b_chamber": 3},
            {"color": colors[2], "a": (rows[0], c0[1]), "a_chamber": 0, "b": (rows[2], c2[0]), "b_chamber": 2},
            {"color": colors[3], "a": (rows[2], c2[1]), "a_chamber": 2, "b": (rows[3], c3[0]), "b_chamber": 3},
        ]
        min_teleports = 2
        require_greedy_trap = True
    elif variant == "three_step_chain":
        c0 = _ordered_cols(rng, *spans[0], 3)
        c1 = _ordered_cols(rng, *spans[1], 2)
        c2 = _ordered_cols(rng, *spans[2], 2)
        c3 = _ordered_cols(rng, *spans[3], 2)
        c4 = _ordered_cols(rng, *spans[4], 3)
        start = {"coord": (rows[0], c0[0]), "chamber": 0}
        goal = {"coord": (rows[4], c4[2]), "chamber": 4}
        portals = [
            {"color": colors[0], "a": (rows[0], c0[2]), "a_chamber": 0, "b": (rows[1], c1[0]), "b_chamber": 1},
            {"color": colors[1], "a": (rows[1], c1[1]), "a_chamber": 1, "b": (rows[2], c2[0]), "b_chamber": 2},
            {"color": colors[2], "a": (rows[2], c2[1]), "a_chamber": 2, "b": (rows[4], c4[1]), "b_chamber": 4},
            {"color": colors[3], "a": (rows[0], c0[1]), "a_chamber": 0, "b": (rows[3], c3[0]), "b_chamber": 3},
            {"color": colors[4], "a": (rows[3], c3[1]), "a_chamber": 3, "b": (rows[4], c4[0]), "b_chamber": 4},
        ]
        min_teleports = 3
        require_greedy_trap = True
    else:
        c0 = _ordered_cols(rng, *spans[0], 3)
        c1 = _ordered_cols(rng, *spans[1], 3)
        c2 = _ordered_cols(rng, *spans[2], 2)
        c3 = _ordered_cols(rng, *spans[3], 2)
        start = {"coord": (rows[0], c0[0]), "chamber": 0}
        goal = {"coord": (rows[3], c3[1]), "chamber": 3}
        portals = [
            {"color": colors[0], "a": (rows[0], c0[2]), "a_chamber": 0, "b": (rows[2], c2[0]), "b_chamber": 2},
            {"color": colors[1], "a": (rows[2], c2[1]), "a_chamber": 2, "b": (rows[1], c1[1]), "b_chamber": 1},
            {"color": colors[2], "a": (rows[1], c1[2]), "a_chamber": 1, "b": (rows[3], c3[0]), "b_chamber": 3},
            {"color": colors[3], "a": (rows[0], c0[1]), "a_chamber": 0, "b": (rows[1], c1[0]), "b_chamber": 1},
        ]
        min_teleports = 2
        require_greedy_trap = True

    grid[start["coord"]] = RED
    grid[goal["coord"]] = RED
    for portal in portals:
        grid[portal["a"]] = portal["color"]
        grid[portal["b"]] = portal["color"]

    meta = {
        "variant": variant,
        "start": start,
        "goal": goal,
        "portals": portals,
        "rows": rows,
        "widths": widths,
        "min_teleports": min_teleports,
        "require_greedy_trap": require_greedy_trap,
    }
    return grid, meta


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    variant = rng.choices(GEN_VARIANTS, weights=GEN_WEIGHTS, k=1)[0]
    chamber_count = 4 if variant != "three_step_chain" else 5
    min_needed = {
        "two_step_trap": [3, 2, 2, 3],
        "three_step_chain": [3, 2, 2, 2, 3],
        "backtrack_chain": [3, 3, 2, 2],
    }[variant]
    for _ in range(120):
        widths = _sample_chamber_widths(rng, chamber_count, min_needed)
        if widths is None:
            continue
        height = rng.randint(max(8, chamber_count + 3), 15)
        rows = _sample_rows(rng, height, chamber_count)
        built = _make_meta(height, widths, rows, variant, rng)
        if built is None:
            continue
        grid, meta = built
        if _validate_generated(grid, meta):
            return grid
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid portal-shortest-path input")
