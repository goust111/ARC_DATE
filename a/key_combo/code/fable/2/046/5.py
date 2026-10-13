from __future__ import annotations

import heapq
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, components_mask, get_bg


_BG_CHOICES = [Color.BLACK, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
_LAYOUTS = [
    (2, 1, 1, 5),
    (2, 1, 2, 6),
    (3, 1, 1, 4),
    (3, 1, 1, 5),
    (3, 1, 2, 5),
    (3, 1, 2, 6),
    (3, 2, 1, 5),
    (3, 2, 2, 6),
]
_MAX_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _square_components(grid: np.ndarray, color: int) -> list[dict]:
    comps = components_mask(grid == int(color), connectivity=4)
    out = []
    for coords in comps:
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        assert h == w >= 2, "node components must be solid squares"
        assert int(coords.shape[0]) == h * w, "node components must be filled squares"
        out.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "coords": coords.astype(int, copy=True),
            }
        )
    out.sort(key=lambda item: item["bbox"])
    return out


def _extract_graph(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    terminals = _square_components(g, Color.RED)
    stations = _square_components(g, Color.BLUE)
    assert len(terminals) == 2, "expected exactly two terminal squares"
    assert stations, "expected at least one station square"

    nodes = []
    for kind, comps in (("terminal", terminals), ("station", stations)):
        for comp in comps:
            nodes.append(
                {
                    "kind": kind,
                    "bbox": comp["bbox"],
                    "coords": comp["coords"],
                }
            )
    nodes.sort(key=lambda item: item["bbox"])

    pix_to_node: dict[tuple[int, int], int] = {}
    for node_id, node in enumerate(nodes):
        node["id"] = node_id
        for r, c in node["coords"]:
            pix_to_node[(int(r), int(c))] = int(node_id)

    edges = []
    adj: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for coords in components_mask(g == int(Color.GREY), connectivity=4):
        pts = [(int(r), int(c)) for r, c in coords]
        touched = set()
        for r, c in pts:
            for nr, nc in _neighbors4(int(r), int(c)):
                node_id = pix_to_node.get((int(nr), int(nc)))
                if node_id is not None:
                    touched.add(int(node_id))
        assert len(touched) == 2, "each track component must touch exactly two nodes"
        a, b = sorted(touched)
        edge_id = len(edges)
        edge = {
            "id": int(edge_id),
            "nodes": (int(a), int(b)),
            "coords": np.asarray(pts, dtype=int),
            "weight": int(len(pts)),
        }
        edges.append(edge)
        adj[int(a)].append((int(b), int(edge_id), int(len(pts))))
        adj[int(b)].append((int(a), int(edge_id), int(len(pts))))

    term_ids = [int(node["id"]) for node in nodes if node["kind"] == "terminal"]
    assert len(term_ids) == 2 and edges, "graph must contain terminals and tracks"
    return {
        "bg": bg,
        "nodes": nodes,
        "edges": edges,
        "adj": dict(adj),
        "term_ids": term_ids,
    }


def _dijkstra_counts(adj: dict[int, list[tuple[int, int, int]]], start: int) -> tuple[dict[int, int], dict[int, int]]:
    dist = {int(start): 0}
    count = {int(start): 1}
    heap = [(0, int(start))]
    while heap:
        cur_dist, node = heapq.heappop(heap)
        if cur_dist != dist[node]:
            continue
        for nxt, _edge_id, weight in adj.get(node, []):
            new_dist = int(cur_dist + weight)
            if nxt not in dist or new_dist < dist[nxt]:
                dist[int(nxt)] = int(new_dist)
                count[int(nxt)] = int(count[node])
                heapq.heappush(heap, (int(new_dist), int(nxt)))
            elif new_dist == dist[nxt]:
                count[int(nxt)] = int(count[nxt] + count[node])
    return dist, count


def _weighted_shortest_route(graph: dict) -> tuple[list[int], list[int], int]:
    start, goal = map(int, graph["term_ids"])
    adj = graph["adj"]
    dist_start, count_start = _dijkstra_counts(adj, start)
    assert goal in dist_start, "terminals must be connected"
    assert count_start[goal] == 1, "weighted shortest route must be unique"
    dist_goal, _ = _dijkstra_counts(adj, goal)

    total = int(dist_start[goal])
    node_path = [int(start)]
    edge_path = []
    prev = None
    cur = int(start)

    while cur != goal:
        options = []
        for nxt, edge_id, weight in adj.get(cur, []):
            if nxt not in dist_start or nxt not in dist_goal:
                continue
            if dist_start.get(nxt) != dist_start[cur] + weight:
                continue
            if dist_start[cur] + weight + dist_goal[nxt] != total:
                continue
            options.append((int(nxt), int(edge_id)))
        if prev is not None:
            forward = [option for option in options if option[0] != prev]
            if forward:
                options = forward
        assert len(options) == 1, "expected one forward edge on the unique shortest route"
        nxt, edge_id = options[0]
        edge_path.append(int(edge_id))
        node_path.append(int(nxt))
        prev, cur = int(cur), int(nxt)

    return node_path, edge_path, total


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the unique mileage-shortest terminal route in a square-and-track graph.

    Input structure:
    - The background is the most common color.
    - Exactly two red solid squares are terminals, and the blue solid squares are station nodes.
    - Grey 1-cell-wide track components each connect exactly two squares, forming a graph whose edge weight is the number of grey cells in that component.

    Rule:
    - Build the graph induced by the red and blue node squares and the grey track components.
    - Among all terminal-to-terminal routes, choose the unique route whose total grey-cell count is minimal; this is not necessarily the route with the fewest track segments.
    - Recolor every blue station square on that winning route to green and every grey track cell on that route to yellow, leaving terminals, off-route stations, off-route tracks, and background unchanged.

    Key insight:
    - The picture is a weighted graph: each track segment's mileage is its pixel count, so splitting a long jump through extra stations can beat a route with fewer hops.
    """
    graph = _extract_graph(np.asarray(grid, dtype=int))
    node_path, edge_path, _ = _weighted_shortest_route(graph)

    out = np.asarray(grid, dtype=int).copy()
    terminal_ids = set(map(int, graph["term_ids"]))

    for node_id in node_path:
        if node_id in terminal_ids:
            continue
        coords = graph["nodes"][int(node_id)]["coords"]
        out[coords[:, 0], coords[:, 1]] = int(Color.GREEN)

    for edge_id in edge_path:
        coords = graph["edges"][int(edge_id)]["coords"]
        out[coords[:, 0], coords[:, 1]] = int(Color.YELLOW)

    return out


def _track_cells_between(a: tuple[int, int], b: tuple[int, int], size: int) -> list[tuple[int, int]]:
    ar, ac = map(int, a)
    br, bc = map(int, b)
    anchor = (int(size) - 1) // 2
    if ar == br:
        row = int(ar + anchor)
        c0 = int(min(ac, bc) + size)
        c1 = int(max(ac, bc) - 1)
        assert c0 <= c1, "horizontal neighbors must leave a track gap"
        return [(row, c) for c in range(c0, c1 + 1)]
    assert ac == bc, "generator only uses axis-aligned edges"
    col = int(ac + anchor)
    r0 = int(min(ar, br) + size)
    r1 = int(max(ar, br) - 1)
    assert r0 <= r1, "vertical neighbors must leave a track gap"
    return [(r, col) for r in range(r0, r1 + 1)]


def _enumerate_paths(adj: dict[str, dict[str, int]], start: str, goal: str) -> list[list[str]]:
    out: list[list[str]] = []

    def dfs(node: str, seen: set[str], path: list[str]) -> None:
        if node == goal:
            out.append(path.copy())
            return
        for nxt in sorted(adj[node]):
            if nxt in seen:
                continue
            seen.add(nxt)
            path.append(nxt)
            dfs(nxt, seen, path)
            path.pop()
            seen.remove(nxt)

    dfs(start, {start}, [start])
    return out


def _path_weight(path: list[str], adj: dict[str, dict[str, int]]) -> int:
    return int(sum(adj[path[i]][path[i + 1]] for i in range(len(path) - 1)))


def _apply_random_transform(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k = random.randrange(4)
    inp2 = np.rot90(inp, k)
    out2 = np.rot90(out, k)
    if random.choice([False, True]):
        inp2 = np.fliplr(inp2)
        out2 = np.fliplr(out2)
    if random.choice([False, True]):
        inp2 = np.flipud(inp2)
        out2 = np.flipud(out2)
    return inp2.copy(), out2.copy()


def _render_case(
    nodes: dict[str, dict[str, int]],
    edges: list[tuple[str, str]],
    size: int,
    bg: int,
    winning_path: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    occupied_nodes = set()
    for info in nodes.values():
        r0 = int(info["r"])
        c0 = int(info["c"])
        for r in range(r0, r0 + size):
            for c in range(c0, c0 + size):
                cell = (int(r), int(c))
                assert cell not in occupied_nodes, "node squares may not overlap"
                occupied_nodes.add(cell)

    edge_pixels: dict[frozenset[str], list[tuple[int, int]]] = {}
    all_tracks = set()
    for a, b in edges:
        coords = _track_cells_between((nodes[a]["r"], nodes[a]["c"]), (nodes[b]["r"], nodes[b]["c"]), size)
        key = frozenset((a, b))
        assert key not in edge_pixels, "parallel duplicate edges are not allowed"
        edge_pixels[key] = coords
        for cell in coords:
            assert cell not in occupied_nodes, "tracks may not overlap nodes"
            assert cell not in all_tracks, "tracks may not overlap each other"
            all_tracks.add(cell)

    mins = [
        min(info["r"] for info in nodes.values()),
        min(info["c"] for info in nodes.values()),
    ]
    maxs = [
        max(info["r"] + size - 1 for info in nodes.values()),
        max(info["c"] + size - 1 for info in nodes.values()),
    ]
    if all_tracks:
        min_tr = min(r for r, _ in all_tracks)
        min_tc = min(c for _, c in all_tracks)
        max_tr = max(r for r, _ in all_tracks)
        max_tc = max(c for _, c in all_tracks)
        mins[0] = min(mins[0], int(min_tr))
        mins[1] = min(mins[1], int(min_tc))
        maxs[0] = max(maxs[0], int(max_tr))
        maxs[1] = max(maxs[1], int(max_tc))

    base_h = int(maxs[0] - mins[0] + 1)
    base_w = int(maxs[1] - mins[1] + 1)
    if base_h > 28 or base_w > 28:
        raise AssertionError("base layout exceeds the ARC size budget")

    top_pad = 1
    left_pad = 1
    bottom_pad = min(2, 30 - (base_h + top_pad))
    right_pad = min(2, 30 - (base_w + left_pad))
    assert bottom_pad >= 1 and right_pad >= 1

    shift_r = int(top_pad - mins[0])
    shift_c = int(left_pad - mins[1])
    height = int(base_h + top_pad + bottom_pad)
    width = int(base_w + left_pad + right_pad)
    inp = np.full((height, width), int(bg), dtype=int)

    shifted_nodes = {}
    for name, info in nodes.items():
        nr = int(info["r"] + shift_r)
        nc = int(info["c"] + shift_c)
        shifted_nodes[name] = {"r": nr, "c": nc, "kind": info["kind"]}
        color = int(Color.RED if info["kind"] == "terminal" else Color.BLUE)
        inp[nr : nr + size, nc : nc + size] = color

    shifted_edges: dict[frozenset[str], np.ndarray] = {}
    for key, coords in edge_pixels.items():
        shifted = np.asarray([(r + shift_r, c + shift_c) for r, c in coords], dtype=int)
        inp[shifted[:, 0], shifted[:, 1]] = int(Color.GREY)
        shifted_edges[key] = shifted

    out = inp.copy()
    for name in winning_path[1:-1]:
        info = shifted_nodes[name]
        assert info["kind"] == "station"
        out[info["r"] : info["r"] + size, info["c"] : info["c"] + size] = int(Color.GREEN)
    for i in range(len(winning_path) - 1):
        key = frozenset((winning_path[i], winning_path[i + 1]))
        coords = shifted_edges[key]
        out[coords[:, 0], coords[:, 1]] = int(Color.YELLOW)

    return inp, out


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    bg = int(random.choice(_BG_CHOICES))
    size, vertical_gap, upper_count, bottom_count = random.choice(_LAYOUTS)

    bottom_x = [0]
    for _ in range(bottom_count - 1):
        bottom_x.append(int(bottom_x[-1] + size + random.randint(1, 2)))

    upper_x = sorted(random.sample(bottom_x[1:-1], upper_count))
    top_r = 0
    bottom_r = int(size + vertical_gap)

    nodes: dict[str, dict[str, int]] = {
        "T0": {"kind": "terminal", "r": int(top_r), "c": int(bottom_x[0])},
        "T1": {"kind": "terminal", "r": int(top_r), "c": int(bottom_x[-1])},
    }
    for i, col in enumerate(upper_x):
        nodes[f"U{i}"] = {"kind": "station", "r": int(top_r), "c": int(col)}
    for i, col in enumerate(bottom_x):
        nodes[f"B{i}"] = {"kind": "station", "r": int(bottom_r), "c": int(col)}

    upper_route = ["T0"] + [f"U{i}" for i in range(upper_count)] + ["T1"]
    lower_route = ["T0"] + [f"B{i}" for i in range(bottom_count)] + ["T1"]
    edges = list(zip(upper_route, upper_route[1:])) + list(zip(lower_route, lower_route[1:]))

    candidates = [name for name in nodes if name.startswith("U") or name.startswith("B")]
    random.shuffle(candidates)
    target_spurs = random.randint(0, min(3, len(candidates)))
    spur_count = 0
    for base in candidates:
        if spur_count >= target_spurs:
            break
        direction = -1 if nodes[base]["r"] == top_r else 1
        chain_len = random.choice([1, 1, 2])
        cur_r = int(nodes[base]["r"])
        col = int(nodes[base]["c"])
        proposed_rows = []
        for _ in range(chain_len):
            cur_r += direction * int(size + random.randint(1, 2))
            proposed_rows.append(int(cur_r))
        if any(
            info["r"] == row and info["c"] == col
            for row in proposed_rows
            for info in nodes.values()
        ):
            continue
        prev = base
        for step, row in enumerate(proposed_rows):
            name = f"S{spur_count}_{step}"
            nodes[name] = {"kind": "station", "r": int(row), "c": int(col)}
            edges.append((prev, name))
            prev = name
        spur_count += 1

    adj: dict[str, dict[str, int]] = defaultdict(dict)
    for a, b in edges:
        weight = len(_track_cells_between((nodes[a]["r"], nodes[a]["c"]), (nodes[b]["r"], nodes[b]["c"]), size))
        if weight <= 0:
            return None
        if b in adj[a] or a in adj[b]:
            return None
        adj[a][b] = int(weight)
        adj[b][a] = int(weight)

    path_infos = []
    for path in _enumerate_paths(adj, "T0", "T1"):
        path_infos.append({"nodes": path, "hops": len(path) - 1, "weight": _path_weight(path, adj)})

    if len(path_infos) < 2:
        return None

    best_weight = min(info["weight"] for info in path_infos)
    winners = [info for info in path_infos if info["weight"] == best_weight]
    if len(winners) != 1:
        return None
    winner = winners[0]

    decoys = [
        info
        for info in path_infos
        if info["hops"] < winner["hops"] and info["weight"] > winner["weight"]
    ]
    if not decoys:
        return None

    inp, out = _render_case(nodes, edges, size, bg, winner["nodes"])
    inp, out = _apply_random_transform(inp, out)

    counts = Counter(map(int, inp.ravel()))
    bg_count = int(counts[bg])
    if any(color != bg and int(count) >= bg_count for color, count in counts.items()):
        return None

    try:
        parsed = _extract_graph(inp)
        if len(parsed["nodes"]) != len(nodes) or len(parsed["edges"]) != len(edges):
            return None
        _, _, parsed_weight = _weighted_shortest_route(parsed)
        if parsed_weight != winner["weight"]:
            return None
        actual = main(inp)
    except Exception:
        return None

    if not np.array_equal(actual, out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        case = _build_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid weighted-shortest-path ARC input")
