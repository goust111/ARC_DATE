# Combo variant of 7b0280bc (index 1)
# combo_spec: D1/1+D12/2
# Name: 轨道宽度加权最短路径

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np

from arc_utils import Color, colors_present, components_mask, get_bg, neighbors8


NODE_SIZE = 3
COLOR_POOL = [0, 1, 2, 4, 6, 7, 8, 9]


def _draw_node(grid: np.ndarray, r0: int, c0: int, color: int) -> None:
    grid[r0 : r0 + NODE_SIZE, c0 : c0 + NODE_SIZE] = int(color)


def _draw_h_edge(grid: np.ndarray, r0: int, c_left: int, c_right: int, width: int, color: int) -> None:
    row_start = int(r0 + (NODE_SIZE - width) // 2)
    grid[row_start : row_start + width, c_left + NODE_SIZE : c_right] = int(color)


def _draw_v_edge(grid: np.ndarray, c0: int, r_top: int, r_bottom: int, width: int, color: int) -> None:
    col_start = int(c0 + (NODE_SIZE - width) // 2)
    grid[r_top + NODE_SIZE : r_bottom, col_start : col_start + width] = int(color)


def _extract_graph(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    square_cands = []
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=8):
            if coords.size == 0:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            if h == NODE_SIZE and w == NODE_SIZE and int(coords.shape[0]) == NODE_SIZE * NODE_SIZE:
                square_cands.append(
                    {
                        "color": int(col),
                        "bbox": (int(r0), int(c0), int(r1), int(c1)),
                        "coords": coords,
                    }
                )
    assert square_cands

    nodes: list[dict] = []
    for i, d in enumerate(square_cands):
        pts = {(int(r), int(c)) for r, c in d["coords"]}
        nodes.append({"id": int(i), "color": int(d["color"]), "pts": pts})

    nodes_by_color: dict[int, list[int]] = defaultdict(list)
    for nd in nodes:
        nodes_by_color[int(nd["color"])].append(int(nd["id"]))
    terminal_color = next(int(col) for col, ids in nodes_by_color.items() if len(ids) == 2)
    station_color = next(int(col) for col, ids in nodes_by_color.items() if len(ids) != 2)
    term_ids = list(map(int, nodes_by_color[int(terminal_color)]))

    non_bg = colors_present(g, exclude={bg})
    track_color = next(int(c) for c in non_bg if int(c) not in {int(terminal_color), int(station_color)})

    pix2node: dict[tuple[int, int], int] = {}
    id2pts: dict[int, set[tuple[int, int]]] = {}
    for nd in nodes:
        nid = int(nd["id"])
        pts = set(nd["pts"])
        id2pts[nid] = pts
        for p in pts:
            pix2node[p] = nid

    track_comps = components_mask(g == int(track_color), connectivity=8)
    track_comp_pts: dict[int, list[tuple[int, int]]] = {}
    edge_weights: dict[int, int] = {}
    adj: dict[int, list[tuple[int, int, int]]] = defaultdict(list)

    for edge_id, coords in enumerate(track_comps, start=1):
        pts = [(int(r), int(c)) for r, c in coords]
        if not pts:
            continue
        touched: set[int] = set()
        rs = [int(r) for r, _ in pts]
        cs = [int(c) for _, c in pts]
        weight = int(min(max(rs) - min(rs) + 1, max(cs) - min(cs) + 1))
        for r, c in pts:
            for rr, cc in neighbors8(int(r), int(c)):
                nid = pix2node.get((int(rr), int(cc)))
                if nid is not None:
                    touched.add(int(nid))
        if len(touched) != 2:
            continue
        a, b = sorted(touched)
        track_comp_pts[int(edge_id)] = pts
        edge_weights[int(edge_id)] = int(weight)
        adj[int(a)].append((int(b), int(edge_id), int(weight)))
        adj[int(b)].append((int(a), int(edge_id), int(weight)))

    return {
        "grid": g,
        "bg": bg,
        "term_ids": term_ids,
        "id2pts": id2pts,
        "track_comp_pts": track_comp_pts,
        "edge_weights": edge_weights,
        "adj": adj,
    }


def _reconstruct(goal: int, prev: dict[int, int | None], prev_edge: dict[int, int | None]) -> tuple[list[int], list[int]]:
    node_path: list[int] = []
    edge_path: list[int] = []
    cur: int | None = int(goal)
    while cur is not None:
        node_path.append(int(cur))
        eid = prev_edge.get(int(cur))
        if eid is not None:
            edge_path.append(int(eid))
        par = prev.get(int(cur))
        cur = int(par) if par is not None else None
    node_path.reverse()
    edge_path.reverse()
    return node_path, edge_path


def best_path_by_weight(graph: dict) -> tuple[list[int], list[int], int]:
    start, goal = map(int, graph["term_ids"])
    frontier: list[tuple[int, int]] = [(0, int(start))]
    dist: dict[int, int] = {int(start): 0}
    prev: dict[int, int | None] = {int(start): None}
    prev_edge: dict[int, int | None] = {int(start): None}
    import heapq

    while frontier:
        cur_dist, u = heapq.heappop(frontier)
        if cur_dist != dist.get(int(u)):
            continue
        if int(u) == int(goal):
            break
        for v, eid, weight in graph["adj"].get(int(u), []):
            nd = int(cur_dist + int(weight))
            if nd < dist.get(int(v), 10**9):
                dist[int(v)] = nd
                prev[int(v)] = int(u)
                prev_edge[int(v)] = int(eid)
                heapq.heappush(frontier, (nd, int(v)))
    node_path, edge_path = _reconstruct(int(goal), prev, prev_edge)
    return node_path, edge_path, int(dist[int(goal)])


def best_path_by_edges(graph: dict) -> tuple[list[int], list[int], int]:
    start, goal = map(int, graph["term_ids"])
    q = deque([int(start)])
    prev: dict[int, int | None] = {int(start): None}
    prev_edge: dict[int, int | None] = {int(start): None}
    dist: dict[int, int] = {int(start): 0}
    while q:
        u = int(q.popleft())
        if int(u) == int(goal):
            break
        for v, eid, _weight in graph["adj"].get(int(u), []):
            if int(v) in prev:
                continue
            prev[int(v)] = int(u)
            prev_edge[int(v)] = int(eid)
            dist[int(v)] = int(dist[int(u)] + 1)
            q.append(int(v))
    node_path, edge_path = _reconstruct(int(goal), prev, prev_edge)
    return node_path, edge_path, int(dist[int(goal)])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the terminal path with minimum total track width.

    Rule:
    - Every edge weight is the visible ribbon width (1..3 pixels), not its geometric length.
    - Choose the terminal-to-terminal path with the smallest sum of edge widths.
    """
    graph = _extract_graph(np.asarray(grid, dtype=int))
    node_path, edge_path, _weight = best_path_by_weight(graph)
    out = graph["grid"].copy()
    for nid in node_path[1:-1]:
        for r, c in graph["id2pts"][int(nid)]:
            out[int(r), int(c)] = int(Color.GREEN)
    for eid in edge_path:
        for r, c in graph["track_comp_pts"][int(eid)]:
            out[int(r), int(c)] = int(Color.GREY)
    return out


def _build_base_instance(rng: random.Random) -> np.ndarray:
    bg, terminal_color, station_color, track_color = rng.sample(COLOR_POOL, 4)
    g = np.full((23, 23), int(bg), dtype=int)

    nodes = {
        "start": (10, 2),
        "goal": (10, 18),
        "mid": (10, 10),
        "top_l": (3, 2),
        "top_m": (3, 10),
        "top_r": (3, 18),
    }

    for name, (r0, c0) in nodes.items():
        color = terminal_color if name in {"start", "goal"} else station_color
        _draw_node(g, int(r0), int(c0), int(color))

    _draw_h_edge(g, 10, 2, 10, 3, track_color)
    _draw_h_edge(g, 10, 10, 18, 3, track_color)

    _draw_v_edge(g, 2, 3, 10, 1, track_color)
    _draw_h_edge(g, 3, 2, 10, 1, track_color)
    _draw_h_edge(g, 3, 10, 18, 1, track_color)
    _draw_v_edge(g, 18, 3, 10, 1, track_color)

    return g


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(200):
        g = _build_base_instance(rng)
        if rng.random() < 0.5:
            g = np.fliplr(g)
        if rng.random() < 0.5:
            g = np.flipud(g)
        graph = _extract_graph(g)
        try:
            _n_w, _e_w, total_w = best_path_by_weight(graph)
            _n_b, _e_b, edges_b = best_path_by_edges(graph)
        except Exception:
            continue
        if total_w == edges_b:
            continue
        try:
            _ = main(g)
        except Exception:
            continue
        return np.asarray(g, dtype=int)
    raise RuntimeError("generate_input: failed to build a width-weighted sample")
