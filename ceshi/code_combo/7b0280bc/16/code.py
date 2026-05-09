# Combo variant of 7b0280bc (index 16)
# combo_spec: D3/0+D8/2+D13/0
# Name: 多对终端渐变色路径合并

from __future__ import annotations

from collections import Counter, defaultdict, deque
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve()
PARENT = HERE.parent.parent
if str(PARENT) not in sys.path:
    sys.path.append(str(PARENT))

from arc_utils import components_mask, get_bg, neighbors8, Color
from _shared import load_variant


_D13 = load_variant("D13", 0)
WARM_EDGE_COLORS = [int(Color.GREY), int(Color.GREEN), int(Color.BLUE)]
COOL_EDGE_COLORS = [2, 6, 8]
PAIR1_NODE_COLOR = int(Color.GREEN)
PAIR2_NODE_COLOR = 8
SHARED_COLOR = 4


def _parse_multi_pair_graph(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    square_cands = _D13.find_solid_squares(g, bg=bg, min_size=2, connectivity=8)
    assert square_cands

    size_counts = Counter(int(d["size"]) for d in square_cands)
    size_n = int(max(size_counts.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])
    nodes_raw = [d for d in square_cands if int(d["size"]) == int(size_n)]

    nodes: list[dict] = []
    for i, d in enumerate(nodes_raw):
        pts = {(int(r), int(c)) for r, c in d["coords"]}
        nodes.append({"id": int(i), "color": int(d["color"]), "pts": pts, "anchor": min(pts)})

    nodes_by_color: dict[int, list[int]] = defaultdict(list)
    for nd in nodes:
        nodes_by_color[int(nd["color"])].append(int(nd["id"]))

    terminal_pair_colors = sorted(int(col) for col, ids in nodes_by_color.items() if len(ids) == 2)
    station_color = next(int(col) for col, ids in nodes_by_color.items() if len(ids) != 2)
    assert len(terminal_pair_colors) == 2
    terminal_pairs = [
        tuple(sorted(map(int, nodes_by_color[int(col)]), key=lambda nid: nodes[int(nid)]["anchor"]))
        for col in terminal_pair_colors
    ]

    track_candidates = [
        int(c) for c in np.unique(g)
        if int(c) not in {bg, station_color, *terminal_pair_colors}
    ]
    assert track_candidates
    track_color = int(max(track_candidates, key=lambda c: int((g == int(c)).sum())))

    pix2node: dict[tuple[int, int], int] = {}
    id2pts: dict[int, set[tuple[int, int]]] = {}
    for nd in nodes:
        nid = int(nd["id"])
        id2pts[nid] = set(nd["pts"])
        for p in nd["pts"]:
            pix2node[p] = nid

    track_comps = components_mask(g == int(track_color), connectivity=8)
    track_comp_pts: dict[int, list[tuple[int, int]]] = {}
    adj: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for edge_id, coords in enumerate(track_comps, start=1):
        pts = [(int(r), int(c)) for r, c in coords]
        if not pts:
            continue
        touched: set[int] = set()
        for r, c in pts:
            for rr, cc in neighbors8(int(r), int(c)):
                nid = pix2node.get((int(rr), int(cc)))
                if nid is not None:
                    touched.add(int(nid))
        if len(touched) != 2:
            continue
        a, b = sorted(touched)
        track_comp_pts[int(edge_id)] = pts
        adj[int(a)].append((int(b), int(edge_id)))
        adj[int(b)].append((int(a), int(edge_id)))

    return {
        "grid": g,
        "terminal_pairs": terminal_pairs,
        "id2pts": id2pts,
        "track_comp_pts": track_comp_pts,
        "adj": adj,
    }


def _shortest_path(adj: dict[int, list[tuple[int, int]]], start: int, goal: int) -> tuple[list[int], list[int]]:
    q = deque([int(start)])
    prev: dict[int, int | None] = {int(start): None}
    prev_edge: dict[int, int | None] = {int(start): None}
    while q:
        u = int(q.popleft())
        if int(u) == int(goal):
            break
        for v, eid in adj.get(int(u), []):
            if int(v) in prev:
                continue
            prev[int(v)] = int(u)
            prev_edge[int(v)] = int(eid)
            q.append(int(v))
    assert int(goal) in prev
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor both terminal-pair paths with separate ordered edge palettes.

    Rule:
    - The two same-colored terminal pairs still define two shortest paths.
    - Pair 1 uses a warm ordered palette, pair 2 uses a cool ordered palette.
    - Shared nodes or edges, if any, use a dedicated mixed color.
    """
    info = _parse_multi_pair_graph(np.asarray(grid, dtype=int))
    out = info["grid"].copy()

    pair_node_paths: list[list[int]] = []
    pair_edge_paths: list[list[int]] = []
    for start, goal in info["terminal_pairs"]:
        node_path, edge_path = _shortest_path(info["adj"], int(start), int(goal))
        pair_node_paths.append(node_path)
        pair_edge_paths.append(edge_path)

    node_counts: Counter[int] = Counter()
    edge_counts: Counter[int] = Counter()
    for path in pair_node_paths:
        for nid in path[1:-1]:
            node_counts[int(nid)] += 1
    for path in pair_edge_paths:
        for eid in path:
            edge_counts[int(eid)] += 1

    node_pair_colors = [PAIR1_NODE_COLOR, PAIR2_NODE_COLOR]
    edge_pair_palettes = [WARM_EDGE_COLORS, COOL_EDGE_COLORS]

    for pair_idx, node_path in enumerate(pair_node_paths):
        for nid in node_path[1:-1]:
            color = SHARED_COLOR if int(node_counts[int(nid)]) > 1 else int(node_pair_colors[pair_idx])
            for r, c in info["id2pts"][int(nid)]:
                out[int(r), int(c)] = int(color)

    for pair_idx, edge_path in enumerate(pair_edge_paths):
        palette = edge_pair_palettes[pair_idx]
        for edge_idx, eid in enumerate(edge_path):
            color = SHARED_COLOR if int(edge_counts[int(eid)]) > 1 else int(palette[min(edge_idx, len(palette) - 1)])
            for r, c in info["track_comp_pts"][int(eid)]:
                out[int(r), int(c)] = int(color)
    return out


def generate_input() -> np.ndarray:
    return np.asarray(_D13.generate_input(), dtype=int)
