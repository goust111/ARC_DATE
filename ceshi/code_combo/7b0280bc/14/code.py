# Combo variant of 7b0280bc (index 14)
# combo_spec: D4/2+D5/2+D13/2
# Name: 同色方块内嵌标记配对寻路

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve()
PARENT = HERE.parent.parent
if str(PARENT) not in sys.path:
    sys.path.append(str(PARENT))

from arc_utils import Color, colors_present, components_mask, get_bg, neighbors8
from _shared import load_variant


_D5 = load_variant("D5", 2)
COLOR_POOL = list(range(10))


def _parse_same_color_marker_graph(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    solid = _D5.find_solid_squares(g, bg=bg, min_size=3, connectivity=8)
    assert solid
    size_counts = Counter(int(d["size"]) for d in solid)
    N = int(max(size_counts.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])
    station_nodes = [d for d in solid if int(d["size"]) == int(N)]
    node_color = int(Counter(int(d["color"]) for d in station_nodes).most_common(1)[0][0])
    station_nodes = [d for d in station_nodes if int(d["color"]) == int(node_color)]

    occupied = np.zeros_like(g, dtype=bool)
    for d in station_nodes:
        r0, c0, r1, c1 = map(int, d["bbox"])
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True
    marked_nodes = _D5._find_marked_squares(g, N, bg, occupied)
    assert len(marked_nodes) >= 2
    assert all(int(d["color"]) == int(node_color) for d in marked_nodes)

    marker_counts = Counter(int(d["marker_color"]) for d in marked_nodes)
    pair_colors = [int(col) for col, cnt in marker_counts.items() if int(cnt) == 2]
    assert len(pair_colors) == 1
    pair_color = int(pair_colors[0])

    nodes: list[dict] = []

    def add_node(pts: set[tuple[int, int]], marker_color: int | None) -> int:
        nid = len(nodes)
        nodes.append(
            {
                "id": int(nid),
                "pts": set(pts),
                "marker_color": None if marker_color is None else int(marker_color),
            }
        )
        return int(nid)

    term_ids: list[int] = []
    for d in station_nodes:
        pts = {(int(r), int(c)) for r, c in d["coords"]}
        add_node(pts, None)
    for d in marked_nodes:
        pts = {(int(r), int(c)) for r, c in d["coords"]}
        nid = add_node(pts, int(d["marker_color"]))
        if int(d["marker_color"]) == int(pair_color):
            term_ids.append(int(nid))
    assert len(term_ids) == 2

    pix2node: dict[tuple[int, int], int] = {}
    id2pts: dict[int, set[tuple[int, int]]] = {}
    node_mask = np.zeros_like(g, dtype=bool)
    for nd in nodes:
        nid = int(nd["id"])
        id2pts[nid] = set(nd["pts"])
        for p in nd["pts"]:
            pix2node[p] = nid
            node_mask[p] = True

    non_bg = colors_present(g, exclude={bg})
    track_candidates = [int(c) for c in non_bg if np.any((g == int(c)) & (~node_mask))]
    assert len(track_candidates) == 1
    track_color = int(track_candidates[0])

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
        "term_ids": term_ids,
        "id2pts": id2pts,
        "track_comp_pts": track_comp_pts,
        "adj": adj,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the shortest path between the unique pair of same-marker squares.

    Input structure:
    - Every node square has the same body color.
    - Some squares have an embedded center marker.
    - The terminal pair is the unique marker color that appears exactly twice.
    """
    info = _parse_same_color_marker_graph(np.asarray(grid, dtype=int))
    g = info["grid"]
    start, goal = map(int, info["term_ids"])

    q = deque([int(start)])
    prev: dict[int, int | None] = {int(start): None}
    prev_edge: dict[int, int | None] = {int(start): None}
    while q:
        u = int(q.popleft())
        if u == int(goal):
            break
        for v, eid in info["adj"].get(u, []):
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
        parent = prev.get(int(cur))
        cur = int(parent) if parent is not None else None
    node_path.reverse()
    edge_path.reverse()

    out = g.copy()
    term_set = {int(start), int(goal)}
    for nid in node_path:
        if int(nid) in term_set:
            continue
        for r, c in info["id2pts"][int(nid)]:
            out[int(r), int(c)] = int(Color.GREEN)
    for eid in edge_path:
        for r, c in info["track_comp_pts"][int(eid)]:
            out[int(r), int(c)] = int(Color.GREY)
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        base = np.asarray(_D5.generate_input(), dtype=int)
        info = _D5._build_instance_graph(base)
        bg = int(info["bg"])
        used = {bg, int(info["track_color"])}
        available = [c for c in COLOR_POOL if int(c) not in used]
        needed = 2 + max(0, len(info["goal_candidate_ids"]) - 1)
        if len(available) < needed:
            continue
        rng.shuffle(available)
        body_color = int(available[0])
        pair_marker = int(available[1])
        distractor_markers = list(map(int, available[2:]))

        out = base.copy()
        distractor_i = 0
        for nd in info["nodes"]:
            r0, c0, r1, c1 = map(int, nd["bbox"])
            out[r0 : r1 + 1, c0 : c1 + 1] = int(body_color)
            role = str(nd["role"])
            if role == "start" or int(nd["id"]) == int(info["goal_id"]):
                out[r0 + 1, c0 + 1] = int(pair_marker)
            elif role == "goal":
                if distractor_i >= len(distractor_markers):
                    break
                out[r0 + 1, c0 + 1] = int(distractor_markers[distractor_i])
                distractor_i += 1
        else:
            try:
                _ = main(out)
            except Exception:
                continue
            return out
    raise RuntimeError("generate_input: failed to build a same-color marker pair sample")
