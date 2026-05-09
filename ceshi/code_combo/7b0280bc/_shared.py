from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path

import numpy as np


TASK_ROOT = Path("/home/lixinyu/snap/new")


@lru_cache(maxsize=None)
def load_variant(dim_id: str, alt_index: int):
    path = TASK_ROOT / "ceshi" / "code" / "7b0280bc" / dim_id / str(alt_index) / "code.py"
    spec = importlib.util.spec_from_file_location(f"combo_7b0280bc_{dim_id}_{alt_index}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load variant module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_d5_prefix_problem(grid: np.ndarray) -> dict:
    """Parse the D5/0 graph and recover the prefix-selected path information."""
    d5 = load_variant("D5", 0)
    g = np.array(grid, dtype=int)
    bg = int(d5.get_bg(g))

    square_cands = [
        d for d in d5.find_solid_squares(g, bg=bg, min_size=2, connectivity=8)
        if int(d["color"]) != int(d5.Color.GREY)
    ]
    assert square_cands

    size_counts = d5.Counter(int(d["size"]) for d in square_cands)
    size_n = int(max(size_counts.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])
    mono_nodes_raw = [d for d in square_cands if int(d["size"]) == int(size_n)]

    by_color: dict[int, list[dict]] = d5.defaultdict(list)
    for d in mono_nodes_raw:
        by_color[int(d["color"])].append(d)

    terminal_color = None
    source_square = None
    encoded_square = None
    for col, solids in by_color.items():
        if len(solids) != 1:
            continue
        mixed = d5._find_encoded_terminal_components(g, int(col), size_n)
        if len(mixed) == 1:
            terminal_color = int(col)
            source_square = solids[0]
            encoded_square = mixed[0]
            break
    assert terminal_color is not None and source_square is not None and encoded_square is not None

    station_color = None
    best_station_count = -1
    for col, solids in by_color.items():
        if int(col) == int(terminal_color):
            continue
        if len(solids) > best_station_count:
            station_color = int(col)
            best_station_count = len(solids)
    assert station_color is not None and best_station_count > 0

    track_candidates = [
        c
        for c in d5.colors_present(g, exclude={bg})
        if int(c) not in (int(terminal_color), int(station_color), int(d5.Color.GREY))
    ]
    assert track_candidates
    track_color = int(max(track_candidates, key=lambda c: int((g == int(c)).sum())))

    nodes: list[dict] = []

    def add_node(kind: str, pts: set[tuple[int, int]], color: int, grey_count: int = 0):
        coords = sorted(pts)
        rs = [r for r, _ in coords]
        cs = [c for _, c in coords]
        nodes.append(
            {
                "id": int(len(nodes)),
                "kind": str(kind),
                "color": int(color),
                "pts": set(pts),
                "grey_count": int(grey_count),
                "bbox": (min(rs), min(cs), max(rs), max(cs)),
                "size": int(max(rs) - min(rs) + 1),
            }
        )

    add_node(
        "source",
        {(int(r), int(c)) for r, c in source_square["coords"]},
        int(terminal_color),
    )
    for d in by_color[int(station_color)]:
        add_node(
            "station",
            {(int(r), int(c)) for r, c in d["coords"]},
            int(station_color),
        )
    add_node(
        "encoded",
        set(encoded_square["pts"]),
        int(terminal_color),
        grey_count=int(encoded_square["grey_count"]),
    )

    source_id = next(int(nd["id"]) for nd in nodes if nd["kind"] == "source")
    encoded_id = next(int(nd["id"]) for nd in nodes if nd["kind"] == "encoded")
    stop_len = int(next(nd["grey_count"] for nd in nodes if nd["kind"] == "encoded"))

    pix2node: dict[tuple[int, int], int] = {}
    id2pts: dict[int, set[tuple[int, int]]] = {}
    id2kind: dict[int, str] = {}
    id2bbox: dict[int, tuple[int, int, int, int]] = {}
    id2size: dict[int, int] = {}
    for nd in nodes:
        nid = int(nd["id"])
        id2pts[nid] = set(nd["pts"])
        id2kind[nid] = str(nd["kind"])
        id2bbox[nid] = tuple(map(int, nd["bbox"]))
        id2size[nid] = int(nd["size"])
        for p in nd["pts"]:
            pix2node[p] = nid

    track_comps = d5.components_mask(g == int(track_color), connectivity=8)
    track_comp_pts: dict[int, list[tuple[int, int]]] = {}
    adj: dict[int, list[tuple[int, int]]] = d5.defaultdict(list)
    for edge_id, coords in enumerate(track_comps, start=1):
        pts = [(int(r), int(c)) for r, c in coords]
        if not pts:
            continue
        touched: set[int] = set()
        for r, c in pts:
            for rr, cc in d5.neighbors8(int(r), int(c)):
                nid = pix2node.get((int(rr), int(cc)))
                if nid is not None:
                    touched.add(int(nid))
        if len(touched) != 2:
            continue
        a, b = sorted(touched)
        track_comp_pts[int(edge_id)] = pts
        adj[int(a)].append((int(b), int(edge_id)))
        adj[int(b)].append((int(a), int(edge_id)))

    q = d5.deque([source_id])
    prev: dict[int, int | None] = {source_id: None}
    prev_edge: dict[int, int | None] = {source_id: None}
    while q:
        u = int(q.popleft())
        if u == encoded_id:
            break
        for v, eid in adj.get(u, []):
            if int(v) in prev:
                continue
            prev[int(v)] = int(u)
            prev_edge[int(v)] = int(eid)
            q.append(int(v))
    assert encoded_id in prev

    node_path: list[int] = []
    edge_path: list[int] = []
    cur = int(encoded_id)
    while cur is not None:
        node_path.append(int(cur))
        e = prev_edge.get(int(cur))
        if e is not None:
            edge_path.append(int(e))
        cur2 = prev.get(int(cur))
        cur = int(cur2) if cur2 is not None else None
    node_path.reverse()
    edge_path.reverse()

    return {
        "grid": g,
        "bg": bg,
        "stop_len": int(stop_len),
        "node_path": node_path,
        "edge_path": edge_path,
        "id2pts": id2pts,
        "id2kind": id2kind,
        "id2bbox": id2bbox,
        "id2size": id2size,
        "track_comp_pts": track_comp_pts,
    }


def parse_base_shortest_path(grid: np.ndarray, *, oriented_start: bool = False) -> dict:
    """Parse the base 7b0280bc graph and recover its unique shortest path."""
    base = load_variant("D3", 0)
    g = np.array(grid, dtype=int)
    bg = int(base.get_bg(g))

    square_cands = base.find_solid_squares(g, bg=bg, min_size=2, connectivity=8)
    assert square_cands
    size_counts = base.Counter(int(d["size"]) for d in square_cands)
    size_n = int(max(size_counts.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])
    nodes_raw = [d for d in square_cands if int(d["size"]) == int(size_n)]

    nodes: list[dict] = []
    for i, d in enumerate(nodes_raw):
        pts = {(int(r), int(c)) for r, c in d["coords"]}
        nodes.append(
            {
                "id": int(i),
                "color": int(d["color"]),
                "pts": pts,
                "bbox": tuple(map(int, d["bbox"])),
                "size": int(d["size"]),
                "anchor": min(pts),
            }
        )

    nodes_by_color: dict[int, list[int]] = base.defaultdict(list)
    for nd in nodes:
        nodes_by_color[int(nd["color"])].append(int(nd["id"]))

    terminal_color = None
    station_color = None
    for col in sorted(nodes_by_color):
        if len(nodes_by_color[int(col)]) == 2:
            terminal_color = int(col)
        else:
            station_color = int(col)
    assert terminal_color is not None and station_color is not None
    term_ids = list(map(int, nodes_by_color[int(terminal_color)]))

    non_bg_colors = base.colors_present(g, exclude={bg})
    track_candidates = [c for c in non_bg_colors if int(c) not in (int(terminal_color), int(station_color))]
    assert track_candidates
    track_color = int(max(track_candidates, key=lambda c: int((g == int(c)).sum())))

    pix2node: dict[tuple[int, int], int] = {}
    id2pts: dict[int, set[tuple[int, int]]] = {}
    id2bbox: dict[int, tuple[int, int, int, int]] = {}
    id2size: dict[int, int] = {}
    id2anchor: dict[int, tuple[int, int]] = {}
    for nd in nodes:
        nid = int(nd["id"])
        id2pts[nid] = set(nd["pts"])
        id2bbox[nid] = tuple(map(int, nd["bbox"]))
        id2size[nid] = int(nd["size"])
        id2anchor[nid] = tuple(map(int, nd["anchor"]))
        for p in nd["pts"]:
            pix2node[p] = nid

    track_comps = base.components_mask(g == int(track_color), connectivity=8)
    track_comp_pts: dict[int, list[tuple[int, int]]] = {}
    adj: dict[int, list[tuple[int, int]]] = base.defaultdict(list)
    for edge_id, coords in enumerate(track_comps, start=1):
        pts = [(int(r), int(c)) for r, c in coords]
        if not pts:
            continue
        touched: set[int] = set()
        for r, c in pts:
            for rr, cc in base.neighbors8(int(r), int(c)):
                nid = pix2node.get((int(rr), int(cc)))
                if nid is not None:
                    touched.add(int(nid))
        if len(touched) != 2:
            continue
        a, b = sorted(touched)
        track_comp_pts[int(edge_id)] = pts
        adj[int(a)].append((int(b), int(edge_id)))
        adj[int(b)].append((int(a), int(edge_id)))

    if oriented_start:
        start, goal = sorted(term_ids, key=lambda nid: id2anchor[int(nid)])
    else:
        start, goal = map(int, term_ids)

    q = base.deque([int(start)])
    prev: dict[int, int | None] = {int(start): None}
    prev_edge: dict[int, int | None] = {int(start): None}
    while q:
        u = int(q.popleft())
        if u == int(goal):
            break
        for v, eid in adj.get(u, []):
            if int(v) in prev:
                continue
            prev[int(v)] = int(u)
            prev_edge[int(v)] = int(eid)
            q.append(int(v))
    assert int(goal) in prev

    node_path: list[int] = []
    edge_path: list[int] = []
    cur = int(goal)
    while cur is not None:
        node_path.append(int(cur))
        e = prev_edge.get(int(cur))
        if e is not None:
            edge_path.append(int(e))
        cur2 = prev.get(int(cur))
        cur = int(cur2) if cur2 is not None else None
    node_path.reverse()
    edge_path.reverse()

    return {
        "grid": g,
        "bg": bg,
        "node_path": node_path,
        "edge_path": edge_path,
        "id2pts": id2pts,
        "id2bbox": id2bbox,
        "id2size": id2size,
        "track_comp_pts": track_comp_pts,
        "term_ids": list(map(int, term_ids)),
    }


def parse_source_goal_shortest_path(grid: np.ndarray) -> dict:
    """Parse the D6/3 source-goal graph and recover its unique shortest path."""
    d63 = load_variant("D6", 3)
    g = np.array(grid, dtype=int)
    bg = int(d63.get_bg(g))

    square_cands = d63.find_solid_squares(g, bg=bg, min_size=2, connectivity=8)
    nodes, start, goal, station_color = d63._select_nodes(square_cands)

    non_bg_colors = d63.colors_present(g, exclude={bg})
    track_candidates = [
        c for c in non_bg_colors
        if int(c) not in (int(d63.SOURCE_INPUT_COLOR), int(d63.GOAL_INPUT_COLOR), int(station_color))
    ]
    assert track_candidates
    track_color = int(max(track_candidates, key=lambda c: int((g == int(c)).sum())))

    pix2node: dict[tuple[int, int], int] = {}
    id2pts: dict[int, set[tuple[int, int]]] = {}
    id2kind: dict[int, str] = {}
    for nd in nodes:
        nid = int(nd["id"])
        pts = set(nd["pts"])
        id2pts[nid] = pts
        if int(nd["color"]) == int(d63.SOURCE_INPUT_COLOR):
            id2kind[nid] = "source"
        elif int(nd["color"]) == int(d63.GOAL_INPUT_COLOR):
            id2kind[nid] = "goal"
        else:
            id2kind[nid] = "station"
        for p in pts:
            pix2node[p] = nid

    track_comps = d63.components_mask(g == int(track_color), connectivity=8)
    track_comp_pts: dict[int, list[tuple[int, int]]] = {}
    edge_endpoints: dict[int, tuple[int, int]] = {}
    adj: dict[int, list[tuple[int, int]]] = d63.defaultdict(list)
    for edge_id, coords in enumerate(track_comps, start=1):
        pts = [(int(r), int(c)) for r, c in coords]
        if not pts:
            continue
        touched: set[int] = set()
        for r, c in pts:
            for rr, cc in d63.neighbors8(int(r), int(c)):
                nid = pix2node.get((int(rr), int(cc)))
                if nid is not None:
                    touched.add(int(nid))
        if len(touched) != 2:
            continue
        a, b = sorted(touched)
        track_comp_pts[int(edge_id)] = pts
        edge_endpoints[int(edge_id)] = (int(a), int(b))
        adj[int(a)].append((int(b), int(edge_id)))
        adj[int(b)].append((int(a), int(edge_id)))

    q = d63.deque([int(start)])
    prev: dict[int, int | None] = {int(start): None}
    prev_edge: dict[int, int | None] = {int(start): None}
    while q:
        u = int(q.popleft())
        if u == int(goal):
            break
        for v, eid in adj.get(u, []):
            if int(v) in prev:
                continue
            prev[int(v)] = int(u)
            prev_edge[int(v)] = int(eid)
            q.append(int(v))
    assert int(goal) in prev

    node_path: list[int] = []
    edge_path: list[int] = []
    cur = int(goal)
    while cur is not None:
        node_path.append(int(cur))
        e = prev_edge.get(int(cur))
        if e is not None:
            edge_path.append(int(e))
        cur2 = prev.get(int(cur))
        cur = int(cur2) if cur2 is not None else None
    node_path.reverse()
    edge_path.reverse()

    return {
        "grid": g,
        "node_path": node_path,
        "edge_path": edge_path,
        "id2pts": id2pts,
        "id2kind": id2kind,
        "track_comp_pts": track_comp_pts,
        "edge_endpoints": edge_endpoints,
    }
