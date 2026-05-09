# Combo variant of 7b0280bc (index 7)
# combo_spec: D2/1+D16/2
# Name: 方向约束有向图寻路

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

from arc_utils import Color, components_mask, colors_present, get_bg, neighbors8
from _shared import load_variant


_D2 = load_variant("D2", 1)
LIGHT_TRACK_COLOR = 4
DARK_TRACK_COLOR = 5


def _parse_dual_directed_graph(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    nodes, terminal_color, station_color, marker_color = _D2._detect_nodes(g, bg)
    term_ids = [int(nd["id"]) for nd in nodes if int(nd["fill_color"]) == int(terminal_color)]
    assert len(term_ids) == 2

    pix2node: dict[tuple[int, int], int] = {}
    id2pts: dict[int, set[tuple[int, int]]] = {}
    id2bbox: dict[int, tuple[int, int, int, int]] = {}
    id2dir: dict[int, str] = {}
    centers: dict[int, tuple[float, float]] = {}
    for nd in nodes:
        nid = int(nd["id"])
        id2pts[nid] = set(nd["pts"])
        id2bbox[nid] = tuple(int(v) for v in nd["bbox"])
        id2dir[nid] = str(nd["direction"])
        r0, c0, r1, c1 = id2bbox[nid]
        centers[nid] = ((r0 + r1) / 2.0, (c0 + c1) / 2.0)
        for p in nd["pts"]:
            pix2node[p] = nid

    track_mask = np.isin(g, [LIGHT_TRACK_COLOR, DARK_TRACK_COLOR])
    track_comps = components_mask(track_mask, connectivity=8)
    track_comp_pts: dict[int, list[tuple[int, int]]] = {}
    directed_adj: dict[int, list[tuple[int, int]]] = defaultdict(list)
    undirected_adj: dict[int, list[tuple[int, int]]] = defaultdict(list)
    edge_info: dict[int, dict] = {}

    for edge_id, coords in enumerate(track_comps, start=1):
        pts = [(int(r), int(c)) for r, c in coords]
        if not pts:
            continue

        touched: set[int] = set()
        side_hits: dict[int, set[str]] = defaultdict(set)
        endpoint_colors: dict[int, list[int]] = defaultdict(list)
        for r, c in pts:
            for rr, cc in neighbors8(int(r), int(c)):
                nid = pix2node.get((int(rr), int(cc)))
                if nid is None:
                    continue
                side = _D2._track_touch_side((int(r), int(c)), id2bbox[int(nid)])
                if side is None:
                    continue
                touched.add(int(nid))
                side_hits[int(nid)].add(str(side))
                endpoint_colors[int(nid)].append(int(g[int(r), int(c)]))

        if len(touched) != 2:
            continue
        if any(len(side_hits[int(nid)]) != 1 for nid in touched):
            continue

        a, b = sorted(touched)
        color_a = Counter(endpoint_colors[int(a)]).most_common(1)[0][0]
        color_b = Counter(endpoint_colors[int(b)]).most_common(1)[0][0]
        if {int(color_a), int(color_b)} != {LIGHT_TRACK_COLOR, DARK_TRACK_COLOR}:
            continue

        if int(color_a) == LIGHT_TRACK_COLOR:
            src, dst = int(a), int(b)
        else:
            src, dst = int(b), int(a)

        side_a = next(iter(side_hits[int(a)]))
        side_b = next(iter(side_hits[int(b)]))
        track_comp_pts[int(edge_id)] = pts
        undirected_adj[int(a)].append((int(b), int(edge_id)))
        undirected_adj[int(b)].append((int(a), int(edge_id)))
        edge_info[int(edge_id)] = {
            "endpoints": (int(a), int(b)),
            "sides": {int(a): str(side_a), int(b): str(side_b)},
            "oriented": (int(src), int(dst)),
        }

        if int(src) == int(a) and str(side_a) == str(id2dir[int(a)]):
            directed_adj[int(a)].append((int(b), int(edge_id)))
        if int(src) == int(b) and str(side_b) == str(id2dir[int(b)]):
            directed_adj[int(b)].append((int(a), int(edge_id)))

    return {
        "grid": g,
        "nodes": nodes,
        "term_ids": term_ids,
        "id2pts": id2pts,
        "id2bbox": id2bbox,
        "id2dir": id2dir,
        "centers": centers,
        "track_comp_pts": track_comp_pts,
        "directed_adj": directed_adj,
        "undirected_adj": undirected_adj,
        "edge_info": edge_info,
    }


def _find_terminal_candidate_paths(term_ids: list[int], adj: dict[int, list[tuple[int, int]]]) -> list[tuple[int, int, int, list[int], list[int]]]:
    candidate_paths: list[tuple[int, int, int, list[int], list[int]]] = []
    for start in term_ids:
        goal = term_ids[0] if int(start) == int(term_ids[1]) else term_ids[1]
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
        if int(goal) not in prev:
            continue
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
        candidate_paths.append((len(edge_path), int(start), int(goal), node_path, edge_path))
    return candidate_paths


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the unique terminal path allowed by both node-facing and edge-facing directions.
    """
    parsed = _parse_dual_directed_graph(np.asarray(grid, dtype=int))
    candidate_paths = _find_terminal_candidate_paths(parsed["term_ids"], parsed["directed_adj"])
    assert candidate_paths
    best_len = min(int(item[0]) for item in candidate_paths)
    best = [item for item in candidate_paths if int(item[0]) == best_len]
    assert len(best) == 1
    _path_len, start, goal, node_path, edge_path = best[0]

    out = parsed["grid"].copy()
    term_set = {int(start), int(goal)}
    for nid in node_path:
        if int(nid) in term_set:
            continue
        for r, c in parsed["id2pts"][int(nid)]:
            out[int(r), int(c)] = int(Color.GREEN)
    for eid in edge_path:
        for r, c in parsed["track_comp_pts"][int(eid)]:
            out[int(r), int(c)] = int(Color.GREY)
    return out


def _recolor_edge_by_orientation(
    grid: np.ndarray,
    pts: list[tuple[int, int]],
    src_center: tuple[float, float],
    dst_center: tuple[float, float],
) -> None:
    for r, c in pts:
        ds = abs(float(r) - float(src_center[0])) + abs(float(c) - float(src_center[1]))
        dd = abs(float(r) - float(dst_center[0])) + abs(float(c) - float(dst_center[1]))
        grid[int(r), int(c)] = int(LIGHT_TRACK_COLOR if ds <= dd else DARK_TRACK_COLOR)


def generate_input() -> np.ndarray:
    for _ in range(400):
        base = np.asarray(_D2.generate_input(), dtype=int)
        bg = int(get_bg(base))
        if LIGHT_TRACK_COLOR in set(map(int, np.unique(base))) or DARK_TRACK_COLOR in set(map(int, np.unique(base))):
            continue
        parsed = _D2._parse_graph(base)
        node_only_paths = _find_terminal_candidate_paths(parsed["term_ids"], parsed["directed_adj"])
        if not node_only_paths:
            continue
        node_only_paths.sort(key=lambda item: (int(item[0]), int(item[1]), int(item[2])))
        _best_len, _start, _goal, node_path, edge_path = node_only_paths[0]
        path_oriented = set(zip(node_path, node_path[1:], edge_path))

        # Re-parse edge endpoints/sides from the node-only directed sample using D2's original track color.
        g = base.copy()
        nodes, terminal_color, station_color, marker_color = _D2._detect_nodes(g, bg)
        track_color = next(
            int(c) for c in colors_present(g, exclude={bg})
            if int(c) not in {int(terminal_color), int(station_color), int(marker_color)}
        )
        pix2node: dict[tuple[int, int], int] = {}
        id2bbox: dict[int, tuple[int, int, int, int]] = {}
        id2dir: dict[int, str] = {}
        centers: dict[int, tuple[float, float]] = {}
        for nd in nodes:
            nid = int(nd["id"])
            id2bbox[nid] = tuple(int(v) for v in nd["bbox"])
            id2dir[nid] = str(nd["direction"])
            r0, c0, r1, c1 = id2bbox[nid]
            centers[nid] = ((r0 + r1) / 2.0, (c0 + c1) / 2.0)
            for p in nd["pts"]:
                pix2node[p] = nid

        edge_meta: dict[int, dict] = {}
        for edge_id, coords in enumerate(components_mask(g == int(track_color), connectivity=8), start=1):
            pts = [(int(r), int(c)) for r, c in coords]
            if not pts:
                continue
            touched: set[int] = set()
            side_hits: dict[int, set[str]] = defaultdict(set)
            for r, c in pts:
                for rr, cc in neighbors8(int(r), int(c)):
                    nid = pix2node.get((int(rr), int(cc)))
                    if nid is None:
                        continue
                    side = _D2._track_touch_side((int(r), int(c)), id2bbox[int(nid)])
                    if side is None:
                        continue
                    touched.add(int(nid))
                    side_hits[int(nid)].add(str(side))
            if len(touched) != 2 or any(len(side_hits[int(nid)]) != 1 for nid in touched):
                continue
            a, b = sorted(touched)
            edge_meta[int(edge_id)] = {
                "pts": pts,
                "endpoints": (int(a), int(b)),
                "sides": {int(a): next(iter(side_hits[int(a)])), int(b): next(iter(side_hits[int(b)]))},
            }

        out = g.copy()
        out[out == int(track_color)] = int(LIGHT_TRACK_COLOR)
        blocked_extra = 0
        for eid, meta in edge_meta.items():
            a, b = map(int, meta["endpoints"])
            side_a = str(meta["sides"][int(a)])
            side_b = str(meta["sides"][int(b)])

            if (int(a), int(b), int(eid)) in path_oriented:
                src, dst = int(a), int(b)
            elif (int(b), int(a), int(eid)) in path_oriented:
                src, dst = int(b), int(a)
            elif str(id2dir[int(a)]) == str(side_a) and str(id2dir[int(b)]) != str(side_b):
                src, dst = int(b), int(a)
                blocked_extra += 1
            elif str(id2dir[int(b)]) == str(side_b) and str(id2dir[int(a)]) != str(side_a):
                src, dst = int(a), int(b)
                blocked_extra += 1
            else:
                src, dst = int(a), int(b)

            _recolor_edge_by_orientation(out, meta["pts"], centers[int(src)], centers[int(dst)])

        try:
            parsed_dual = _parse_dual_directed_graph(out)
            dual_paths = _find_terminal_candidate_paths(parsed_dual["term_ids"], parsed_dual["directed_adj"])
            if not dual_paths:
                continue
            _ = main(out)
        except Exception:
            continue
        if blocked_extra < 1:
            continue
        return out
    raise RuntimeError("generate_input: failed to build a dual-direction sample")
