from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, grid.ravel()))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _find_solid_squares(grid: np.ndarray, bg: int) -> list[dict]:
    squares: list[dict] = []
    for color in colors_present(grid, exclude={bg}):
        for coords in components_mask(grid == color):
            if coords.size == 0:
                continue
            r0, c0, r1, c1 = bbox_coords(coords)
            side_h = r1 - r0 + 1
            side_w = c1 - c0 + 1
            if side_h < 2 or side_h != side_w:
                continue
            if int(coords.shape[0]) != side_h * side_w:
                continue
            squares.append(
                {
                    "color": int(color),
                    "bbox": (r0, c0, r1, c1),
                    "side": int(side_h),
                    "coords": {(int(r), int(c)) for r, c in coords},
                }
            )
    return squares


def _looks_like_cap_color(grid: np.ndarray, color: int) -> bool:
    comps = components_mask(grid == int(color))
    if not comps:
        return False
    for coords in comps:
        r0, c0, r1, c1 = bbox_coords(coords)
        if int(coords.shape[0]) != 3 or r1 - r0 != 2 or c1 != c0:
            return False
    return True


def _shortest_path_info(
    start: int, goal: int, adj: dict[int, list[tuple[int, int]]]
) -> dict | None:
    q = deque([start])
    dist = {int(start): 0}
    count = {int(start): 1}
    parent: dict[int, int | None] = {int(start): None}
    parent_edge: dict[int, int | None] = {int(start): None}

    while q:
        u = int(q.popleft())
        for v, eid in adj.get(u, []):
            v = int(v)
            nd = dist[u] + 1
            if v not in dist:
                dist[v] = nd
                count[v] = count[u]
                parent[v] = u
                parent_edge[v] = int(eid)
                q.append(v)
            elif dist[v] == nd:
                count[v] += count[u]

    if goal not in dist:
        return None

    node_path: list[int] = []
    edge_path: list[int] = []
    cur: int | None = int(goal)
    while cur is not None:
        node_path.append(int(cur))
        eid = parent_edge[int(cur)]
        if eid is not None:
            edge_path.append(int(eid))
        cur = parent[int(cur)]
    node_path.reverse()
    edge_path.reverse()
    return {
        "dist": int(dist[goal]),
        "count": int(count[goal]),
        "nodes": node_path,
        "edges": edge_path,
    }


def _alternate_route_distance(
    start: int,
    goal: int,
    adj: dict[int, list[tuple[int, int]]],
    shortest_edges: list[int],
) -> int | None:
    """Find the best route that differs from the unique shortest route."""
    alternatives: list[int] = []
    for banned_eid in shortest_edges:
        q = deque([int(start)])
        dist = {int(start): 0}
        while q:
            u = int(q.popleft())
            for v, eid in adj.get(u, []):
                if int(eid) == int(banned_eid) or int(v) in dist:
                    continue
                dist[int(v)] = int(dist[u]) + 1
                q.append(int(v))
        if int(goal) in dist:
            alternatives.append(int(dist[int(goal)]))
    return min(alternatives) if alternatives else None


def _parse_graph(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    square_candidates = _find_solid_squares(g, bg)
    assert square_candidates, "expected at least one solid square"
    size_counts = Counter(int(sq["side"]) for sq in square_candidates)
    node_side = max(size_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
    node_squares = [sq for sq in square_candidates if int(sq["side"]) == int(node_side)]
    assert len(node_squares) >= 4, "expected start, end, and several station squares"

    nodes: list[dict] = []
    pix2node: dict[Coord, int] = {}
    nodes_by_color: dict[int, list[int]] = {}
    for nid, sq in enumerate(node_squares):
        node = {
            "id": int(nid),
            "color": int(sq["color"]),
            "bbox": sq["bbox"],
            "coords": set(sq["coords"]),
        }
        nodes.append(node)
        nodes_by_color.setdefault(int(sq["color"]), []).append(int(nid))
        for p in node["coords"]:
            pix2node[p] = int(nid)

    station_colors = [color for color, ids in nodes_by_color.items() if len(ids) > 1]
    assert len(station_colors) == 1, "expected exactly one repeated station color"
    terminal_ids = [ids[0] for color, ids in nodes_by_color.items() if len(ids) == 1]
    assert len(terminal_ids) == 2, "expected exactly two singleton terminal squares"

    non_bg = colors_present(g, exclude={bg})
    node_colors = set(nodes_by_color)
    extra_colors = [color for color in non_bg if color not in node_colors]
    assert 1 <= len(extra_colors) <= 2, "expected track color plus optional cap color"
    if len(extra_colors) == 1:
        track_color = int(extra_colors[0])
        cap_color: int | None = None
    else:
        if _looks_like_cap_color(g, int(extra_colors[0])) and not _looks_like_cap_color(g, int(extra_colors[1])):
            cap_color, track_color = int(extra_colors[0]), int(extra_colors[1])
        elif _looks_like_cap_color(g, int(extra_colors[1])) and not _looks_like_cap_color(g, int(extra_colors[0])):
            cap_color, track_color = int(extra_colors[1]), int(extra_colors[0])
        else:
            extra_sorted = sorted(extra_colors, key=lambda color: int((g == int(color)).sum()))
            cap_color, track_color = int(extra_sorted[0]), int(extra_sorted[1])

    edge_mask = g == int(track_color)
    if cap_color is not None:
        edge_mask |= g == int(cap_color)
    edge_components = components_mask(edge_mask)
    assert edge_components, "expected track components"

    edges: list[dict] = []
    adj: dict[int, list[tuple[int, int]]] = {int(node["id"]): [] for node in nodes}
    for eid, coords in enumerate(edge_components):
        pts = [(int(r), int(c)) for r, c in coords]
        touched: set[int] = set()
        for r, c in pts:
            for nr, nc in neighbors4(r, c):
                nid = pix2node.get((nr, nc))
                if nid is not None:
                    touched.add(int(nid))
        if len(touched) != 2:
            continue

        cap_pts = [(r, c) for r, c in pts if cap_color is not None and int(g[r, c]) == int(cap_color)]
        edge = {"id": int(eid), "nodes": tuple(sorted(touched)), "coords": pts, "one_way": False}
        if cap_pts:
            cap_arr = np.array(cap_pts, dtype=int)
            r0, c0, r1, c1 = bbox_coords(cap_arr)
            assert len(cap_pts) == 3 and c0 == c1 and r1 - r0 == 2, "caps must be 3-cell vertical lines"
            cap_nodes: set[int] = set()
            for r, c in cap_pts:
                for nr, nc in neighbors4(r, c):
                    nid = pix2node.get((nr, nc))
                    if nid is not None:
                        cap_nodes.add(int(nid))
            assert len(cap_nodes) == 1, "cap must sit against exactly one node"
            capped_node = next(iter(cap_nodes))
            start_node = next(iter(touched - {capped_node}))
            edge["one_way"] = True
            edge["start"] = int(start_node)
            edge["end"] = int(capped_node)
            adj[int(start_node)].append((int(capped_node), int(eid)))
        else:
            a, b = edge["nodes"]
            adj[int(a)].append((int(b), int(eid)))
            adj[int(b)].append((int(a), int(eid)))
        edges.append(edge)

    assert edges, "expected at least one parsed edge"
    return {
        "bg": int(bg),
        "nodes": nodes,
        "edges": edges,
        "adj": adj,
        "terminal_ids": [int(terminal_ids[0]), int(terminal_ids[1])],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the unique legal shortest route from the start square to the end square in a square-and-track graph with one-way capped rails.

    Input structure:
    - The background is the dominant color.
    - One solid square marks the start, one solid square marks the end, and several same-colored solid squares act as station nodes.
    - A fourth color forms 1-cell-wide rail segments between node squares; some rails have a 3-cell vertical cap of a fifth color at one endpoint.

    Rule:
    - Uncapped rails are bidirectional, while a capped rail may only be traversed from its uncapped endpoint toward the capped endpoint.
    - Find the unique legal shortest route from the start square to the end square, minimizing the number of traversed rails.
    - Output a background grid of the same shape that keeps only the start square, end square, stations on that route, traversed rail pixels, and any caps belonging to those traversed rails.

    Key insight:
    - The cap is not decoration: it orients an edge. The right solution is a shortest path in a directed graph extracted from the node squares and rail components.
    """
    parsed = _parse_graph(np.asarray(grid, dtype=int))
    term_a, term_b = parsed["terminal_ids"]
    info_ab = _shortest_path_info(int(term_a), int(term_b), parsed["adj"])
    info_ba = _shortest_path_info(int(term_b), int(term_a), parsed["adj"])

    chosen = None
    if info_ab is not None and info_ba is None:
        chosen = info_ab
    elif info_ba is not None and info_ab is None:
        chosen = info_ba
    elif info_ab is not None and info_ba is not None:
        if info_ab["dist"] < info_ba["dist"]:
            chosen = info_ab
        elif info_ba["dist"] < info_ab["dist"]:
            chosen = info_ba
        else:
            if set(info_ab["edges"]) == set(info_ba["edges"]) and set(info_ab["nodes"]) == set(info_ba["nodes"]):
                chosen = info_ab
    assert chosen is not None and int(chosen["count"]) == 1, "route must be uniquely determined"

    g = np.asarray(grid, dtype=int)
    out = np.full_like(g, int(parsed["bg"]))
    edge_by_id = {int(edge["id"]): edge for edge in parsed["edges"]}
    node_by_id = {int(node["id"]): node for node in parsed["nodes"]}

    keep: set[Coord] = set()
    for nid in chosen["nodes"]:
        keep.update(node_by_id[int(nid)]["coords"])
    for eid in chosen["edges"]:
        keep.update(edge_by_id[int(eid)]["coords"])

    for r, c in keep:
        out[int(r), int(c)] = int(g[int(r), int(c)])
    return out


NODE_SIDE = 3
GRID_ROWS_CHOICES = [2, 3, 4]
GRID_COLS_CHOICES = [3, 4]
GAP_CHOICES = [4, 5]
EXTRA_EDGE_CHOICES = [1, 2, 3, 4]
ONE_WAY_PROB = 0.7
MAX_ATTEMPTS = 500
PALETTE = list(range(10))


def _cell_neighbors(cell: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < rows:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < cols:
        out.append((r, c + 1))
    return out


def _all_edges(rows: int, cols: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    edges = []
    for r in range(rows):
        for c in range(cols):
            if r + 1 < rows:
                edges.append(((r, c), (r + 1, c)))
            if c + 1 < cols:
                edges.append(((r, c), (r, c + 1)))
    return edges


def _sample_connected_edges(
    rows: int, cols: int, rng: random.Random
) -> tuple[list[tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int]]]]:
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    connected = {rng.choice(cells)}
    tree_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    while len(connected) < len(cells):
        frontier = []
        for cell in connected:
            for nb in _cell_neighbors(cell, rows, cols):
                if nb not in connected:
                    frontier.append((cell, nb))
        if not frontier:
            raise AssertionError("frontier unexpectedly empty")
        a, b = rng.choice(frontier)
        tree_edges.add(tuple(sorted((a, b))))
        connected.add(b)

    all_edges = [tuple(sorted(edge)) for edge in _all_edges(rows, cols)]
    remaining = [edge for edge in all_edges if edge not in tree_edges]
    extra_max = min(len(remaining), max(EXTRA_EDGE_CHOICES))
    extra_n = 0 if extra_max == 0 else rng.randint(1, extra_max)
    extra_edges = rng.sample(remaining, extra_n) if extra_n else []
    edges = sorted(tree_edges | set(extra_edges))
    return cells, edges


def _build_edge_specs(
    cells: list[tuple[int, int]],
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
    rng: random.Random,
) -> tuple[dict[tuple[int, int], int], list[dict], dict[int, list[tuple[int, int]]], dict[int, list[tuple[int, int]]]]:
    cell_to_id = {cell: idx for idx, cell in enumerate(cells)}
    specs: list[dict] = []
    directed_adj: dict[int, list[tuple[int, int]]] = {idx: [] for idx in range(len(cells))}
    undirected_adj: dict[int, list[tuple[int, int]]] = {idx: [] for idx in range(len(cells))}

    for eid, (a_cell, b_cell) in enumerate(edges):
        a = int(cell_to_id[a_cell])
        b = int(cell_to_id[b_cell])
        spec = {"id": int(eid), "nodes": (a, b), "cells": (a_cell, b_cell), "one_way": False, "cap_node": None}
        if a_cell[0] == b_cell[0] and rng.random() < ONE_WAY_PROB:
            capped = a if rng.random() < 0.5 else b
            start = b if capped == a else a
            spec["one_way"] = True
            spec["cap_node"] = int(capped)
            spec["start"] = int(start)
            spec["end"] = int(capped)
            directed_adj[int(start)].append((int(capped), int(eid)))
        else:
            directed_adj[int(a)].append((int(b), int(eid)))
            directed_adj[int(b)].append((int(a), int(eid)))
        undirected_adj[int(a)].append((int(b), int(eid)))
        undirected_adj[int(b)].append((int(a), int(eid)))
        specs.append(spec)
    return cell_to_id, specs, directed_adj, undirected_adj


def _edge_pixels(
    spec: dict,
    id_to_pos: dict[int, tuple[int, int]],
) -> tuple[set[Coord], set[Coord]]:
    a, b = spec["nodes"]
    ra, ca = id_to_pos[int(a)]
    rb, cb = id_to_pos[int(b)]
    track: set[Coord] = set()
    cap: set[Coord] = set()

    if ra == rb:
        row = ra + NODE_SIDE // 2
        left_col = min(ca, cb)
        right_col = max(ca, cb)
        cols = list(range(left_col + NODE_SIDE, right_col))
        assert len(cols) >= 3
        if spec["one_way"]:
            cap_col = left_col + NODE_SIDE if int(spec["cap_node"]) == (a if ca < cb else b) else right_col - 1
            for col in cols:
                if col != cap_col:
                    track.add((row, col))
            cap.update({(row - 1, cap_col), (row, cap_col), (row + 1, cap_col)})
        else:
            for col in cols:
                track.add((row, col))
    else:
        col = ca + NODE_SIDE // 2
        top_row = min(ra, rb)
        bottom_row = max(ra, rb)
        rows = list(range(top_row + NODE_SIDE, bottom_row))
        assert len(rows) >= 3
        for row in rows:
            track.add((row, col))
    return track, cap


def _pick_route(
    directed_adj: dict[int, list[tuple[int, int]]],
    undirected_adj: dict[int, list[tuple[int, int]]],
    rng: random.Random,
) -> tuple[int, int, dict] | None:
    node_ids = sorted(directed_adj)
    candidates = []
    for start in node_ids:
        for end in node_ids:
            if start == end:
                continue
            info = _shortest_path_info(int(start), int(end), directed_adj)
            if info is None or int(info["count"]) != 1 or int(info["dist"]) < 2:
                continue
            alternate_dist = _alternate_route_distance(
                int(start), int(end), directed_adj, list(info["edges"])
            )
            if alternate_dist is None or int(alternate_dist) <= int(info["dist"]):
                continue
            reverse = _shortest_path_info(int(end), int(start), directed_adj)
            if reverse is not None:
                continue
            undirected = _shortest_path_info(int(start), int(end), undirected_adj)
            assert undirected is not None
            if int(info["dist"]) > max(3, len(node_ids) - 2):
                continue
            candidates.append(
                {
                    "start": int(start),
                    "end": int(end),
                    "info": info,
                    "alternate_dist": int(alternate_dist),
                    "detour": int(info["dist"]) > int(undirected["dist"]),
                }
            )
    if not candidates:
        return None
    detours = [cand for cand in candidates if cand["detour"]]
    # Every generated puzzle must contain a geometrically shorter route that
    # becomes illegal once the caps are interpreted as one-way directions.
    # The remaining directed graph still has multiple legal routes, and its
    # unique shortest legal route is the answer.
    if not detours:
        return None
    chosen = rng.choice(detours)
    return int(chosen["start"]), int(chosen["end"]), chosen["info"]


def _render_grid(
    rows: int,
    cols: int,
    gap: int,
    start_id: int,
    end_id: int,
    specs: list[dict],
    rng: random.Random,
) -> tuple[np.ndarray, dict[int, set[Coord]], dict[int, set[Coord]], int]:
    pitch = NODE_SIDE + gap
    height = rows * NODE_SIDE + (rows - 1) * gap + 2
    width = cols * NODE_SIDE + (cols - 1) * gap + 2
    assert height <= 30 and width <= 30

    colors = rng.sample(PALETTE, 6)
    bg, start_color, end_color, station_color, track_color, cap_color = colors
    grid = np.full((height, width), int(bg), dtype=int)

    cell_to_id = {(r, c): r * cols + c for r in range(rows) for c in range(cols)}
    id_to_pos: dict[int, tuple[int, int]] = {}
    node_pixels: dict[int, set[Coord]] = {}
    for r in range(rows):
        for c in range(cols):
            nid = int(cell_to_id[(r, c)])
            r0 = 1 + r * pitch
            c0 = 1 + c * pitch
            id_to_pos[nid] = (r0, c0)
            color = station_color
            if nid == int(start_id):
                color = start_color
            elif nid == int(end_id):
                color = end_color
            pixels = {(rr, cc) for rr in range(r0, r0 + NODE_SIDE) for cc in range(c0, c0 + NODE_SIDE)}
            node_pixels[nid] = pixels
            for rr, cc in pixels:
                grid[rr, cc] = int(color)

    edge_pixels: dict[int, set[Coord]] = {}
    for spec in specs:
        track, cap = _edge_pixels(spec, id_to_pos)
        edge_pixels[int(spec["id"])] = set(track) | set(cap)
        for r, c in track:
            assert grid[r, c] == int(bg)
            grid[r, c] = int(track_color)
        for r, c in cap:
            assert grid[r, c] == int(bg)
            grid[r, c] = int(cap_color)

    assert np.argmax(np.bincount(grid.ravel(), minlength=10)) == int(bg)
    return grid, node_pixels, edge_pixels, int(bg)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    rows = rng.choice(GRID_ROWS_CHOICES)
    cols = rng.choice(GRID_COLS_CHOICES)
    gap = rng.choice(GAP_CHOICES)

    cells, edges = _sample_connected_edges(rows, cols, rng)
    cell_to_id, specs, directed_adj, undirected_adj = _build_edge_specs(cells, edges, rng)
    picked = _pick_route(directed_adj, undirected_adj, rng)
    if picked is None:
        return None
    start_id, end_id, info = picked

    grid, node_pixels, edge_pixels, bg = _render_grid(rows, cols, gap, start_id, end_id, specs, rng)
    keep: set[Coord] = set()
    for nid in info["nodes"]:
        keep.update(node_pixels[int(nid)])
    for eid in info["edges"]:
        keep.update(edge_pixels[int(eid)])

    expected = np.full_like(grid, int(bg))
    for r, c in keep:
        expected[r, c] = int(grid[r, c])

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_ATTEMPTS} attempts")
