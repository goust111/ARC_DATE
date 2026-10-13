from __future__ import annotations

import heapq
import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

NODE_SIZE = 3
GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_nodes(grid: np.ndarray) -> list[dict]:
    nodes: list[dict] = []
    for color in (BLUE, RED):
        for comp in _components(grid == color):
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            if r1 - r0 + 1 != NODE_SIZE or c1 - c0 + 1 != NODE_SIZE:
                continue
            if len(comp) != NODE_SIZE * NODE_SIZE:
                continue
            if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color):
                continue
            nodes.append(
                {
                    "color": int(color),
                    "bbox": (r0, c0, r1, c1),
                    "coords": set(comp),
                    "center": (r0 + 1, c0 + 1),
                }
            )
    nodes.sort(key=lambda node: node["bbox"])
    for idx, node in enumerate(nodes):
        node["id"] = idx
    return nodes


def _extract_graph(grid: np.ndarray) -> tuple[list[dict], list[dict], int, int, dict[int, list[tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    nodes = _find_nodes(g)
    pix_to_node: dict[tuple[int, int], int] = {}
    for node in nodes:
        for cell in node["coords"]:
            pix_to_node[cell] = int(node["id"])

    red_nodes = [int(node["id"]) for node in nodes if int(node["color"]) == RED]
    if len(red_nodes) != 2:
        raise AssertionError("expected exactly two red terminal nodes")

    edge_cells = _components((g == ORANGE) | (g == GREY))
    edges: list[dict] = []
    adjacency: dict[int, list[tuple[int, int]]] = {int(node["id"]): [] for node in nodes}
    seen_pairs: set[tuple[int, int]] = set()
    for edge_id, comp in enumerate(edge_cells):
        touched: set[int] = set()
        for r, c in comp:
            for nr, nc in _neighbors4(r, c):
                node_id = pix_to_node.get((nr, nc))
                if node_id is not None:
                    touched.add(int(node_id))
        if len(touched) != 2:
            raise AssertionError("each track component must touch exactly two nodes")
        a, b = sorted(touched)
        if (a, b) in seen_pairs:
            raise AssertionError("multi-edge pair detected")
        seen_pairs.add((a, b))
        weight = sum(1 for r, c in comp if int(g[r, c]) == GREY)
        if not (1 <= weight <= 5):
            raise AssertionError("track weight must be encoded by 1..5 grey pixels")
        rows = {r for r, _ in comp}
        cols = {c for _, c in comp}
        if len(rows) != 1 and len(cols) != 1:
            raise AssertionError("track segment must be a straight horizontal or vertical line")
        edge = {"id": edge_id, "a": a, "b": b, "cells": list(comp), "weight": int(weight)}
        edges.append(edge)
        adjacency[a].append((b, edge_id))
        adjacency[b].append((a, edge_id))

    for node_id in adjacency:
        adjacency[node_id].sort()
    return nodes, edges, red_nodes[0], red_nodes[1], adjacency


def _best_path(nodes: list[dict], edges: list[dict], start: int, goal: int, adjacency: dict[int, list[tuple[int, int]]]):
    edge_by_id = {int(edge["id"]): edge for edge in edges}
    heap: list[tuple[int, int, int]] = [(0, 1, int(start))]
    best: dict[int, tuple[int, int]] = {int(start): (0, 1)}
    prev_node: dict[int, int | None] = {int(start): None}
    prev_edge: dict[int, int | None] = {int(start): None}

    while heap:
        weight, node_count, node_id = heapq.heappop(heap)
        if (weight, node_count) != best.get(int(node_id)):
            continue
        if int(node_id) == int(goal):
            break
        for nxt, edge_id in adjacency[int(node_id)]:
            edge = edge_by_id[int(edge_id)]
            cand = (int(weight) + int(edge["weight"]), int(node_count) + 1)
            if cand < best.get(int(nxt), (10**9, 10**9)):
                best[int(nxt)] = cand
                prev_node[int(nxt)] = int(node_id)
                prev_edge[int(nxt)] = int(edge_id)
                heapq.heappush(heap, (cand[0], cand[1], int(nxt)))

    if int(goal) not in best:
        raise AssertionError("terminals are disconnected")

    node_path: list[int] = []
    edge_path: list[int] = []
    cur: int | None = int(goal)
    while cur is not None:
        node_path.append(int(cur))
        edge_id = prev_edge.get(int(cur))
        if edge_id is not None:
            edge_path.append(int(edge_id))
        cur = prev_node.get(int(cur))
    node_path.reverse()
    edge_path.reverse()
    return node_path, edge_path, best[int(goal)]


def _enumerate_paths(start: int, goal: int, edges: list[dict], adjacency: dict[int, list[tuple[int, int]]]) -> list[dict]:
    edge_by_id = {int(edge["id"]): edge for edge in edges}
    out: list[dict] = []

    def dfs(node_id: int, visited: set[int], node_path: list[int], edge_path: list[int], weight: int) -> None:
        if int(node_id) == int(goal):
            out.append(
                {
                    "nodes": list(node_path),
                    "edges": list(edge_path),
                    "weight": int(weight),
                }
            )
            return
        for nxt, edge_id in adjacency[int(node_id)]:
            if int(nxt) in visited:
                continue
            edge = edge_by_id[int(edge_id)]
            visited.add(int(nxt))
            node_path.append(int(nxt))
            edge_path.append(int(edge_id))
            dfs(int(nxt), visited, node_path, edge_path, int(weight) + int(edge["weight"]))
            edge_path.pop()
            node_path.pop()
            visited.remove(int(nxt))

    dfs(int(start), {int(start)}, [int(start)], [], 0)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the minimum-weight terminal-to-terminal route in a square-and-track graph.

    Input structure:
    - A black background contains 5-8 solid 3x3 node blocks.
    - Exactly two nodes are red terminals and the remaining nodes are blue stations.
    - Orange 1-pixel-wide straight track segments connect node edges, and grey pixels may replace
      some track cells while still belonging to the same segment.

    Rule:
    - Treat each 3x3 block as a graph node and each orange-or-grey track component as one edge.
    - The weight of an edge is the number of grey pixels on that track segment.
    - Find the red-to-red path with minimum total weight; if weights tie, choose the one using
      fewer nodes.
    - Recolor every station on that chosen path green, recolor every track cell on that path
      yellow, and erase all remaining grey pixels back to orange.

    Key insight:
    - Grey track pixels have a dual role: they still preserve connectivity like orange track, but
      their count secretly encodes weighted edge costs, so plain BFS by edge count gives the wrong path.
    """
    g = np.asarray(grid, dtype=int)
    nodes, edges, start, goal, adjacency = _extract_graph(g)
    node_path, edge_path, _cost = _best_path(nodes, edges, start, goal, adjacency)

    out = g.copy()
    out[out == GREY] = ORANGE

    node_by_id = {int(node["id"]): node for node in nodes}
    edge_by_id = {int(edge["id"]): edge for edge in edges}
    edge_ids = set(map(int, edge_path))
    terminal_ids = {int(start), int(goal)}

    for node_id in node_path:
        if int(node_id) in terminal_ids:
            continue
        for r, c in node_by_id[int(node_id)]["coords"]:
            out[r, c] = GREEN

    for edge_id in edge_ids:
        for r, c in edge_by_id[int(edge_id)]["cells"]:
            out[r, c] = YELLOW

    return out


def _rotate_positions(pos_map: dict[str, tuple[int, int]], turns: int) -> dict[str, tuple[int, int]]:
    turns %= 4
    cur = dict(pos_map)
    for _ in range(turns):
        max_r = max(r for r, _ in cur.values())
        rotated = {name: (c, max_r - r) for name, (r, c) in cur.items()}
        min_r = min(r for r, _ in rotated.values())
        min_c = min(c for _, c in rotated.values())
        cur = {name: (r - min_r, c - min_c) for name, (r, c) in rotated.items()}
    return cur


def _split_positive(total: int, parts: int) -> list[int]:
    assert total >= parts >= 1
    cuts = sorted(random.sample(range(1, total), parts - 1))
    values = []
    prev = 0
    for cut in cuts + [total]:
        values.append(cut - prev)
        prev = cut
    return values


def _build_template() -> tuple[dict[str, tuple[int, int]], set[str], list[tuple[str, str, int]], str]:
    if random.random() < 0.55:
        positions = {
            "A": (0, 0),
            "B": (0, 1),
            "C": (0, 2),
            "D": (1, 0),
            "E": (1, 1),
            "F": (1, 2),
        }
        terminals = {"A", "C"}
        ad = random.randint(1, 2)
        fc = random.randint(1, 2)
        bottom_weight = ad + 1 + 1 + fc
        ab = random.randint(3, 5)
        bc = random.randint(max(3, bottom_weight + 2 - ab), 5)
        be = random.randint(4, 5)
        edges = [
            ("A", "B", ab),
            ("B", "C", bc),
            ("A", "D", ad),
            ("D", "E", 1),
            ("E", "F", 1),
            ("F", "C", fc),
            ("B", "E", be),
        ]
        if random.random() < 0.6:
            positions["G"] = (2, 0)
            edges.append(("D", "G", random.randint(1, 3)))
        if random.random() < 0.45:
            positions["H"] = (2, 2)
            edges.append(("F", "H", random.randint(1, 3)))
        return positions, terminals, edges, "detour"

    positions = {
        "A": (0, 0),
        "B": (0, 1),
        "C": (0, 2),
        "D": (1, 0),
        "E": (1, 1),
        "F": (1, 2),
        "H": (2, 1),
        "I": (2, 2),
    }
    terminals = {"A", "C"}
    extra_sum = random.randint(3, 4)
    eh, hi, inf = _split_positive(extra_sum, 3)
    ab = random.randint(3, 5)
    bc = random.randint(max(3, 8 - ab), 5)
    if ab + bc <= 3 + extra_sum:
        if ab < 5:
            ab += 1
        else:
            bc += 1
    edges = [
        ("A", "B", ab),
        ("B", "C", bc),
        ("A", "D", 1),
        ("D", "E", 1),
        ("E", "F", extra_sum),
        ("F", "C", 1),
        ("E", "H", eh),
        ("H", "I", hi),
        ("I", "F", inf),
    ]
    return positions, terminals, edges, "tie_break"


def _canvas_positions(pos_map: dict[str, tuple[int, int]]) -> tuple[dict[str, tuple[int, int]], tuple[int, int]]:
    row_count = max(r for r, _ in pos_map.values()) + 1
    col_count = max(c for _, c in pos_map.values()) + 1
    row_tracks = [random.randint(5, 7) for _ in range(row_count - 1)]
    col_tracks = [random.randint(5, 7) for _ in range(col_count - 1)]
    top_margin = random.randint(1, 3)
    left_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)

    row_tops = [top_margin]
    for track_len in row_tracks:
        row_tops.append(row_tops[-1] + NODE_SIZE + track_len)
    col_tops = [left_margin]
    for track_len in col_tracks:
        col_tops.append(col_tops[-1] + NODE_SIZE + track_len)

    h = row_tops[-1] + NODE_SIZE + bottom_margin
    w = col_tops[-1] + NODE_SIZE + right_margin
    placed = {name: (row_tops[r], col_tops[c]) for name, (r, c) in pos_map.items()}
    return placed, (h, w)


def _track_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    if ar == br:
        row = ar + 1
        if ac < bc:
            return [(row, c) for c in range(ac + NODE_SIZE, bc)]
        return [(row, c) for c in range(bc + NODE_SIZE, ac)]
    if ac == bc:
        col = ac + 1
        if ar < br:
            return [(r, col) for r in range(ar + NODE_SIZE, br)]
        return [(r, col) for r in range(br + NODE_SIZE, ar)]
    raise AssertionError("edge endpoints must be aligned horizontally or vertically")


def _render_input(pos_map: dict[str, tuple[int, int]], terminals: set[str], edges: list[tuple[str, str, int]]) -> np.ndarray:
    placed, shape = _canvas_positions(pos_map)
    h, w = shape
    if max(shape) > 30:
        raise AssertionError("grid exceeds ARC size limit")

    grid = np.full((h, w), BLACK, dtype=int)
    occupied = set()
    for name, (r0, c0) in placed.items():
        color = RED if name in terminals else BLUE
        grid[r0 : r0 + NODE_SIZE, c0 : c0 + NODE_SIZE] = color
        for rr in range(r0, r0 + NODE_SIZE):
            for cc in range(c0, c0 + NODE_SIZE):
                occupied.add((rr, cc))

    for a, b, weight in edges:
        cells = _track_cells(placed[a], placed[b])
        if len(cells) < weight:
            raise AssertionError("track too short for encoded weight")
        for cell in cells:
            if cell in occupied:
                raise AssertionError("track overlaps existing object")
            occupied.add(cell)
            grid[cell] = ORANGE
        grey_cells = random.sample(cells, weight)
        for cell in grey_cells:
            grid[cell] = GREY

    return grid


def _semantic_validate(grid: np.ndarray) -> bool:
    nodes, edges, start, goal, adjacency = _extract_graph(grid)
    if not (5 <= len(nodes) <= 8):
        return False
    if len(edges) < len(nodes):
        return False

    all_paths = _enumerate_paths(start, goal, edges, adjacency)
    if len(all_paths) < 2:
        return False

    optimal = min(all_paths, key=lambda path: (int(path["weight"]), len(path["nodes"])))
    optimal_key = (int(optimal["weight"]), len(optimal["nodes"]))
    if sum((int(path["weight"]), len(path["nodes"])) == optimal_key for path in all_paths) != 1:
        return False

    min_edge_count = min(len(path["edges"]) for path in all_paths)
    min_edge_paths = [path for path in all_paths if len(path["edges"]) == min_edge_count]
    if min(path["weight"] for path in min_edge_paths) <= int(optimal["weight"]):
        return False

    node_path, edge_path, best_cost = _best_path(nodes, edges, start, goal, adjacency)
    if (int(best_cost[0]), len(node_path)) != optimal_key:
        return False
    if node_path != optimal["nodes"] or edge_path != optimal["edges"]:
        return False

    out = main(grid)
    if np.any(out == GREY):
        return False

    path_edge_cells = {
        cell
        for edge in edges
        if int(edge["id"]) in set(optimal["edges"])
        for cell in edge["cells"]
    }
    non_path_grey = [(r, c) for r, c in np.argwhere(grid == GREY) if (int(r), int(c)) not in path_edge_cells]
    if not non_path_grey:
        return False
    if any(int(out[r, c]) != ORANGE for r, c in non_path_grey):
        return False

    path_station_ids = set(optimal["nodes"][1:-1])
    for node in nodes:
        target = GREEN if int(node["id"]) in path_station_ids else int(node["color"])
        if any(int(out[r, c]) != target for r, c in node["coords"]):
            return False

    if not any(int(out[r, c]) == YELLOW for r, c in path_edge_cells):
        return False
    if any(int(out[r, c]) != YELLOW for r, c in path_edge_cells):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    pos_map, terminals, edges, _mode = _build_template()
    pos_map = _rotate_positions(pos_map, random.randint(0, 3))
    try:
        grid = _render_input(pos_map, terminals, edges)
    except AssertionError:
        return None
    if not np.array_equal(main(grid), main(np.array(grid, copy=True))):
        return None
    if not _semantic_validate(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid weighted-path ARC input")
