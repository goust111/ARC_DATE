from __future__ import annotations

from collections import Counter, defaultdict, deque
from pathlib import Path
import random
import sys

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg  # noqa: E402


BG = Color.BLACK
BLUE = Color.BLUE
RED = Color.RED
GREEN = Color.GREEN
YELLOW = Color.YELLOW
GREY = Color.GREY

DIRS = {
    "NW": (-1, -1),
    "NE": (-1, 1),
    "SW": (1, -1),
    "SE": (1, 1),
}
DIR_FROM_MISSING = {
    "NW": (0, 0),
    "NE": (0, 1),
    "SW": (1, 0),
    "SE": (1, 1),
}
EDGE_PRIORITY = {"direct": 0, "yellow": 1, "green": 2}


def _diag_region(r0: int, c0: int, dir_name: str) -> tuple[int, int]:
    if dir_name == "NW":
        return r0 - 2, c0 - 2
    if dir_name == "NE":
        return r0 - 2, c0 + 2
    if dir_name == "SW":
        return r0 + 2, c0 - 2
    if dir_name == "SE":
        return r0 + 2, c0 + 2
    raise ValueError(dir_name)


def _start_from_node(r0: int, c0: int, dir_name: str) -> tuple[int, int]:
    if dir_name == "NW":
        return r0 - 2, c0 - 2
    if dir_name == "NE":
        return r0 - 2, c0 + 3
    if dir_name == "SW":
        return r0 + 3, c0 - 2
    if dir_name == "SE":
        return r0 + 3, c0 + 3
    raise ValueError(dir_name)


def _node_ring(r0: int, c0: int) -> set[tuple[int, int]]:
    return {
        (r0 - 1, c0),
        (r0 - 1, c0 + 1),
        (r0 + 2, c0),
        (r0 + 2, c0 + 1),
        (r0, c0 - 1),
        (r0 + 1, c0 - 1),
        (r0, c0 + 2),
        (r0 + 1, c0 + 2),
    }


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _extract_nodes(grid: np.ndarray) -> list[dict]:
    nodes: list[dict] = []
    for comp in find_components(grid, bg=BG, connectivity=4, monochromatic=True):
        color = int(comp["color"])
        if color not in (BLUE, RED):
            continue
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(patch == color):
            continue
        nodes.append(
            {
                "color": color,
                "r0": r0,
                "c0": c0,
                "cells": {(r0 + dr, c0 + dc) for dr in range(2) for dc in range(2)},
                "ring": _node_ring(r0, c0),
            }
        )
    nodes.sort(key=lambda d: (d["r0"], d["c0"]))
    return nodes


def _find_arrow_direction(grid: np.ndarray, node: dict) -> str:
    h, w = grid.shape
    r0, c0 = int(node["r0"]), int(node["c0"])
    found: list[str] = []
    for dir_name, missing in DIR_FROM_MISSING.items():
        ar, ac = _diag_region(r0, c0, dir_name)
        if ar < 0 or ac < 0 or ar + 2 > h or ac + 2 > w:
            continue
        block = grid[ar : ar + 2, ac : ac + 2]
        if int(np.sum(block == GREY)) != 3:
            continue
        if int(np.sum(block == BG)) != 1:
            continue
        mr, mc = missing
        if int(block[mr, mc]) != BG:
            continue
        ok = True
        for rr in range(2):
            for cc in range(2):
                if (rr, cc) == missing:
                    continue
                if int(block[rr, cc]) != GREY:
                    ok = False
        if ok:
            found.append(dir_name)
    if len(found) != 1:
        raise AssertionError("expected exactly one diagonal arrow per node")
    return found[0]


def _scene_from_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("grid must be 2D")
    if int(get_bg(g)) != BG:
        raise AssertionError("background must be black")

    nodes = _extract_nodes(g)
    if len(nodes) < 5:
        raise AssertionError("expected 5..7 nodes")

    terminals = [i for i, nd in enumerate(nodes) if int(nd["color"]) == RED]
    if len(terminals) != 2:
        raise AssertionError("expected exactly two red terminals")

    wall_map: dict[tuple[int, int], int] = {}
    for r, c in np.argwhere((g == YELLOW) | (g == GREEN)):
        wall_map[(int(r), int(c))] = int(g[int(r), int(c)])

    for nd in nodes:
        dir_name = _find_arrow_direction(g, nd)
        nd["dir_name"] = dir_name
        nd["dir"] = DIRS[dir_name]
        nd["start"] = _start_from_node(int(nd["r0"]), int(nd["c0"]), dir_name)

    return {"grid": g, "nodes": nodes, "terminals": terminals, "wall_map": wall_map}


def _trace_ray(scene: dict, src_idx: int) -> tuple[int, str] | None:
    g = scene["grid"]
    h, w = g.shape
    nodes = scene["nodes"]
    wall_map = scene["wall_map"]

    src = nodes[src_idx]
    pos = tuple(map(int, src["start"]))
    dr, dc = map(int, src["dir"])
    last_wall_color: int | None = None
    seen: set[tuple[tuple[int, int], tuple[int, int], int | None]] = set()

    if not _in_bounds((h, w), pos[0], pos[1]):
        return None

    for _ in range(h * w * 8):
        state = (pos, (dr, dc), last_wall_color)
        if state in seen:
            return None
        seen.add(state)

        for j, nd in enumerate(nodes):
            if j != src_idx and pos in nd["ring"]:
                if last_wall_color is None:
                    return j, "direct"
                if last_wall_color == YELLOW:
                    return j, "yellow"
                if last_wall_color == GREEN:
                    return j, "green"
                return None

        front = (pos[0] + dr, pos[1] + dc)
        vertical = (pos[0] + dr, pos[1])
        horizontal = (pos[0], pos[1] + dc)

        if front in wall_map:
            dr, dc = -dr, -dc
            last_wall_color = int(wall_map[front])
            continue

        flip_r = vertical in wall_map
        flip_c = horizontal in wall_map
        if flip_r:
            dr = -dr
            last_wall_color = int(wall_map[vertical])
        if flip_c:
            dc = -dc
            last_wall_color = int(wall_map[horizontal])

        pos = (pos[0] + dr, pos[1] + dc)
        if not _in_bounds((h, w), pos[0], pos[1]):
            return None

    return None


def _collapsed_edges(scene: dict) -> tuple[dict[tuple[int, int], str], list[tuple[int, int, str]]]:
    collapsed: dict[tuple[int, int], str] = {}
    directed: list[tuple[int, int, str]] = []
    for src_idx in range(len(scene["nodes"])):
        hit = _trace_ray(scene, src_idx)
        if hit is None:
            continue
        dst_idx, kind = hit
        directed.append((src_idx, dst_idx, kind))
        edge = tuple(sorted((src_idx, dst_idx)))
        prev = collapsed.get(edge)
        if prev is None or EDGE_PRIORITY[kind] < EDGE_PRIORITY[prev]:
            collapsed[edge] = kind
    return collapsed, directed


def _bfs_with_counts(adj: dict[int, set[int]], start: int, goal: int) -> tuple[dict[int, int], dict[int, int], dict[int, int | None]]:
    q = deque([start])
    dist = {start: 0}
    count = {start: 1}
    parent: dict[int, int | None] = {start: None}
    while q:
        u = int(q.popleft())
        for v in sorted(adj.get(u, ())):
            if v not in dist:
                dist[v] = dist[u] + 1
                count[v] = count[u]
                parent[v] = u
                q.append(v)
            elif dist[v] == dist[u] + 1:
                count[v] += count[u]
    return dist, count, parent


def _path_from_parent(parent: dict[int, int | None], goal: int) -> list[int]:
    out: list[int] = []
    cur: int | None = int(goal)
    while cur is not None:
        out.append(cur)
        cur = parent.get(cur)
    out.reverse()
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the blue stations on the shortest terminal path in a graph discovered by bouncing diagonal rays.

    Input structure:
    - The background is black.
    - There are 5..7 solid 2x2 node squares: exactly 2 red terminals and the rest blue stations.
    - Each node has one adjacent grey 3-pixel L-arrow in a 2x2 diagonal quadrant, and the grid also contains yellow/green single-pixel walls.

    Rule:
    - From each node, launch a diagonal ray from the arrow's missing corner in the indicated diagonal direction.
    - A front hit on a wall reverses both axes; a side contact with a wall flips only that axis, and the ray remembers the last wall color it touched.
    - When a ray reaches any 4-neighbor cell of another node, it creates a graph edge: no wall touched = direct edge, last yellow wall = yellow edge, last green wall = green edge.
    - Ignore green edges, find the unique shortest path between the two red terminals by edge count, and recolor only the blue nodes on that path to green.

    Key insight:
    The visible output depends on an implicit graph that must first be reconstructed from ray-bounce physics; wall color inheritance then filters which discovered edges are actually usable.
    """
    scene = _scene_from_grid(grid)
    collapsed, _directed = _collapsed_edges(scene)

    start, goal = map(int, scene["terminals"])
    adj: dict[int, set[int]] = defaultdict(set)
    for (a, b), kind in collapsed.items():
        if kind == "green":
            continue
        adj[int(a)].add(int(b))
        adj[int(b)].add(int(a))

    dist, count, parent = _bfs_with_counts(adj, start, goal)
    if goal not in dist or count.get(goal, 0) != 1:
        raise AssertionError("terminal path must be uniquely determined")

    path = _path_from_parent(parent, goal)
    out = np.array(grid, dtype=int, copy=True)
    terminal_set = set(scene["terminals"])
    for idx in path:
        if idx in terminal_set:
            continue
        nd = scene["nodes"][idx]
        for r, c in nd["cells"]:
            out[int(r), int(c)] = GREEN
    return out


def _place_node(grid: np.ndarray, r0: int, c0: int, color: int, dir_name: str) -> None:
    grid[r0 : r0 + 2, c0 : c0 + 2] = int(color)
    ar, ac = _diag_region(r0, c0, dir_name)
    mr, mc = DIR_FROM_MISSING[dir_name]
    for rr in range(2):
        for cc in range(2):
            if (rr, cc) == (mr, mc):
                continue
            grid[ar + rr, ac + cc] = GREY


def _build_canonical(include_pair: bool) -> np.ndarray:
    grid = np.full((22, 22), BG, dtype=int)

    core_nodes = [
        (7, 3, RED, "SE"),
        (14, 13, BLUE, "NW"),
        (5, 13, BLUE, "SE"),
        (17, 11, BLUE, "NW"),
        (11, 17, RED, "SW"),
    ]
    for r0, c0, color, dir_name in core_nodes:
        _place_node(grid, r0, c0, color, dir_name)

    grid[16, 11] = GREEN
    grid[17, 19] = GREEN
    grid[2, 1] = GREEN
    grid[16, 16] = YELLOW

    if include_pair:
        _place_node(grid, 3, 5, BLUE, "SE")
        _place_node(grid, 5, 9, BLUE, "NW")

    return grid


def _apply_d4(grid: np.ndarray, rot_k: int, flip: bool) -> np.ndarray:
    out = np.rot90(grid, rot_k)
    if flip:
        out = np.fliplr(out)
    return np.asarray(out, dtype=int)


def _embed(grid: np.ndarray, h: int, w: int, r0: int, c0: int) -> np.ndarray:
    out = np.full((h, w), BG, dtype=int)
    gh, gw = grid.shape
    out[r0 : r0 + gh, c0 : c0 + gw] = grid
    return out


def _validate_semantics(grid: np.ndarray) -> None:
    scene = _scene_from_grid(grid)
    if len(scene["nodes"]) not in (5, 7):
        raise AssertionError("generator must produce 5 or 7 nodes")

    collapsed, directed = _collapsed_edges(scene)
    if len(directed) != len(scene["nodes"]):
        raise AssertionError("every node must connect to another node")

    counts = Counter(collapsed.values())
    if min(counts.get(kind, 0) for kind in ("direct", "yellow", "green")) < 1:
        raise AssertionError("need direct, yellow, and green edge categories")

    start, goal = map(int, scene["terminals"])
    adj_allow: dict[int, set[int]] = defaultdict(set)
    adj_all: dict[int, set[int]] = defaultdict(set)
    for (a, b), kind in collapsed.items():
        adj_all[int(a)].add(int(b))
        adj_all[int(b)].add(int(a))
        if kind != "green":
            adj_allow[int(a)].add(int(b))
            adj_allow[int(b)].add(int(a))

    dist_allow, count_allow, parent = _bfs_with_counts(adj_allow, start, goal)
    if goal not in dist_allow or count_allow.get(goal, 0) != 1 or dist_allow[goal] < 2:
        raise AssertionError("allowed graph must have a unique nontrivial terminal path")

    dist_all, count_all, _ = _bfs_with_counts(adj_all, start, goal)
    if goal not in dist_all:
        raise AssertionError("full graph must connect terminals")
    if not (
        dist_all[goal] < dist_allow[goal]
        or (dist_all[goal] == dist_allow[goal] and count_all.get(goal, 0) > count_allow.get(goal, 0))
    ):
        raise AssertionError("green edges must materially affect shortest-path semantics")

    path = _path_from_parent(parent, goal)
    if len(path) <= 2:
        raise AssertionError("output must recolor at least one blue node")

    out = main(grid)
    if np.array_equal(out, grid):
        raise AssertionError("output must differ from input")


def generate_input() -> np.ndarray:
    include_pair = random.random() < 0.45
    base = _build_canonical(include_pair=include_pair)
    rot_k = random.randrange(4)
    flip = bool(random.randrange(2))
    transformed = _apply_d4(base, rot_k, flip)

    bh, bw = transformed.shape
    h = random.randint(bh, min(28, bh + 6))
    w = random.randint(bw, min(28, bw + 6))
    r0 = random.randint(0, h - bh)
    c0 = random.randint(0, w - bw)
    grid = _embed(transformed, h, w, r0, c0)

    _validate_semantics(grid)
    return grid
