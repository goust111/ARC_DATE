from __future__ import annotations

from collections import deque
import random
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np


GREY = 5
MAROON = 9
SOURCE_SIDES = ("top", "bottom", "left", "right")
DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
NODE_DIRS = {
    "U": (-2, 0),
    "D": (2, 0),
    "L": (0, -2),
    "R": (0, 2),
}
PERP_DIRS = {
    "U": ("L", "R"),
    "D": ("L", "R"),
    "L": ("U", "D"),
    "R": ("U", "D"),
}
SIDE_TO_DIR = {"top": "D", "bottom": "U", "left": "R", "right": "L"}
GEN_RETRIES = 400
TREE_RETRIES = 120
TRUNK_RETRIES = 120
COLOR_POOL = [0, 1, 2, 3, 4, 6, 7, 8]


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> Iterable[Tuple[int, int]]:
    for dr, dc in DIRS.values():
        yield r + dr, c + dc


def _border_sources(grid: np.ndarray, bg: int) -> List[Tuple[Tuple[int, int], int]]:
    h, w = grid.shape
    sources: List[Tuple[Tuple[int, int], int]] = []
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            color = int(grid[r, c])
            if color in (bg, GREY, MAROON):
                continue
            sources.append(((r, c), color))
    return sources


def _detect_valves(grid: np.ndarray) -> Dict[Tuple[int, int], str]:
    h, w = grid.shape
    valves: Dict[Tuple[int, int], str] = {}
    for r, c in np.argwhere(grid == MAROON):
        rr = int(r)
        cc = int(c)
        up = rr > 0 and int(grid[rr - 1, cc]) == GREY
        down = rr + 1 < h and int(grid[rr + 1, cc]) == GREY
        left = cc > 0 and int(grid[rr, cc - 1]) == GREY
        right = cc + 1 < w and int(grid[rr, cc + 1]) == GREY
        if up and down:
            valves[(rr, cc)] = "v"
        elif left and right:
            valves[(rr, cc)] = "h"
        else:
            raise AssertionError("invalid valve neighborhood")
    return valves


def _axis_allows(axis: str, dr: int, dc: int) -> bool:
    if axis == "v":
        return dr != 0
    if axis == "h":
        return dc != 0
    raise AssertionError("unknown valve axis")


def _can_step(
    grid: np.ndarray,
    cur: Tuple[int, int],
    nxt: Tuple[int, int],
    source: Tuple[int, int],
    source_positions: Set[Tuple[int, int]],
    valves: Dict[Tuple[int, int], str],
    bg: int,
) -> bool:
    h, w = grid.shape
    nr, nc = nxt
    if not (0 <= nr < h and 0 <= nc < w):
        return False

    nxt_color = int(grid[nr, nc])
    if nxt_color == bg:
        return False
    if nxt in source_positions and nxt != source:
        return False

    cr, cc = cur
    dr = nr - cr
    dc = nc - cc
    if cur in valves and not _axis_allows(valves[cur], dr, dc):
        return False
    if nxt in valves and not _axis_allows(valves[nxt], dr, dc):
        return False
    return True


def _reachable_from_source(
    grid: np.ndarray,
    source: Tuple[int, int],
    source_positions: Set[Tuple[int, int]],
    valves: Dict[Tuple[int, int], str],
    bg: int,
) -> Set[Tuple[int, int]]:
    seen = {source}
    queue = deque([source])
    while queue:
        cur = queue.popleft()
        for nxt in _neighbors4(*cur):
            if nxt in seen:
                continue
            if not _can_step(grid, cur, nxt, source, source_positions, valves, bg):
                continue
            seen.add(nxt)
            queue.append(nxt)
    return seen


def _solve(grid: np.ndarray) -> Tuple[np.ndarray, Dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)
    sources = _border_sources(g, bg)
    assert 1 <= len(sources) <= 3, "expected 1-3 border sources"
    source_positions = {pos for pos, _color in sources}
    assert len(source_positions) == len(sources), "source pixels must be distinct"
    valves = _detect_valves(g)

    out = g.copy()
    reach_count = np.zeros(g.shape, dtype=int)
    owner_index = np.full(g.shape, -1, dtype=int)
    reach_sets: List[Set[Tuple[int, int]]] = []

    for idx, (source_pos, _source_color) in enumerate(sources):
        reach = _reachable_from_source(g, source_pos, source_positions, valves, bg)
        reach_sets.append(reach)
        for r, c in reach:
            if int(g[r, c]) != GREY:
                continue
            reach_count[r, c] += 1
            if reach_count[r, c] == 1:
                owner_index[r, c] = idx
            else:
                owner_index[r, c] = -2

    for r, c in np.argwhere(g == GREY):
        rr = int(r)
        cc = int(c)
        if reach_count[rr, cc] != 1:
            continue
        out[rr, cc] = sources[int(owner_index[rr, cc])][1]

    meta: Dict[str, object] = {
        "bg": bg,
        "sources": sources,
        "valves": valves,
        "reach_count": reach_count,
        "reach_sets": reach_sets,
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill every pipe segment reachable from each colored edge source while
    respecting the directional behavior of maroon valve junctions.
    """
    out, _meta = _solve(grid)
    return out


def _edge_key(a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    return (a, b) if a <= b else (b, a)


def _step_node(node: Tuple[int, int], direction: str) -> Tuple[int, int]:
    dr, dc = NODE_DIRS[direction]
    return node[0] + dr, node[1] + dc


def _in_node_bounds(node: Tuple[int, int], height: int, width: int) -> bool:
    r, c = node
    return 1 <= r <= height - 2 and 1 <= c <= width - 2 and r % 2 == 1 and c % 2 == 1


def _source_cell(side: str, node: Tuple[int, int], height: int, width: int) -> Tuple[int, int]:
    r, c = node
    if side == "top":
        return 0, c
    if side == "bottom":
        return height - 1, c
    if side == "left":
        return r, 0
    if side == "right":
        return r, width - 1
    raise AssertionError("unknown side")


def _start_nodes_for_side(side: str, height: int, width: int) -> List[Tuple[int, int]]:
    if side in ("top", "bottom"):
        row = 1 if side == "top" else height - 2
        return [(row, c) for c in range(1, width - 1, 2)]
    col = 1 if side == "left" else width - 2
    return [(r, col) for r in range(1, height - 1, 2)]


def _build_trunk(
    start: Tuple[int, int],
    initial_dir: str,
    height: int,
    width: int,
    occupied_nodes: Set[Tuple[int, int]],
    occupied_edges: Set[Tuple[Tuple[int, int], Tuple[int, int]]],
    rng: random.Random,
) -> Tuple[List[Tuple[int, int]], Set[Tuple[int, int]], Set[Tuple[Tuple[int, int], Tuple[int, int]]]] | None:
    for _ in range(TRUNK_RETRIES):
        segment_count = rng.choices([1, 2, 3], weights=[2, 3, 2], k=1)[0]
        directions = [initial_dir]
        if segment_count >= 2:
            directions.append(rng.choice(PERP_DIRS[initial_dir]))
        if segment_count == 3:
            directions.append(initial_dir)

        if segment_count == 1:
            lengths = [rng.randint(3, 6)]
        elif segment_count == 2:
            lengths = [rng.randint(2, 4), rng.randint(2, 4)]
        else:
            lengths = [rng.randint(2, 4), rng.randint(2, 4), rng.randint(2, 4)]

        path = [start]
        nodes = {start}
        edges: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()
        cur = start
        ok = True
        for direction, seg_len in zip(directions, lengths):
            for _step in range(seg_len):
                nxt = _step_node(cur, direction)
                edge = _edge_key(cur, nxt)
                if not _in_node_bounds(nxt, height, width):
                    ok = False
                    break
                if nxt in nodes or nxt in occupied_nodes or edge in edges or edge in occupied_edges:
                    ok = False
                    break
                nodes.add(nxt)
                edges.add(edge)
                path.append(nxt)
                cur = nxt
            if not ok:
                break
        if ok and len(path) >= 5:
            return path, nodes, edges
    return None


def _straight_trunk_nodes(path: Sequence[Tuple[int, int]]) -> List[Tuple[Tuple[int, int], str]]:
    result: List[Tuple[Tuple[int, int], str]] = []
    for idx in range(1, len(path) - 1):
        prev_node = path[idx - 1]
        node = path[idx]
        next_node = path[idx + 1]
        if prev_node[0] == node[0] == next_node[0]:
            result.append((node, "h"))
        elif prev_node[1] == node[1] == next_node[1]:
            result.append((node, "v"))
    return result


def _incident_dirs(
    node: Tuple[int, int],
    edges: Set[Tuple[Tuple[int, int], Tuple[int, int]]],
) -> Set[str]:
    dirs: Set[str] = set()
    r, c = node
    for a, b in edges:
        if a == node:
            other = b
        elif b == node:
            other = a
        else:
            continue
        dr = other[0] - r
        dc = other[1] - c
        if dr == -2 and dc == 0:
            dirs.add("U")
        elif dr == 2 and dc == 0:
            dirs.add("D")
        elif dr == 0 and dc == -2:
            dirs.add("L")
        elif dr == 0 and dc == 2:
            dirs.add("R")
    return dirs


def _free_run_length(
    start: Tuple[int, int],
    direction: str,
    height: int,
    width: int,
    used_nodes: Set[Tuple[int, int]],
    used_edges: Set[Tuple[Tuple[int, int], Tuple[int, int]]],
    limit: int = 3,
) -> int:
    cur = start
    length = 0
    while length < limit:
        nxt = _step_node(cur, direction)
        edge = _edge_key(cur, nxt)
        if not _in_node_bounds(nxt, height, width):
            break
        if nxt in used_nodes or edge in used_edges:
            break
        length += 1
        cur = nxt
    return length


def _extend_branch(
    start: Tuple[int, int],
    direction: str,
    length: int,
    nodes: Set[Tuple[int, int]],
    edges: Set[Tuple[Tuple[int, int], Tuple[int, int]]],
) -> None:
    cur = start
    for _ in range(length):
        nxt = _step_node(cur, direction)
        edges.add(_edge_key(cur, nxt))
        nodes.add(nxt)
        cur = nxt


def _add_valves(
    path: Sequence[Tuple[int, int]],
    local_nodes: Set[Tuple[int, int]],
    local_edges: Set[Tuple[Tuple[int, int], Tuple[int, int]]],
    occupied_nodes: Set[Tuple[int, int]],
    occupied_edges: Set[Tuple[Tuple[int, int], Tuple[int, int]]],
    height: int,
    width: int,
    rng: random.Random,
) -> Set[Tuple[int, int]]:
    candidates = _straight_trunk_nodes(path)
    rng.shuffle(candidates)
    target = rng.randint(1, max(1, min(3, len(candidates))))
    valves: Set[Tuple[int, int]] = set()

    for node, orient in candidates:
        if len(valves) >= target:
            break
        used_nodes = occupied_nodes | local_nodes
        used_edges = occupied_edges | local_edges
        options: List[List[str]] = []
        if orient == "v":
            left_len = _free_run_length(node, "L", height, width, used_nodes, used_edges)
            right_len = _free_run_length(node, "R", height, width, used_nodes, used_edges)
            if left_len > 0:
                options.append(["L"])
            if right_len > 0:
                options.append(["R"])
            if left_len > 0 and right_len > 0:
                options.append(["L", "R"])
        else:
            up_len = _free_run_length(node, "U", height, width, used_nodes, used_edges)
            down_len = _free_run_length(node, "D", height, width, used_nodes, used_edges)
            if up_len > 0:
                options.append(["U"])
            if down_len > 0:
                options.append(["D"])

        if not options:
            continue

        branch_dirs = rng.choice(options)
        for direction in branch_dirs:
            max_len = _free_run_length(node, direction, height, width, occupied_nodes | local_nodes, occupied_edges | local_edges)
            _extend_branch(node, direction, rng.randint(1, min(3, max_len)), local_nodes, local_edges)
        valves.add(node)

    return valves


def _add_reachable_branches(
    path: Sequence[Tuple[int, int]],
    valves: Set[Tuple[int, int]],
    local_nodes: Set[Tuple[int, int]],
    local_edges: Set[Tuple[Tuple[int, int], Tuple[int, int]]],
    occupied_nodes: Set[Tuple[int, int]],
    occupied_edges: Set[Tuple[Tuple[int, int], Tuple[int, int]]],
    height: int,
    width: int,
    rng: random.Random,
) -> None:
    if rng.random() < 0.35:
        return

    target = rng.randint(1, 2)
    candidates = [node for node in path[1:] if node not in valves]
    rng.shuffle(candidates)
    added = 0
    for node in candidates:
        if added >= target:
            break
        incident = _incident_dirs(node, local_edges)
        options = []
        for direction in "UDLR":
            if direction in incident:
                continue
            max_len = _free_run_length(
                node,
                direction,
                height,
                width,
                occupied_nodes | local_nodes,
                occupied_edges | local_edges,
                limit=2,
            )
            if max_len > 0:
                options.append((direction, max_len))
        if not options:
            continue
        direction, max_len = rng.choice(options)
        _extend_branch(node, direction, rng.randint(1, max_len), local_nodes, local_edges)
        added += 1


def _place_tree(
    side: str,
    color: int,
    height: int,
    width: int,
    occupied_nodes: Set[Tuple[int, int]],
    occupied_edges: Set[Tuple[Tuple[int, int], Tuple[int, int]]],
    occupied_sources: Set[Tuple[int, int]],
    rng: random.Random,
) -> Dict[str, object] | None:
    starts = _start_nodes_for_side(side, height, width)
    rng.shuffle(starts)
    for _ in range(TREE_RETRIES):
        if not starts:
            break
        start = rng.choice(starts)
        source = _source_cell(side, start, height, width)
        if source in occupied_sources or start in occupied_nodes:
            continue

        trunk = _build_trunk(start, SIDE_TO_DIR[side], height, width, occupied_nodes, occupied_edges, rng)
        if trunk is None:
            continue
        path, local_nodes, local_edges = trunk
        if path[0] != start:
            continue

        valves = _add_valves(path, local_nodes, local_edges, occupied_nodes, occupied_edges, height, width, rng)
        if not valves:
            continue

        _add_reachable_branches(path, valves, local_nodes, local_edges, occupied_nodes, occupied_edges, height, width, rng)
        return {
            "side": side,
            "source": source,
            "color": color,
            "nodes": local_nodes,
            "edges": local_edges,
            "valves": valves,
        }
    return None


def _paint_edge(
    grid: np.ndarray,
    a: Tuple[int, int],
    b: Tuple[int, int],
    color: int,
) -> None:
    mr = (a[0] + b[0]) // 2
    mc = (a[1] + b[1]) // 2
    grid[mr, mc] = color


def _rasterize(
    height: int,
    width: int,
    bg: int,
    trees: Sequence[Dict[str, object]],
) -> np.ndarray:
    grid = np.full((height, width), bg, dtype=int)
    for tree in trees:
        for node in tree["nodes"]:
            r, c = node
            grid[r, c] = GREY
        for a, b in tree["edges"]:
            _paint_edge(grid, a, b, GREY)
        for vr, vc in tree["valves"]:
            grid[vr, vc] = MAROON
        sr, sc = tree["source"]
        grid[sr, sc] = int(tree["color"])
    return grid


def _is_valid_generated(grid: np.ndarray) -> bool:
    out, meta = _solve(grid)
    bg = int(meta["bg"])
    sources = meta["sources"]
    valves = meta["valves"]
    reach_count = meta["reach_count"]

    if bg in (GREY, MAROON):
        return False
    if len(sources) < 1 or len(sources) > 3:
        return False
    if len(valves) == 0:
        return False
    if np.any(reach_count > 1):
        return False

    grey_input = grid == GREY
    painted_mask = grey_input & (out != GREY)
    blocked_mask = grey_input & (out == GREY)
    if not np.any(painted_mask):
        return False
    if not np.any(blocked_mask):
        return False

    source_colors = {color for _pos, color in sources}
    if len(source_colors) != len(sources):
        return False
    for _pos, color in sources:
        if not np.any((out == color) & grey_input):
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randrange(15, 24, 2)
    width = rng.randrange(15, 24, 2)
    bg = rng.choice(COLOR_POOL)
    source_count = rng.randint(1, 3)
    source_colors = rng.sample([c for c in COLOR_POOL if c != bg], source_count)

    if source_count == 1 or rng.random() < 0.75:
        sides = rng.sample(SOURCE_SIDES, source_count)
    else:
        sides = [rng.choice(SOURCE_SIDES) for _ in range(source_count)]

    occupied_nodes: Set[Tuple[int, int]] = set()
    occupied_edges: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()
    occupied_sources: Set[Tuple[int, int]] = set()
    trees: List[Dict[str, object]] = []

    for side, color in zip(sides, source_colors):
        tree = _place_tree(side, color, height, width, occupied_nodes, occupied_edges, occupied_sources, rng)
        if tree is None:
            return None
        occupied_nodes |= tree["nodes"]
        occupied_edges |= tree["edges"]
        occupied_sources.add(tree["source"])
        trees.append(tree)

    grid = _rasterize(height, width, bg, trees)
    if not _is_valid_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
