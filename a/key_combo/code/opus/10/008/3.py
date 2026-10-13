from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

RNG = np.random.default_rng()
NODE_SIZE = 2
MAX_GEN_ATTEMPTS = 400
ODD_GAPS = (5, 7)
EVEN_GAP = 4
MARGIN_CHOICES = (1, 2, 3)


@dataclass(frozen=True)
class Node:
    idx: int
    r: int
    c: int
    terminal: bool


TEMPLATE_6 = {
    "points": ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 3)),
    "terminal_pairs": ((0, 2), (2, 4)),
}

TEMPLATE_7 = {
    "points": ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 3)),
    "terminal_pairs": ((0, 2), (0, 3), (2, 5), (3, 5)),
}

TEMPLATES = (TEMPLATE_6, TEMPLATE_7)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw parity-controlled side rays between 2x2 node blocks, then recolor the unique terminal-to-terminal shortest path.

    Input structure:
    - A black background contains 6-7 blue 2x2 solid blocks that act as graph nodes.
    - Exactly two node blocks are terminals: their top-left cell is red while the other three cells stay blue.
    - Nodes are axis-aligned and well separated, with some pairs sharing a row or a column so their side rays can face each other.

    Rule:
    - Treat each 2x2 block as a node whose four rays leave from the four block sides: left/up use the block's top row/left column, right/down use the bottom row/right column.
    - Along every ray, the first cell outside the node is grey, the next is blue, then grey/blue alternates until the grid edge or the nearest node on that ray.
    - Two row-aligned or column-aligned consecutive nodes are connected iff the distance to the encountered node side has even parity, equivalently iff their top-left coordinates differ by an odd amount along that axis.
    - Build the undirected graph from those parity-valid adjacencies, find the unique shortest path between the two terminals, recolor every non-terminal node on that path to green, and recolor all ray cells between adjacent path nodes to yellow.

    Key insight:
    - The graph is implicit: visible row/column alignment is not enough, because the alternating ray parity decides whether an aligned neighbor is an edge or only a non-connecting ray encounter.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    nodes = _detect_nodes(g)
    analysis = _analyze_nodes(nodes, g.shape)
    _assert_conflict_free(analysis["ray_colors"])

    term_ids = [node.idx for node in nodes if node.terminal]
    assert len(term_ids) == 2, "expected exactly two terminals"
    start, goal = term_ids
    node_path, edge_path = _unique_shortest_path(analysis["adj"], start, goal)

    out = g.copy()
    for (r, c), color in analysis["ray_colors"].items():
        out[r, c] = color

    terminal_set = set(term_ids)
    for node_id in node_path:
        if node_id in terminal_set:
            continue
        node = nodes[node_id]
        out[node.r : node.r + NODE_SIZE, node.c : node.c + NODE_SIZE] = GREEN

    for edge in edge_path:
        for r, c in analysis["edge_strips"][edge]:
            out[r, c] = YELLOW

    for node in nodes:
        if not node.terminal:
            continue
        out[node.r : node.r + NODE_SIZE, node.c : node.c + NODE_SIZE] = BLUE
        out[node.r, node.c] = RED

    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is None:
            continue
        _validate_generated_instance(grid)
        return grid
    raise RuntimeError("failed to generate a valid odd-even ray shortest-path instance")


def _generate_attempt() -> np.ndarray | None:
    template = TEMPLATES[int(RNG.integers(len(TEMPLATES)))]
    points = _transform_points(
        template["points"],
        rot=int(RNG.integers(4)),
        mirror=bool(RNG.integers(2)),
    )
    terminal_pair = template["terminal_pairs"][int(RNG.integers(len(template["terminal_pairs"])))]

    row_levels = sorted({r for r, _ in points})
    col_levels = sorted({c for _, c in points})
    row_map = _sample_axis_positions(row_levels)
    col_map = _sample_axis_positions(col_levels)
    if row_map is None or col_map is None:
        return None

    placed = [(row_map[r], col_map[c]) for r, c in points]
    max_r = max(r for r, _ in placed) + NODE_SIZE
    max_c = max(c for _, c in placed) + NODE_SIZE
    bottom = int(RNG.choice(MARGIN_CHOICES))
    right = int(RNG.choice(MARGIN_CHOICES))
    height = max_r + bottom
    width = max_c + right
    if height > 30 or width > 30:
        return None

    grid = np.zeros((height, width), dtype=int)
    terminal_ids = set(terminal_pair)
    for idx, (r, c) in enumerate(placed):
        grid[r : r + NODE_SIZE, c : c + NODE_SIZE] = BLUE
        if idx in terminal_ids:
            grid[r, c] = RED

    nodes = _detect_nodes(grid)
    analysis = _analyze_nodes(nodes, grid.shape)

    if len(nodes) not in (6, 7):
        return None
    if not _has_required_nonedge(analysis):
        return None
    if not _has_unique_shortest_with_alternative(nodes, analysis["adj"]):
        return None
    if not _assert_no_shortcut_degeneracy(nodes, analysis):
        return None
    if _has_conflicting_rays(analysis["ray_colors"]):
        return None

    out = main(grid)
    if not np.any(out == GREEN) or not np.any(out == YELLOW):
        return None
    return grid


def _detect_nodes(grid: np.ndarray) -> list[Node]:
    h, w = grid.shape
    occupied = np.zeros((h, w), dtype=bool)
    nodes: list[Node] = []
    for r in range(h - 1):
        for c in range(w - 1):
            block = grid[r : r + 2, c : c + 2]
            is_station = np.array_equal(block, np.full((2, 2), BLUE, dtype=int))
            is_terminal = np.array_equal(block, np.array([[RED, BLUE], [BLUE, BLUE]], dtype=int))
            if not (is_station or is_terminal):
                continue
            if occupied[r : r + 2, c : c + 2].any():
                raise AssertionError("overlapping node candidates")
            occupied[r : r + 2, c : c + 2] = True
            nodes.append(Node(idx=len(nodes), r=r, c=c, terminal=is_terminal))

    if not nodes:
        raise AssertionError("no nodes detected")
    if not np.array_equal((grid != BG), occupied):
        raise AssertionError("unexpected non-node foreground pixels present")
    return nodes


def _analyze_nodes(nodes: list[Node], shape: tuple[int, int]) -> dict:
    by_row: dict[int, list[Node]] = defaultdict(list)
    by_col: dict[int, list[Node]] = defaultdict(list)
    for node in nodes:
        by_row[node.r].append(node)
        by_col[node.c].append(node)
    for row_nodes in by_row.values():
        row_nodes.sort(key=lambda node: node.c)
    for col_nodes in by_col.values():
        col_nodes.sort(key=lambda node: node.r)

    adj: dict[int, set[int]] = {node.idx: set() for node in nodes}
    edge_strips: dict[frozenset[int], set[tuple[int, int]]] = {}
    ray_colors: dict[tuple[int, int], int] = {}

    row_pos = {node.idx: i for row in by_row.values() for i, node in enumerate(row)}
    col_pos = {node.idx: i for col in by_col.values() for i, node in enumerate(col)}

    for node in nodes:
        for direction in ("left", "right", "up", "down"):
            coords = _ray_cells(node, direction, by_row, by_col, shape, row_pos, col_pos)
            for step, cell in enumerate(coords, start=1):
                color = BLUE if step % 2 == 0 else GREY
                prev = ray_colors.get(cell)
                if prev is not None and prev != color:
                    ray_colors[cell] = -1
                else:
                    ray_colors[cell] = color

    for row_nodes in by_row.values():
        for left, right in zip(row_nodes, row_nodes[1:]):
            gap = right.c - left.c
            strip = _horizontal_strip(left, right)
            edge_key = frozenset((left.idx, right.idx))
            edge_strips[edge_key] = strip
            if gap % 2 == 1:
                adj[left.idx].add(right.idx)
                adj[right.idx].add(left.idx)

    for col_nodes in by_col.values():
        for upper, lower in zip(col_nodes, col_nodes[1:]):
            gap = lower.r - upper.r
            strip = _vertical_strip(upper, lower)
            edge_key = frozenset((upper.idx, lower.idx))
            edge_strips[edge_key] = strip
            if gap % 2 == 1:
                adj[upper.idx].add(lower.idx)
                adj[lower.idx].add(upper.idx)

    return {
        "adj": adj,
        "edge_strips": edge_strips,
        "ray_colors": ray_colors,
        "by_row": by_row,
        "by_col": by_col,
    }


def _ray_cells(
    node: Node,
    direction: str,
    by_row: dict[int, list[Node]],
    by_col: dict[int, list[Node]],
    shape: tuple[int, int],
    row_pos: dict[int, int],
    col_pos: dict[int, int],
) -> list[tuple[int, int]]:
    h, w = shape
    if direction == "left":
        row_nodes = by_row[node.r]
        pos = row_pos[node.idx]
        stop = row_nodes[pos - 1].c + NODE_SIZE if pos > 0 else 0
        return [(node.r, c) for c in range(node.c - 1, stop - 1, -1)]
    if direction == "right":
        row_nodes = by_row[node.r]
        pos = row_pos[node.idx]
        stop = row_nodes[pos + 1].c - 1 if pos + 1 < len(row_nodes) else w - 1
        return [(node.r + 1, c) for c in range(node.c + NODE_SIZE, stop + 1)]
    if direction == "up":
        col_nodes = by_col[node.c]
        pos = col_pos[node.idx]
        stop = col_nodes[pos - 1].r + NODE_SIZE if pos > 0 else 0
        return [(r, node.c) for r in range(node.r - 1, stop - 1, -1)]
    if direction == "down":
        col_nodes = by_col[node.c]
        pos = col_pos[node.idx]
        stop = col_nodes[pos + 1].r - 1 if pos + 1 < len(col_nodes) else h - 1
        return [(r, node.c + 1) for r in range(node.r + NODE_SIZE, stop + 1)]
    raise ValueError(f"unknown direction: {direction}")


def _horizontal_strip(left: Node, right: Node) -> set[tuple[int, int]]:
    cells = set()
    for c in range(left.c + NODE_SIZE, right.c):
        cells.add((left.r, c))
        cells.add((left.r + 1, c))
    return cells


def _vertical_strip(upper: Node, lower: Node) -> set[tuple[int, int]]:
    cells = set()
    for r in range(upper.r + NODE_SIZE, lower.r):
        cells.add((r, upper.c))
        cells.add((r, upper.c + 1))
    return cells


def _unique_shortest_path(adj: dict[int, set[int]], start: int, goal: int) -> tuple[list[int], list[frozenset[int]]]:
    dist = {start: 0}
    q = deque([start])
    while q:
        u = q.popleft()
        for v in sorted(adj[u]):
            if v in dist:
                continue
            dist[v] = dist[u] + 1
            q.append(v)
    if goal not in dist:
        raise AssertionError("terminals are disconnected")

    ways = {start: 1}
    for u in sorted(dist, key=dist.get):
        for v in adj[u]:
            if dist.get(v) == dist[u] + 1:
                ways[v] = ways.get(v, 0) + ways[u]
    if ways.get(goal, 0) != 1:
        raise AssertionError("shortest path is not unique")

    path = [goal]
    cur = goal
    while cur != start:
        preds = [v for v in adj[cur] if dist.get(v) == dist[cur] - 1]
        if len(preds) != 1:
            raise AssertionError("path predecessor is ambiguous")
        cur = preds[0]
        path.append(cur)
    path.reverse()
    edge_path = [frozenset((a, b)) for a, b in zip(path, path[1:])]
    return path, edge_path


def _transform_points(points: Iterable[tuple[int, int]], rot: int, mirror: bool) -> tuple[tuple[int, int], ...]:
    pts = list(points)
    max_r = max(r for r, _ in pts)
    max_c = max(c for _, c in pts)
    transformed = pts[:]
    cur_max_r, cur_max_c = max_r, max_c
    for _ in range(rot % 4):
        transformed = [(c, cur_max_r - r) for r, c in transformed]
        cur_max_r, cur_max_c = cur_max_c, cur_max_r
    if mirror:
        transformed = [(r, cur_max_c - c) for r, c in transformed]
    min_r = min(r for r, _ in transformed)
    min_c = min(c for _, c in transformed)
    transformed = [(r - min_r, c - min_c) for r, c in transformed]
    return tuple(transformed)


def _sample_axis_positions(levels: list[int]) -> dict[int, int] | None:
    levels = sorted(levels)
    start = int(RNG.choice(MARGIN_CHOICES))
    actual = [start]
    for prev, cur in zip(levels, levels[1:]):
        parity = (cur - prev) % 2
        choices = ODD_GAPS if parity == 1 else (EVEN_GAP,)
        actual.append(actual[-1] + int(RNG.choice(choices)))
    if actual[-1] + NODE_SIZE + max(MARGIN_CHOICES) > 30:
        return None
    return dict(zip(levels, actual))


def _has_required_nonedge(analysis: dict) -> bool:
    by_row = analysis["by_row"]
    by_col = analysis["by_col"]
    for row_nodes in by_row.values():
        for left, right in zip(row_nodes, row_nodes[1:]):
            if (right.c - left.c) % 2 == 0:
                return True
    for col_nodes in by_col.values():
        for upper, lower in zip(col_nodes, col_nodes[1:]):
            if (lower.r - upper.r) % 2 == 0:
                return True
    return False


def _has_unique_shortest_with_alternative(nodes: list[Node], adj: dict[int, set[int]]) -> bool:
    terminals = [node.idx for node in nodes if node.terminal]
    if len(terminals) != 2:
        return False
    start, goal = terminals
    try:
        path, _ = _unique_shortest_path(adj, start, goal)
    except AssertionError:
        return False
    return len(path) >= 3 and _count_simple_paths(adj, start, goal) >= 2


def _count_simple_paths(adj: dict[int, set[int]], start: int, goal: int) -> int:
    count = 0

    def dfs(cur: int, seen: set[int]) -> None:
        nonlocal count
        if cur == goal:
            count += 1
            return
        for nxt in adj[cur]:
            if nxt in seen:
                continue
            seen.add(nxt)
            dfs(nxt, seen)
            seen.remove(nxt)

    dfs(start, {start})
    return count


def _assert_no_shortcut_degeneracy(nodes: list[Node], analysis: dict) -> bool:
    if not _has_required_nonedge(analysis):
        return False
    terminals = [node.idx for node in nodes if node.terminal]
    if len(terminals) != 2:
        return False
    try:
        path, _ = _unique_shortest_path(analysis["adj"], terminals[0], terminals[1])
    except AssertionError:
        return False
    return len(set(range(len(nodes))) - set(path)) >= 2


def _has_conflicting_rays(ray_colors: dict[tuple[int, int], int]) -> bool:
    return any(color == -1 for color in ray_colors.values())


def _assert_conflict_free(ray_colors: dict[tuple[int, int], int]) -> None:
    if _has_conflicting_rays(ray_colors):
        raise AssertionError("ray color conflict detected")


def _validate_generated_instance(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    assert max(grid.shape) <= 30

    nodes = _detect_nodes(grid)
    assert len(nodes) in (6, 7)
    assert sum(node.terminal for node in nodes) == 2

    analysis = _analyze_nodes(nodes, grid.shape)
    _assert_conflict_free(analysis["ray_colors"])
    assert _has_required_nonedge(analysis)
    assert _has_unique_shortest_with_alternative(nodes, analysis["adj"])
    assert _assert_no_shortcut_degeneracy(nodes, analysis)

    terminals = [node.idx for node in nodes if node.terminal]
    node_path, edge_path = _unique_shortest_path(analysis["adj"], terminals[0], terminals[1])
    assert len(node_path) >= 3
    assert len(edge_path) == len(node_path) - 1

    out = main(grid)
    assert isinstance(out, np.ndarray)
    assert out.shape == grid.shape
    assert np.any(out == GREEN)
    assert np.any(out == YELLOW)
