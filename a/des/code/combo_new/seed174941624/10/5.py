from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
END = 1
START = 2
WALL = 3
PATH = 4
SKELETON = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
STEP_CHOICES = (2, 4, 6)
NODE_COUNT_RANGE = (3, 6)
MARGIN_RANGE = (1, 4)
GEN_RETRIES = 400

Coord = tuple[int, int]
Node = tuple[int, int]
Edge = tuple[Node, Node]


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(int(x) for x in grid.flat).most_common(1)[0][0])


def _orth_neighbors(mask: np.ndarray, r: int, c: int) -> list[Coord]:
    h, w = mask.shape
    out: list[Coord] = []
    for dr, dc in DIR4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]):
            out.append((rr, cc))
    return out


def _is_straight_pixel(mask: np.ndarray, r: int, c: int) -> bool:
    neighbors = _orth_neighbors(mask, r, c)
    if len(neighbors) != 2:
        return False
    dr0 = neighbors[0][0] - r
    dc0 = neighbors[0][1] - c
    dr1 = neighbors[1][0] - r
    dc1 = neighbors[1][1] - c
    return dr0 + dr1 == 0 and dc0 + dc1 == 0


def _node_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    for r, c in np.argwhere(mask):
        out[int(r), int(c)] = not _is_straight_pixel(mask, int(r), int(c))
    return out


def _trace_wall_segment(mask: np.ndarray, node_mask: np.ndarray, marker: Coord) -> set[Coord]:
    r, c = marker
    neighbors = _orth_neighbors(mask, r, c)
    assert len(neighbors) == 2 and _is_straight_pixel(mask, r, c), "wall marker must sit on a straight segment"

    dr = neighbors[0][0] - r
    dc = neighbors[0][1] - c
    cells = {(int(r), int(c))}

    for sign in (-1, 1):
        rr = r + sign * dr
        cc = c + sign * dc
        while 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and bool(mask[rr, cc]):
            if bool(node_mask[rr, cc]):
                break
            cells.add((int(rr), int(cc)))
            rr += sign * dr
            cc += sign * dc
        else:
            raise AssertionError("wall segment must terminate at an intersection")
        assert bool(node_mask[rr, cc]), "wall segment must end next to an intersection"

    return cells


def _bfs_path(mask: np.ndarray, start: Coord, end: Coord) -> list[Coord]:
    queue: deque[Coord] = deque([start])
    parent: dict[Coord, Coord | None] = {start: None}

    while queue:
        cur = queue.popleft()
        if cur == end:
            break
        for nxt in _orth_neighbors(mask, cur[0], cur[1]):
            if nxt in parent:
                continue
            parent[nxt] = cur
            queue.append(nxt)

    assert end in parent, "start and end must remain connected after removing walls"

    path: list[Coord] = []
    cur: Coord | None = end
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the green-marked wall segments from the gray lattice and highlight the unique route between the two colored intersections.

    Input structure:
    - The background is black.
    - Gray pixels form a one-cell-wide orthogonal lattice whose straight segments connect neighboring intersections.
    - Two non-gray, non-green colored pixels sit on intersection cells and mark the start and end.
    - Green pixels sit at the midpoints of some straight lattice segments and mark those segments for deletion.

    Rule:
    - Treat every green midpoint as selecting its whole straight segment between the two adjacent intersections.
    - Erase that segment to black, including the green midpoint itself, while keeping the endpoint intersections.
    - In the remaining lattice, find the unique 4-connected path from the start pixel to the end pixel.
    - Recolor every gray pixel on that path to yellow, but keep the start and end pixels in their original colors.

    Key insight:
    - The green pixels do not mark the desired route; they remove corridors first, so the yellow route must be found only after interpreting each green midpoint as a deleted edge of the lattice graph.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    non_bg = [int(color) for color in np.unique(g) if int(color) != bg]
    assert non_bg, "expected a foreground lattice"
    skeleton = max(non_bg, key=lambda color: int(np.count_nonzero(g == color)))

    occupied = g != bg
    node_mask = _node_mask(occupied)

    wall_markers: list[Coord] = []
    endpoints: list[Coord] = []
    for r, c in np.argwhere((g != bg) & (g != skeleton)):
        coord = (int(r), int(c))
        if _is_straight_pixel(occupied, coord[0], coord[1]):
            wall_markers.append(coord)
        else:
            endpoints.append(coord)

    assert len(endpoints) == 2, "expected exactly two colored intersection endpoints"

    out = g.copy()
    for marker in wall_markers:
        for rr, cc in _trace_wall_segment(occupied, node_mask, marker):
            out[rr, cc] = bg

    maze_mask = out != bg
    start, end = endpoints
    path = _bfs_path(maze_mask, start, end)

    for r, c in path:
        if int(out[r, c]) == int(skeleton):
            out[r, c] = PATH

    return out


def _edge_key(a: Node, b: Node) -> Edge:
    return (a, b) if a <= b else (b, a)


def _grid_nodes(n_rows: int, n_cols: int) -> list[Node]:
    return [(r, c) for r in range(n_rows) for c in range(n_cols)]


def _grid_edges(n_rows: int, n_cols: int) -> list[Edge]:
    edges: list[Edge] = []
    for r in range(n_rows):
        for c in range(n_cols):
            if r + 1 < n_rows:
                edges.append(_edge_key((r, c), (r + 1, c)))
            if c + 1 < n_cols:
                edges.append(_edge_key((r, c), (r, c + 1)))
    return edges


def _sample_layout() -> tuple[int, int, int, int, int, int, int, int]:
    for _ in range(200):
        n_rows = random.randint(*NODE_COUNT_RANGE)
        n_cols = random.randint(*NODE_COUNT_RANGE)
        row_step = random.choice(STEP_CHOICES)
        col_step = random.choice(STEP_CHOICES)
        top = random.randint(MARGIN_RANGE[0], MARGIN_RANGE[1] - 1)
        bottom = random.randint(MARGIN_RANGE[0], MARGIN_RANGE[1] - 1)
        left = random.randint(MARGIN_RANGE[0], MARGIN_RANGE[1] - 1)
        right = random.randint(MARGIN_RANGE[0], MARGIN_RANGE[1] - 1)
        height = top + (n_rows - 1) * row_step + bottom + 1
        width = left + (n_cols - 1) * col_step + right + 1
        if height <= 30 and width <= 30:
            return n_rows, n_cols, row_step, col_step, top, left, height, width
    raise RuntimeError("failed to sample a valid lattice layout")


def _node_positions(
    n_rows: int,
    n_cols: int,
    row_step: int,
    col_step: int,
    top: int,
    left: int,
) -> dict[Node, Coord]:
    return {
        (r, c): (top + r * row_step, left + c * col_step)
        for r in range(n_rows)
        for c in range(n_cols)
    }


def _edge_inner_pixels(a: Node, b: Node, positions: dict[Node, Coord]) -> list[Coord]:
    (r0, c0) = positions[a]
    (r1, c1) = positions[b]
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        return [(r0, c) for c in range(lo + 1, hi)]
    lo, hi = sorted((r0, r1))
    return [(r, c0) for r in range(lo + 1, hi)]


def _edge_pixels(a: Node, b: Node, positions: dict[Node, Coord]) -> list[Coord]:
    return [positions[a], * _edge_inner_pixels(a, b, positions), positions[b]]


def _edge_midpoint(a: Node, b: Node, positions: dict[Node, Coord]) -> Coord:
    (r0, c0) = positions[a]
    (r1, c1) = positions[b]
    return ((r0 + r1) // 2, (c0 + c1) // 2)


class _DSU:
    def __init__(self, nodes: list[Node]) -> None:
        self.parent = {node: node for node in nodes}

    def find(self, node: Node) -> Node:
        parent = self.parent[node]
        if parent != node:
            self.parent[node] = self.find(parent)
        return self.parent[node]

    def union(self, a: Node, b: Node) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        self.parent[rb] = ra
        return True


def _random_tree(nodes: list[Node], edges: list[Edge]) -> set[Edge]:
    shuffled = edges[:]
    random.shuffle(shuffled)
    dsu = _DSU(nodes)
    tree: set[Edge] = set()
    for edge in shuffled:
        if dsu.union(edge[0], edge[1]):
            tree.add(edge)
    assert len(tree) == len(nodes) - 1
    return tree


def _adjacency(nodes: list[Node], edges: set[Edge]) -> dict[Node, list[Node]]:
    adj = {node: [] for node in nodes}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    return adj


def _node_path(adj: dict[Node, list[Node]], start: Node, end: Node) -> list[Node]:
    queue: deque[Node] = deque([start])
    parent: dict[Node, Node | None] = {start: None}
    while queue:
        cur = queue.popleft()
        if cur == end:
            break
        for nxt in adj[cur]:
            if nxt in parent:
                continue
            parent[nxt] = cur
            queue.append(nxt)
    assert end in parent
    path: list[Node] = []
    cur: Node | None = end
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _best_endpoint_pairs(nodes: list[Node], adj: dict[Node, list[Node]]) -> list[tuple[Node, Node, list[Node]]]:
    scored: list[tuple[int, int, int, Node, Node, list[Node]]] = []
    for i, start in enumerate(nodes):
        queue: deque[Node] = deque([start])
        parent: dict[Node, Node | None] = {start: None}
        while queue:
            cur = queue.popleft()
            for nxt in adj[cur]:
                if nxt in parent:
                    continue
                parent[nxt] = cur
                queue.append(nxt)
        for end in nodes[i + 1 :]:
            path: list[Node] = []
            cur: Node | None = end
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            dirs = {
                (path[k + 1][0] - path[k][0], path[k + 1][1] - path[k][1])
                for k in range(len(path) - 1)
            }
            has_turn = int(len(dirs) >= 2)
            scored.append((len(path), has_turn, random.randint(0, 10**6), start, end, path))

    assert scored
    scored.sort(reverse=True)
    best_len = scored[0][0]
    candidates = [item for item in scored if item[0] >= max(4, best_len - 1)]
    candidates.sort(key=lambda item: (item[1], item[0], item[2]), reverse=True)
    top_turn = candidates[0][1]
    return [(a, b, path) for _, turn, _, a, b, path in candidates if turn == top_turn]


def _render_input(
    n_rows: int,
    n_cols: int,
    positions: dict[Node, Coord],
    tree_edges: set[Edge],
    start_node: Node,
    end_node: Node,
    height: int,
    width: int,
) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    all_edges = _grid_edges(n_rows, n_cols)
    for edge in all_edges:
        for r, c in _edge_pixels(edge[0], edge[1], positions):
            grid[r, c] = SKELETON

    for edge in all_edges:
        if edge in tree_edges:
            continue
        mr, mc = _edge_midpoint(edge[0], edge[1], positions)
        grid[mr, mc] = WALL

    sr, sc = positions[start_node]
    er, ec = positions[end_node]
    grid[sr, sc] = START
    grid[er, ec] = END
    return grid


def _render_output(
    n_rows: int,
    n_cols: int,
    positions: dict[Node, Coord],
    tree_edges: set[Edge],
    start_node: Node,
    end_node: Node,
    path_nodes: list[Node],
    height: int,
    width: int,
) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for edge in tree_edges:
        for r, c in _edge_pixels(edge[0], edge[1], positions):
            grid[r, c] = SKELETON

    for node in path_nodes[1:-1]:
        r, c = positions[node]
        grid[r, c] = PATH

    for a, b in zip(path_nodes, path_nodes[1:]):
        for r, c in _edge_inner_pixels(a, b, positions):
            grid[r, c] = PATH

    sr, sc = positions[start_node]
    er, ec = positions[end_node]
    grid[sr, sc] = START
    grid[er, ec] = END
    return grid


def _tree_quality_ok(nodes: list[Node], tree_edges: set[Edge], all_edges: list[Edge]) -> bool:
    deg = Counter()
    for a, b in tree_edges:
        deg[a] += 1
        deg[b] += 1
    if max(deg.values()) < 3:
        return False

    removed = [edge for edge in all_edges if edge not in tree_edges]
    removed_h = sum(edge[0][0] == edge[1][0] for edge in removed)
    removed_v = len(removed) - removed_h
    return removed_h > 0 and removed_v > 0 and len(removed) >= 3 and len(nodes) >= 9


def _attempt() -> np.ndarray | None:
    n_rows, n_cols, row_step, col_step, top, left, height, width = _sample_layout()
    nodes = _grid_nodes(n_rows, n_cols)
    all_edges = _grid_edges(n_rows, n_cols)
    positions = _node_positions(n_rows, n_cols, row_step, col_step, top, left)

    tree_edges = _random_tree(nodes, all_edges)
    if not _tree_quality_ok(nodes, tree_edges, all_edges):
        return None

    adj = _adjacency(nodes, tree_edges)
    pairs = _best_endpoint_pairs(nodes, adj)
    start_node, end_node, path_nodes = random.choice(pairs)

    inp = _render_input(n_rows, n_cols, positions, tree_edges, start_node, end_node, height, width)
    expected = _render_output(n_rows, n_cols, positions, tree_edges, start_node, end_node, path_nodes, height, width)

    try:
        solved = main(inp)
    except AssertionError:
        return None

    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lattice-maze input")
