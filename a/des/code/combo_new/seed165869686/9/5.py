from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np


Coord = Tuple[int, int]
DIRS: Tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
CONNECTOR = 5
PRIORITY_ORDER = (1, 2, 3, 4, 5, 6, 7, 8, 9)
PRIORITY_RANK = {color: rank for rank, color in enumerate(PRIORITY_ORDER)}

# --------------- solver helpers ---------------


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _in_bounds(grid: np.ndarray, row: int, col: int) -> bool:
    height, width = grid.shape
    return 0 <= row < height and 0 <= col < width


def _non_bg_neighbors(grid: np.ndarray, bg: int, coord: Coord) -> List[Coord]:
    row, col = coord
    neighbors: List[Coord] = []
    for dr, dc in DIRS:
        rr, cc = row + dr, col + dc
        if _in_bounds(grid, rr, cc) and int(grid[rr, cc]) != bg:
            neighbors.append((rr, cc))
    return neighbors


def _dir_from(src: Coord, dst: Coord) -> Coord:
    return int(dst[0] - src[0]), int(dst[1] - src[1])


def _opposite(a: Coord, b: Coord) -> bool:
    return a[0] == -b[0] and a[1] == -b[1]


def _is_logical_grey_node(grid: np.ndarray, bg: int, coord: Coord) -> bool:
    directions = [_dir_from(coord, nbr) for nbr in _non_bg_neighbors(grid, bg, coord)]
    if len(directions) != 2:
        return True
    return not _opposite(directions[0], directions[1])


def _logical_nodes(grid: np.ndarray, bg: int) -> Dict[Coord, str]:
    nodes: Dict[Coord, str] = {}
    for row, col in np.argwhere(grid != bg):
        coord = (int(row), int(col))
        color = int(grid[coord])
        if color == CONNECTOR:
            if _is_logical_grey_node(grid, bg, coord):
                nodes[coord] = "junction"
        else:
            nodes[coord] = "seed"
    return nodes


def _build_graph(grid: np.ndarray, bg: int, nodes: Dict[Coord, str]) -> Dict[Coord, Dict[Coord, Coord]]:
    adjacency: Dict[Coord, Dict[Coord, Coord]] = {coord: {} for coord in nodes}
    node_set = set(nodes)

    for node in nodes:
        row, col = node
        for dr, dc in DIRS:
            rr, cc = row + dr, col + dc
            if not _in_bounds(grid, rr, cc) or int(grid[rr, cc]) == bg:
                continue

            prev = node
            cur = (rr, cc)
            while cur not in node_set:
                nxts = []
                for ddr, ddc in DIRS:
                    nr, nc = cur[0] + ddr, cur[1] + ddc
                    if not _in_bounds(grid, nr, nc) or int(grid[nr, nc]) == bg or (nr, nc) == prev:
                        continue
                    nxts.append((nr, nc))
                assert len(nxts) == 1, "compressed line pixels must continue to exactly one next cell"
                prev, cur = cur, nxts[0]

            if cur == node:
                continue
            first_dir = (dr, dc)
            existing = adjacency[node].get(cur)
            assert existing is None or existing == first_dir, "duplicate edge must keep the same direction"
            adjacency[node][cur] = first_dir

    edge_count = sum(len(nbrs) for nbrs in adjacency.values()) // 2
    assert edge_count == len(nodes) - 1, "non-background structure must compress to one tree"
    return adjacency


def _merge_colors(a: int, b: int) -> int:
    if a == b:
        return int(a)
    return int(a if PRIORITY_RANK[a] < PRIORITY_RANK[b] else b)


def _find_root(grid: np.ndarray, adjacency: Dict[Coord, Dict[Coord, Coord]], nodes: Dict[Coord, str]) -> Coord:
    grey_nodes = [coord for coord, role in nodes.items() if role == "junction"]
    assert grey_nodes, "expected at least one grey internal node"
    roots = [coord for coord in grey_nodes if all(direction != (-1, 0) for direction in adjacency[coord].values())]
    assert len(roots) == 1, "expected exactly one root without an upward connection"
    root = roots[0]
    assert len(adjacency[root]) == 2, "root must have exactly two children"
    return root


def _final_color(
    grid: np.ndarray,
    adjacency: Dict[Coord, Dict[Coord, Coord]],
    nodes: Dict[Coord, str],
    node: Coord,
    parent: Coord | None,
) -> int:
    role = nodes[node]
    if role == "seed":
        return int(grid[node])

    if parent is None:
        children = list(adjacency[node])
    else:
        assert adjacency[node][parent] == (-1, 0), "non-root internal nodes must connect upward to their parent"
        children = [nbr for nbr in adjacency[node] if nbr != parent]

    assert len(children) == 2, "each internal node must have exactly two children"
    left = _final_color(grid, adjacency, nodes, children[0], node)
    right = _final_color(grid, adjacency, nodes, children[1], node)
    return _merge_colors(left, right)


def main(grid: np.ndarray) -> np.ndarray:
    """Collapse the grey connector tree upward and mark the root with the merged color.

    Input structure:
    - The grid has one dominant background color, several colored single-pixel seeds,
      and grey orthogonal connector segments.
    - The grey pixels form one connected acyclic binary tree after straight line runs are
      compressed; leaves are the colored seeds and internal nodes are grey branch pixels.
    - Exactly one grey internal node has no upward connector, making it the root.

    Rule:
    - Read the grey connectors as a tree, keeping the colored seeds as leaf colors.
    - For each internal node, merge its two child colors bottom-up: identical children keep
      that color, otherwise the winner is the earlier color in the ARC priority order
      blue, red, green, yellow, grey, magenta, orange, azure, maroon.
    - Clear all grey connector pixels back to the background, preserve the seeds, and paint
      the root pixel plus its four orthogonal neighbors with the root's final merged color.

    Key insight:
    - The grey pixels are not decoration; they define a tree that must be compressed into
      parent-child relations before the visible color-priority merge can be evaluated.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _bg_color(g)

    assert np.any(g == CONNECTOR), "expected grey connector pixels"
    nodes = _logical_nodes(g, bg)
    adjacency = _build_graph(g, bg, nodes)
    root = _find_root(g, adjacency, nodes)

    reachable = set()
    stack = [root]
    while stack:
        cur = stack.pop()
        if cur in reachable:
            continue
        reachable.add(cur)
        stack.extend(adjacency[cur])
    assert reachable == set(nodes), "every logical node must belong to the same tree"

    color = _final_color(g, adjacency, nodes, root, None)
    out = g.copy()
    out[out == CONNECTOR] = bg

    for dr, dc in ((0, 0),) + DIRS:
        rr, cc = root[0] + dr, root[1] + dc
        if _in_bounds(out, rr, cc):
            out[rr, cc] = color

    return out


# --------------- generator helpers ---------------

BG = 0
SEED_COLORS = (1, 2, 3)
DEPTH_RANGE = (1, 5)
VERTICAL_GAP_RANGE = (2, 4)
BOTTOM_GAP_RANGE = (2, 4)
HORIZONTAL_GAP_RANGE = (2, 6)
EXTRA_MARGIN_MAX = 6
MAX_SIDE_SPAN = HORIZONTAL_GAP_RANGE[1]
MAX_ATTEMPTS = 400


def _paint_line(grid: np.ndarray, start: Coord, end: Coord, color: int) -> None:
    r0, c0 = start
    r1, c1 = end
    assert r0 == r1 or c0 == c1
    if r0 == r1:
        for col in range(min(c0, c1), max(c0, c1) + 1):
            grid[r0, col] = color
    else:
        for row in range(min(r0, r1), max(r0, r1) + 1):
            grid[row, c0] = color


def _sample_leaf_colors(depth: int) -> Tuple[List[int], int]:
    bottom_seed_color = random.choice(SEED_COLORS)
    trunk_color = bottom_seed_color
    side_colors = [0] * depth
    for index in range(depth - 1, -1, -1):
        if random.random() < 0.5:
            side_color = trunk_color
        else:
            side_color = random.choice([c for c in SEED_COLORS if c != trunk_color])
        side_colors[index] = side_color
        trunk_color = _merge_colors(trunk_color, side_color)
    return side_colors, bottom_seed_color


def _build_attempt() -> np.ndarray | None:
    depth = random.randint(*DEPTH_RANGE)
    vertical_gaps = [random.randint(*VERTICAL_GAP_RANGE) for _ in range(depth - 1)]
    bottom_gap = random.randint(*BOTTOM_GAP_RANGE)
    side_lengths = [random.randint(*HORIZONTAL_GAP_RANGE) for _ in range(depth)]
    side_dirs = [random.choice((-1, 1)) for _ in range(depth)]

    node_rows = [0]
    for gap in vertical_gaps:
        node_rows.append(node_rows[-1] + gap)

    nodes_rel = [(row, 0) for row in node_rows]
    side_seeds_rel = [
        (node_rows[index], side_dirs[index] * side_lengths[index]) for index in range(depth)
    ]
    bottom_seed_rel = (node_rows[-1] + bottom_gap, 0)

    min_row = 0
    max_row = bottom_seed_rel[0]
    min_col = min(0, *(coord[1] for coord in side_seeds_rel))
    max_col = max(0, *(coord[1] for coord in side_seeds_rel))
    base_height = max_row - min_row + 1
    base_width = max_col - min_col + 1

    max_extra_h = min(EXTRA_MARGIN_MAX, 30 - base_height)
    max_extra_w = min(EXTRA_MARGIN_MAX, 30 - base_width)
    if max_extra_h < 2 or max_extra_w < 2:
        return None

    extra_h = random.randint(2, max_extra_h)
    extra_w = random.randint(2, max_extra_w)
    top_margin = random.randint(1, extra_h - 1)
    left_margin = random.randint(1, extra_w - 1)
    bottom_margin = extra_h - top_margin
    right_margin = extra_w - left_margin

    height = base_height + top_margin + bottom_margin
    width = base_width + left_margin + right_margin
    row_shift = top_margin - min_row
    col_shift = left_margin - min_col

    def shift(coord: Coord) -> Coord:
        return coord[0] + row_shift, coord[1] + col_shift

    grid = np.full((height, width), BG, dtype=int)

    nodes = [shift(coord) for coord in nodes_rel]
    side_seeds = [shift(coord) for coord in side_seeds_rel]
    bottom_seed = shift(bottom_seed_rel)

    for start, end in zip(nodes, nodes[1:]):
        _paint_line(grid, start, end, CONNECTOR)
    for node, seed in zip(nodes, side_seeds):
        _paint_line(grid, node, seed, CONNECTOR)
    _paint_line(grid, nodes[-1], bottom_seed, CONNECTOR)

    side_colors, bottom_seed_color = _sample_leaf_colors(depth)
    for seed, color in zip(side_seeds, side_colors):
        grid[seed] = color
    grid[bottom_seed] = bottom_seed_color

    try:
        out = main(grid)
    except AssertionError:
        return None

    root = nodes[0]
    cross = {root, (root[0] - 1, root[1]), (root[0] + 1, root[1]), (root[0], root[1] - 1), (root[0], root[1] + 1)}
    seed_set = set(side_seeds + [bottom_seed])
    if cross & seed_set:
        return None

    assert out.shape == grid.shape
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid tree-merge ARC input")
