from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
TRACK = 5
TERMINAL = 7
PATH_COLORS = (RED, GREEN, BLUE)
SWATCH_ORDER = (RED, GREEN, BLUE)

NODE_SIZE = 2
TEMPLATE_SIZE = 3

TEMPLATE_LIBRARY = (
    np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool),
)


def _row_runs(values: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(values.tolist()):
        if value and start is None:
            start = int(i)
        elif not value and start is not None:
            runs.append((int(start), int(i - 1)))
            start = None
    if start is not None:
        runs.append((int(start), int(len(values) - 1)))
    return runs


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            out.append(np.asarray(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_solid_blocks(grid: np.ndarray, colors: tuple[int, ...]) -> list[dict]:
    blocks: list[dict] = []
    for color in colors:
        for coords in _components(grid == int(color)):
            if coords.shape[0] != NODE_SIZE * NODE_SIZE:
                continue
            r0, c0, r1, c1 = _bbox(coords)
            if (r1 - r0 + 1, c1 - c0 + 1) != (NODE_SIZE, NODE_SIZE):
                continue
            if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == int(color)):
                continue
            blocks.append(
                {
                    "color": int(color),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "coords": coords,
                }
            )
    return blocks


def _legend_templates(top: np.ndarray) -> dict[int, np.ndarray]:
    swatches = _find_solid_blocks(top, SWATCH_ORDER)
    assert len(swatches) == 3
    swatches.sort(key=lambda block: (block["bbox"][1], block["bbox"][0]))

    mapping: dict[int, np.ndarray] = {}
    for swatch in swatches:
        r0, c0, r1, c1 = swatch["bbox"]
        assert r0 == 0 and r1 == 1
        t0 = c1 + 2
        t1 = t0 + TEMPLATE_SIZE
        assert t1 <= top.shape[1]
        patch = top[:, t0:t1]
        assert patch.shape == (TEMPLATE_SIZE, TEMPLATE_SIZE)
        mask = patch == TRACK
        assert np.any(mask)
        mapping[int(swatch["color"])] = mask
    assert set(mapping) == set(SWATCH_ORDER)
    return mapping


def _graph_from_region(graph: np.ndarray) -> tuple[list[dict], dict[int, list[tuple[int, int]]], dict[int, np.ndarray]]:
    nodes = _find_solid_blocks(graph, PATH_COLORS + (TERMINAL,))
    assert len([node for node in nodes if int(node["color"]) == TERMINAL]) == 2

    pix_to_node: dict[tuple[int, int], int] = {}
    for node_id, node in enumerate(nodes):
        node["id"] = int(node_id)
        for r, c in node["coords"]:
            pix_to_node[(int(r), int(c))] = int(node_id)

    track_components = _components(graph == TRACK)
    adj: dict[int, list[tuple[int, int]]] = {int(node["id"]): [] for node in nodes}
    edge_pixels: dict[int, np.ndarray] = {}
    for edge_id, coords in enumerate(track_components):
        touched: set[int] = set()
        for r, c in coords:
            for nr, nc in _neighbors4(int(r), int(c)):
                node_id = pix_to_node.get((int(nr), int(nc)))
                if node_id is not None:
                    touched.add(int(node_id))
        assert len(touched) == 2
        a, b = sorted(touched)
        adj[int(a)].append((int(b), int(edge_id)))
        adj[int(b)].append((int(a), int(edge_id)))
        edge_pixels[int(edge_id)] = coords
    return nodes, adj, edge_pixels


def _ordered_terminals(nodes: list[dict]) -> tuple[int, int]:
    terminals = [node for node in nodes if int(node["color"]) == TERMINAL]
    assert len(terminals) == 2
    terminals.sort(key=lambda node: (node["bbox"][1], node["bbox"][0]))
    left, right = terminals
    if left["bbox"][1] == right["bbox"][1]:
        terminals.sort(key=lambda node: (node["bbox"][0], node["bbox"][1]))
        left, right = terminals
    return int(left["id"]), int(right["id"])


def _bfs_shortest(adj: dict[int, list[tuple[int, int]]], start: int) -> tuple[dict[int, int], dict[int, int]]:
    dist = {int(start): 0}
    count = {int(start): 1}
    q = deque([int(start)])
    while q:
        u = int(q.popleft())
        for v, _edge_id in adj[u]:
            nd = int(dist[u]) + 1
            if int(v) not in dist:
                dist[int(v)] = int(nd)
                count[int(v)] = int(count[u])
                q.append(int(v))
            elif int(dist[v]) == int(nd):
                count[int(v)] = int(count[v]) + int(count[u])
    return dist, count


def _unique_shortest_path(
    adj: dict[int, list[tuple[int, int]]],
    start: int,
    goal: int,
) -> tuple[list[int], list[int], int]:
    dist, count = _bfs_shortest(adj, int(start))
    assert int(goal) in dist and int(count[int(goal)]) == 1

    node_path = [int(goal)]
    edge_path: list[int] = []
    cur = int(goal)
    while cur != int(start):
        pred_options = [
            (int(v), int(edge_id))
            for v, edge_id in adj[cur]
            if int(v) in dist and int(dist[int(v)]) == int(dist[cur]) - 1
        ]
        assert len(pred_options) == 1
        prev, edge_id = pred_options[0]
        edge_path.append(int(edge_id))
        node_path.append(int(prev))
        cur = int(prev)
    node_path.reverse()
    edge_path.reverse()
    return node_path, edge_path, int(dist[int(goal)])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the unique shortest terminal-to-terminal route, map its station colors to legend templates, and concatenate the recolored templates.

    Input structure:
    - Black background separates an upper legend band from a lower graph band.
    - The legend band contains three 2x2 color swatches and, one blank column to the right of each swatch, a paired 3x3 grey template mask.
    - The graph band contains 2x2 solid node blocks in blue/red/green plus exactly two orange terminal blocks, linked by 1-pixel-wide grey horizontal or vertical track segments.

    Rule:
    - Build the node-edge graph from the lower band and find the unique shortest path between the two orange terminals, measured by number of track edges.
    - Read that path from the left terminal to the right terminal, or from the upper terminal to the lower one if both terminals share a column, and collect only the non-terminal node colors.
    - For each collected color, look up the paired 3x3 legend template, recolor grey template pixels to that node color, keep holes black, and place the resulting blocks left-to-right with a one-column black gap.

    Key insight:
    - The lower graph chooses both which station colors matter and their order, while the upper legend supplies the shape attached to each color; neither half of the input is sufficient on its own.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    row_runs = _row_runs(np.any(g != BG, axis=1))
    assert len(row_runs) == 2
    (top_r0, top_r1), (graph_r0, graph_r1) = row_runs
    top = g[top_r0 : top_r1 + 1]
    graph = g[graph_r0 : graph_r1 + 1]
    assert top.shape[0] == TEMPLATE_SIZE

    templates = _legend_templates(top)
    nodes, adj, _edge_pixels = _graph_from_region(graph)
    start, goal = _ordered_terminals(nodes)
    node_path, _edge_path, _ = _unique_shortest_path(adj, int(start), int(goal))

    color_sequence = [int(nodes[node_id]["color"]) for node_id in node_path[1:-1]]
    assert 2 <= len(color_sequence) <= 4

    width = len(color_sequence) * TEMPLATE_SIZE + (len(color_sequence) - 1)
    out = np.zeros((TEMPLATE_SIZE, width), dtype=int)
    for i, color in enumerate(color_sequence):
        mask = templates[int(color)]
        c0 = i * (TEMPLATE_SIZE + 1)
        out[:, c0 : c0 + TEMPLATE_SIZE][mask] = int(color)
    return out


def _count_simple_paths(adj: dict[int, set[int]], start: int, goal: int, limit: int = 3) -> int:
    count = 0
    stack = [(int(start), {int(start)})]
    while stack:
        node, seen = stack.pop()
        if node == int(goal):
            count += 1
            if count >= limit:
                return count
            continue
        for nxt in adj[node]:
            if int(nxt) not in seen:
                stack.append((int(nxt), seen | {int(nxt)}))
    return count


def _grow_cells(rows: int, cols: int, n_nodes: int, rng: random.Random) -> tuple[list[tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int]]]]:
    start = (rng.randrange(rows), rng.randrange(cols))
    cells = [start]
    cell_set = {start}
    tree_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    while len(cells) < n_nodes:
        frontier: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for cell in cells:
            r, c = cell
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < rows and 0 <= nc < cols and (int(nr), int(nc)) not in cell_set:
                    frontier.append((cell, (int(nr), int(nc))))
        assert frontier
        parent, child = rng.choice(frontier)
        if child in cell_set:
            continue
        cell_set.add(child)
        cells.append(child)
        tree_edges.append((parent, child))
    return cells, tree_edges


def _make_graph_spec(rng: random.Random):
    target_length = rng.choice((3, 4, 5))
    prefer_same_col = rng.random() < 0.4

    lattice_rows = 4 if (target_length == 5 or prefer_same_col) else rng.choice((3, 4))
    lattice_cols = 5 if target_length >= 4 else rng.choice((4, 5))
    min_nodes = 8 if target_length == 5 else 7 if target_length == 4 else 6
    n_nodes = rng.randint(min_nodes, min(9, lattice_rows * lattice_cols))
    cells, tree_edges = _grow_cells(lattice_rows, lattice_cols, n_nodes, rng)
    cell_set = set(cells)

    tree_edge_set = {tuple(sorted(edge)) for edge in tree_edges}
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r, c in cells:
        for nr, nc in ((r + 1, c), (r, c + 1)):
            if (int(nr), int(nc)) in cell_set:
                edge = tuple(sorted(((int(r), int(c)), (int(nr), int(nc)))))
                if edge not in tree_edge_set:
                    candidates.append(edge)

    rng.shuffle(candidates)
    if candidates:
        extra_min = 2 if target_length >= 4 and len(candidates) >= 2 else 1
        extra_count = rng.randint(extra_min, min(3, len(candidates)))
    else:
        extra_count = 0
    chosen_edges = list(tree_edges) + [tuple(edge) for edge in candidates[:extra_count]]

    index_of = {cell: i for i, cell in enumerate(cells)}
    adj: dict[int, set[int]] = {i: set() for i in range(len(cells))}
    for a_cell, b_cell in chosen_edges:
        a = int(index_of[a_cell])
        b = int(index_of[b_cell])
        adj[a].add(int(b))
        adj[b].add(int(a))

    valid_pairs: list[tuple[int, int, list[int], int]] = []
    for a in range(len(cells)):
        for b in range(a + 1, len(cells)):
            start_cell = cells[a]
            end_cell = cells[b]
            ordered = (a, b)
            if start_cell[1] > end_cell[1] or (start_cell[1] == end_cell[1] and start_cell[0] > end_cell[0]):
                ordered = (b, a)
            start, goal = ordered

            dist, count = _bfs_shortest({k: [(v, 0) for v in vs] for k, vs in adj.items()}, int(start))
            if int(goal) not in dist or not (3 <= int(dist[int(goal)]) <= 5):
                continue
            if int(count[int(goal)]) != 1:
                continue
            if _count_simple_paths(adj, int(start), int(goal), limit=3) < 2:
                continue

            path, _, length = _unique_shortest_path({k: [(v, 0) for v in vs] for k, vs in adj.items()}, int(start), int(goal))
            if len(path) != int(length) + 1:
                continue
            valid_pairs.append((int(start), int(goal), path, int(length)))

    if not valid_pairs:
        return None

    desired = [item for item in valid_pairs if int(item[3]) == int(target_length)]
    if prefer_same_col:
        same_col = [item for item in desired if cells[int(item[0])][1] == cells[int(item[1])][1]]
        if same_col:
            desired = same_col
    if not desired:
        return None

    start, goal, path, length = rng.choice(desired)
    return {
        "cells": cells,
        "edges": chosen_edges,
        "adj": adj,
        "start": int(start),
        "goal": int(goal),
        "path": [int(node) for node in path],
        "length": int(length),
        "shape": (int(lattice_rows), int(lattice_cols)),
    }


def _pick_templates(rng: random.Random) -> dict[int, np.ndarray]:
    masks = [mask.copy() for mask in rng.sample(TEMPLATE_LIBRARY, k=3)]
    return {color: mask for color, mask in zip(SWATCH_ORDER, masks)}


def _path_sequence(length: int, rng: random.Random) -> list[int]:
    count = int(length) - 1
    while True:
        seq = [int(rng.choice(PATH_COLORS)) for _ in range(count)]
        if len(set(seq)) >= 2:
            return seq


def _assign_node_colors(spec: dict, rng: random.Random) -> dict[int, int]:
    path = [int(node) for node in spec["path"]]
    path_internal = path[1:-1]
    seq = _path_sequence(int(spec["length"]), rng)

    colors: dict[int, int] = {int(spec["start"]): TERMINAL, int(spec["goal"]): TERMINAL}
    for node, color in zip(path_internal, seq):
        colors[int(node)] = int(color)

    remaining_nodes = [node for node in range(len(spec["cells"])) if node not in colors]
    remaining_needed = [color for color in PATH_COLORS if color not in seq]
    rng.shuffle(remaining_nodes)
    for node, color in zip(remaining_nodes, remaining_needed):
        colors[int(node)] = int(color)
    for node in remaining_nodes[len(remaining_needed) :]:
        colors[int(node)] = int(rng.choice(PATH_COLORS))

    all_nonterm = [color for node, color in colors.items() if int(color) != TERMINAL]
    assert set(PATH_COLORS).issubset(set(all_nonterm))
    assert len(set(seq)) >= 2
    return colors


def _render_graph(spec: dict, node_colors: dict[int, int], rng: random.Random) -> np.ndarray:
    step = rng.choice((4, 5))
    margin_r = rng.choice((0, 1))
    margin_c = rng.choice((0, 1, 2))
    rows, cols = spec["shape"]

    height = margin_r * 2 + (rows - 1) * step + NODE_SIZE
    width = margin_c * 2 + (cols - 1) * step + NODE_SIZE
    grid = np.zeros((height, width), dtype=int)

    pixel_pos: dict[int, tuple[int, int]] = {}
    for node, (r, c) in enumerate(spec["cells"]):
        rr = margin_r + int(r) * step
        cc = margin_c + int(c) * step
        pixel_pos[int(node)] = (int(rr), int(cc))
        grid[rr : rr + NODE_SIZE, cc : cc + NODE_SIZE] = int(node_colors[int(node)])

    cell_to_node = {cell: i for i, cell in enumerate(spec["cells"])}
    for a_cell, b_cell in spec["edges"]:
        a = int(cell_to_node[a_cell])
        b = int(cell_to_node[b_cell])
        ar, ac = pixel_pos[int(a)]
        br, bc = pixel_pos[int(b)]
        if ar == br:
            row = ar + 1
            c0 = min(ac, bc) + NODE_SIZE
            c1 = max(ac, bc)
            grid[row, c0:c1] = TRACK
        else:
            col = ac + 1
            r0 = min(ar, br) + NODE_SIZE
            r1 = max(ar, br)
            grid[r0:r1, col] = TRACK
    return grid


def _render_legend(template_map: dict[int, np.ndarray], rng: random.Random) -> np.ndarray:
    left_margin = rng.choice((0, 1, 2))
    between = [rng.randint(2, 4), rng.randint(2, 4)]
    pair_starts = [int(left_margin)]
    pair_width = NODE_SIZE + 1 + TEMPLATE_SIZE
    pair_starts.append(pair_starts[-1] + pair_width + between[0])
    pair_starts.append(pair_starts[-1] + pair_width + between[1])
    width = pair_starts[-1] + pair_width + rng.choice((0, 1, 2))

    grid = np.zeros((TEMPLATE_SIZE, width), dtype=int)
    for start, color in zip(pair_starts, SWATCH_ORDER):
        grid[0:NODE_SIZE, start : start + NODE_SIZE] = int(color)
        t0 = start + NODE_SIZE + 1
        grid[:, t0 : t0 + TEMPLATE_SIZE][template_map[int(color)]] = TRACK
    return grid


def _build_expected(template_map: dict[int, np.ndarray], path_colors: list[int]) -> np.ndarray:
    width = len(path_colors) * TEMPLATE_SIZE + (len(path_colors) - 1)
    out = np.zeros((TEMPLATE_SIZE, width), dtype=int)
    for i, color in enumerate(path_colors):
        c0 = i * (TEMPLATE_SIZE + 1)
        out[:, c0 : c0 + TEMPLATE_SIZE][template_map[int(color)]] = int(color)
    return out


def _semantic_checks(grid: np.ndarray, expected: np.ndarray) -> bool:
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False

    row_runs = _row_runs(np.any(grid != BG, axis=1))
    if len(row_runs) != 2:
        return False
    top = grid[row_runs[0][0] : row_runs[0][1] + 1]
    graph = grid[row_runs[1][0] : row_runs[1][1] + 1]
    if top.shape[0] != TEMPLATE_SIZE:
        return False

    try:
        templates = _legend_templates(top)
        nodes, adj, _edge_pixels = _graph_from_region(graph)
        start, goal = _ordered_terminals(nodes)
        path, _edges, dist = _unique_shortest_path(adj, int(start), int(goal))
    except AssertionError:
        return False

    if not (3 <= dist <= 5):
        return False
    if _count_simple_paths({k: {v for v, _eid in vs} for k, vs in adj.items()}, int(start), int(goal), limit=3) < 2:
        return False

    seq = [int(nodes[node]["color"]) for node in path[1:-1]]
    if not (2 <= len(seq) <= 4):
        return False
    if len(set(seq)) < 2:
        return False
    if set(PATH_COLORS) - {int(node["color"]) for node in nodes if int(node["color"]) != TERMINAL}:
        return False
    if not np.array_equal(_build_expected(templates, seq), expected):
        return False
    if not np.array_equal(main(grid), expected):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    spec = _make_graph_spec(rng)
    if spec is None:
        return None

    template_map = _pick_templates(rng)
    node_colors = _assign_node_colors(spec, rng)
    graph = _render_graph(spec, node_colors, rng)
    legend = _render_legend(template_map, rng)

    path_colors = [int(node_colors[int(node)]) for node in spec["path"][1:-1]]
    expected = _build_expected(template_map, path_colors)

    gap_rows = rng.randint(1, 3)
    graph_left = rng.choice((0, 1, 2, 3))
    legend_left = 0
    width = max(legend.shape[1] + legend_left, graph.shape[1] + graph_left) + rng.choice((0, 1, 2))
    height = legend.shape[0] + gap_rows + graph.shape[0] + rng.choice((0, 1, 2))
    if max(width, height) > 30:
        return None

    grid = np.zeros((height, width), dtype=int)
    grid[0 : legend.shape[0], legend_left : legend_left + legend.shape[1]] = legend
    r0 = legend.shape[0] + gap_rows
    grid[r0 : r0 + graph.shape[0], graph_left : graph_left + graph.shape[1]] = graph

    if not _semantic_checks(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid shortest-path template puzzle")
