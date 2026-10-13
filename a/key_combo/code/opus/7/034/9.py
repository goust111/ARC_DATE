from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 600


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(coords)
    return parts


def _find_nodes(grid: np.ndarray) -> list[dict]:
    nodes: list[dict] = []
    for coords in _components(grid == BLUE):
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if len(coords) != 4 or r1 - r0 != 1 or c1 - c0 != 1:
            continue
        nodes.append(
            {
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "pixels": [(r0, c0), (r0, c1), (r1, c0), (r1, c1)],
            }
        )
    nodes.sort(key=lambda node: (node["r0"], node["c0"]))
    for idx, node in enumerate(nodes):
        node["id"] = idx
    return nodes


def _trace_track(
    grid: np.ndarray,
    node_at: np.ndarray,
    start_node: int,
    starter: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[int, list[tuple[int, int]]] | None:
    dr, dc = direction
    r, c = starter
    pixels: list[tuple[int, int]] = []
    while _in_bounds(grid.shape, r, c) and grid[r, c] == RED:
        if node_at[r, c] >= 0:
            return None
        pixels.append((r, c))
        nr, nc = r + dr, c + dc
        if not _in_bounds(grid.shape, nr, nc):
            return None
        if node_at[nr, nc] >= 0:
            end_node = int(node_at[nr, nc])
            if end_node == start_node:
                return None
            return end_node, pixels
        r, c = nr, nc
    return None


def _analyze_graph(grid: np.ndarray) -> tuple[list[dict], list[dict], dict[int, int]]:
    g = np.asarray(grid, dtype=int)
    nodes = _find_nodes(g)
    assert len(nodes) >= 2, "expected at least two 2x2 blue nodes"

    node_at = np.full(g.shape, -1, dtype=int)
    for node in nodes:
        for r, c in node["pixels"]:
            node_at[r, c] = int(node["id"])

    edges_by_key: dict[tuple[int, int], dict] = {}
    side_specs = {
        "up": ((-1, 0), lambda n: [(n["r0"] - 1, n["c0"]), (n["r0"] - 1, n["c1"])]),
        "down": ((1, 0), lambda n: [(n["r1"] + 1, n["c0"]), (n["r1"] + 1, n["c1"])]),
        "left": ((0, -1), lambda n: [(n["r0"], n["c0"] - 1), (n["r1"], n["c0"] - 1)]),
        "right": ((0, 1), lambda n: [(n["r0"], n["c1"] + 1), (n["r1"], n["c1"] + 1)]),
    }

    for node in nodes:
        for direction, starter_fn in side_specs.values():
            starters = [
                cell
                for cell in starter_fn(node)
                if _in_bounds(g.shape, cell[0], cell[1]) and g[cell] == RED
            ]
            assert len(starters) <= 1, "each node side may launch at most one track"
            if not starters:
                continue
            traced = _trace_track(g, node_at, int(node["id"]), starters[0], direction)
            assert traced is not None, "every red starter must reach exactly one other node"
            end_node, pixels = traced
            a, b = sorted((int(node["id"]), int(end_node)))
            axis = "h" if direction[0] == 0 else "v"
            key = (a, b)
            if key in edges_by_key:
                assert set(edges_by_key[key]["pixels"]) == set(pixels), "duplicate edge disagrees"
                continue
            edges_by_key[key] = {"a": a, "b": b, "axis": axis, "pixels": pixels}

    edges = list(edges_by_key.values())
    degrees = {int(node["id"]): 0 for node in nodes}
    adjacency = {int(node["id"]): set() for node in nodes}
    for edge in edges:
        a, b = int(edge["a"]), int(edge["b"])
        degrees[a] += 1
        degrees[b] += 1
        adjacency[a].add(b)
        adjacency[b].add(a)

    seen = set()
    stack = [0]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(sorted(adjacency[cur] - seen))
    assert len(seen) == len(nodes), "graph must be connected"
    return nodes, edges, degrees


def _source_node(nodes: list[dict], degrees: dict[int, int], a: int, b: int) -> int:
    da, db = int(degrees[a]), int(degrees[b])
    if da != db:
        return a if da > db else b
    pa = (int(nodes[a]["r0"]), int(nodes[a]["c0"]))
    pb = (int(nodes[b]["r0"]), int(nodes[b]["c0"]))
    return a if pa < pb else b


def _ordered_edge_pixels(edge: dict, nodes: list[dict], source: int) -> list[tuple[int, int]]:
    a, b = int(edge["a"]), int(edge["b"])
    other = b if source == a else a
    pixels = list(edge["pixels"])
    if edge["axis"] == "h":
        ascending = int(nodes[source]["c0"]) < int(nodes[other]["c0"])
        return sorted(pixels, key=lambda rc: rc[1], reverse=not ascending)
    ascending = int(nodes[source]["r0"]) < int(nodes[other]["r0"])
    return sorted(pixels, key=lambda rc: rc[0], reverse=not ascending)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor graph nodes by degree parity and stamp each track with a degree-sized period from its source node.

    Input structure:
    - The grid has a black background, several 2x2 blue node blocks, and 1-pixel-wide red horizontal or vertical tracks.
    - Every red track is a straight segment connecting the edge of one blue block to the edge of another.
    - The blue blocks together with the red tracks form one connected undirected graph.

    Rule:
    - Detect the 2x2 blue blocks as nodes and count each node's degree from the incident red tracks.
    - Recolor odd-degree nodes to green and even-degree nodes to yellow.
    - For each track, choose the endpoint with larger degree as the source; if the degrees tie, choose the endpoint whose node is higher, then lefter.
    - Starting from the track pixel adjacent to that source node, recolor every D-th track pixel grey, where D is the source node degree; keep the other track pixels red.

    Key insight:
    - A single hidden property, the node degree, drives two visible outputs at once: parity chooses the node color and the numeric degree chooses the track-mark spacing.
    """
    g = np.asarray(grid, dtype=int)
    nodes, edges, degrees = _analyze_graph(g)
    out = g.copy()

    for node in nodes:
        color = GREEN if degrees[int(node["id"])] % 2 == 1 else YELLOW
        for r, c in node["pixels"]:
            out[r, c] = color

    for edge in edges:
        a, b = int(edge["a"]), int(edge["b"])
        source = _source_node(nodes, degrees, a, b)
        step = int(degrees[source])
        ordered = _ordered_edge_pixels(edge, nodes, source)
        for idx, (r, c) in enumerate(ordered):
            out[r, c] = GREY if idx % step == 0 else RED

    return out


def _lattice_neighbors(cell: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            out.append((nr, nc))
    return out


def _random_connected_cells(rows: int, cols: int, count: int) -> tuple[list[tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int]]]]:
    start = (random.randrange(rows), random.randrange(cols))
    chosen = {start}
    tree_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    while len(chosen) < count:
        frontier: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for cell in chosen:
            for nb in _lattice_neighbors(cell, rows, cols):
                if nb not in chosen:
                    frontier.append((cell, nb))
        parent, new_cell = random.choice(frontier)
        chosen.add(new_cell)
        tree_edges.append(tuple(sorted((parent, new_cell))))
    return sorted(chosen), tree_edges


def _candidate_extra_edges(cells: list[tuple[int, int]], tree_edges: list[tuple[tuple[int, int], tuple[int, int]]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    cell_set = set(cells)
    tree_set = set(tree_edges)
    extras = set()
    for cell in cells:
        for nb in _lattice_neighbors(cell, 99, 99):
            if nb in cell_set:
                edge = tuple(sorted((cell, nb)))
                if edge not in tree_set:
                    extras.add(edge)
    return sorted(extras)


def _graph_degrees(
    cells: list[tuple[int, int]],
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
) -> dict[tuple[int, int], int]:
    degrees = {cell: 0 for cell in cells}
    for a, b in edges:
        degrees[a] += 1
        degrees[b] += 1
    return degrees


def _choose_layout_counts(node_count: int) -> tuple[int, int] | None:
    options = [(r, c) for r in range(2, 5) for c in range(2, 5) if r * c >= node_count]
    return random.choice(options) if options else None


def _axis_positions(count: int) -> tuple[list[int], int]:
    start = random.randint(1, 3)
    positions = [start]
    for _ in range(count - 1):
        positions.append(positions[-1] + random.randint(6, 7))
    extent = positions[-1] + 2 + random.randint(1, 3)
    return positions, extent


def _blueprint_output(
    inp: np.ndarray,
    node_map: dict[tuple[int, int], dict],
    edge_specs: list[dict],
    degrees: dict[tuple[int, int], int],
) -> np.ndarray:
    out = inp.copy()
    for cell, node in node_map.items():
        color = GREEN if degrees[cell] % 2 == 1 else YELLOW
        r0, c0 = node["r0"], node["c0"]
        out[r0 : r0 + 2, c0 : c0 + 2] = color

    for edge in edge_specs:
        a, b = edge["a"], edge["b"]
        da, db = degrees[a], degrees[b]
        if da != db:
            source = a if da > db else b
        else:
            ra = (node_map[a]["r0"], node_map[a]["c0"])
            rb = (node_map[b]["r0"], node_map[b]["c0"])
            source = a if ra < rb else b
        step = degrees[source]
        pixels = list(edge["pixels"])
        if edge["axis"] == "h":
            ascending = node_map[source]["c0"] < node_map[a if source == b else b]["c0"]
            ordered = sorted(pixels, key=lambda rc: rc[1], reverse=not ascending)
        else:
            ascending = node_map[source]["r0"] < node_map[a if source == b else b]["r0"]
            ordered = sorted(pixels, key=lambda rc: rc[0], reverse=not ascending)
        for idx, (r, c) in enumerate(ordered):
            out[r, c] = GREY if idx % step == 0 else RED
    return out


def _render_input(
    cells: list[tuple[int, int]],
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
    row_positions: list[int],
    col_positions: list[int],
    height: int,
    width: int,
) -> tuple[np.ndarray, dict[tuple[int, int], dict], list[dict]]:
    grid = np.zeros((height, width), dtype=int)
    node_map: dict[tuple[int, int], dict] = {}
    for cell in cells:
        rr, cc = cell
        r0, c0 = row_positions[rr], col_positions[cc]
        grid[r0 : r0 + 2, c0 : c0 + 2] = BLUE
        node_map[cell] = {"r0": r0, "c0": c0}

    edge_specs: list[dict] = []
    for a, b in edges:
        ra, ca = a
        rb, cb = b
        if ra == rb:
            left, right = (a, b) if node_map[a]["c0"] < node_map[b]["c0"] else (b, a)
            row = node_map[left]["r0"] + random.randint(0, 1)
            c_start = node_map[left]["c0"] + 2
            c_stop = node_map[right]["c0"] - 1
            pixels = [(row, c) for c in range(c_start, c_stop + 1)]
            axis = "h"
        else:
            top, bottom = (a, b) if node_map[a]["r0"] < node_map[b]["r0"] else (b, a)
            col = node_map[top]["c0"] + random.randint(0, 1)
            r_start = node_map[top]["r0"] + 2
            r_stop = node_map[bottom]["r0"] - 1
            pixels = [(r, col) for r in range(r_start, r_stop + 1)]
            axis = "v"
        for r, c in pixels:
            assert grid[r, c] == BLACK
            grid[r, c] = RED
        edge_specs.append({"a": a, "b": b, "axis": axis, "pixels": pixels})
    return grid, node_map, edge_specs


def _valid_branch_mix(
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
    degrees: dict[tuple[int, int], int],
    mode: str,
) -> bool:
    tie = any(degrees[a] == degrees[b] for a, b in edges)
    non_tie = any(degrees[a] != degrees[b] for a, b in edges)
    if mode == "both":
        return tie and non_tie
    if mode == "tie":
        return tie
    return non_tie


def _gen_attempt() -> np.ndarray | None:
    node_count = random.randint(4, 6)
    layout = _choose_layout_counts(node_count)
    if layout is None:
        return None
    lattice_rows, lattice_cols = layout
    cells, tree_edges = _random_connected_cells(lattice_rows, lattice_cols, node_count)
    extras = _candidate_extra_edges(cells, tree_edges)
    extra_count = min(len(extras), random.choices([0, 1, 2], weights=[3, 3, 2])[0])
    random.shuffle(extras)
    edges = sorted(tree_edges + extras[:extra_count])
    degrees = _graph_degrees(cells, edges)

    if not any(deg % 2 == 1 for deg in degrees.values()):
        return None
    if not any(deg % 2 == 0 for deg in degrees.values()):
        return None

    branch_mode = random.choices(["both", "tie", "non_tie"], weights=[5, 2, 2])[0]
    if not _valid_branch_mix(edges, degrees, branch_mode):
        return None

    row_positions, height = _axis_positions(lattice_rows)
    col_positions, width = _axis_positions(lattice_cols)
    if max(height, width) > 30:
        return None

    try:
        inp, node_map, edge_specs = _render_input(
            cells, edges, row_positions, col_positions, height, width
        )
    except AssertionError:
        return None

    expected = _blueprint_output(inp, node_map, edge_specs, degrees)
    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(inp, actual):
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-9 input")
