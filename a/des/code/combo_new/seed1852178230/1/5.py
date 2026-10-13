import random
from typing import Dict, List, Tuple

import numpy as np


BG = 0
PATH_ORDER = (2, 3, 4)
PATH_RANK = {color: rank for rank, color in enumerate(PATH_ORDER)}
NODE_COLOR_POOL = (1, 5, 6, 7, 8, 9)
LADDER_SHAPES = ((3, 2), (2, 3))
MIN_STEP = 4
MAX_STEP = 8
MIN_MARGIN = 1
MAX_MARGIN = 6
MAX_SIDE = 30
GENERATION_ATTEMPTS = 400
EDGE_COLOR_ATTEMPTS = 400
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

Node = Tuple[int, int]
Edge = Tuple[Node, Node]


def _normalize_edge(a: Node, b: Node) -> Edge:
    return (a, b) if a <= b else (b, a)


def _monochrome_components(grid: np.ndarray) -> List[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: List[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: List[Tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": len(coords),
                }
            )
    return components


def _is_node_component(grid: np.ndarray, comp: dict) -> bool:
    if int(comp["area"]) != 4:
        return False
    r0, c0, r1, c1 = comp["bbox"]
    if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
        return False
    return bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == int(comp["color"])))


def _is_straight_path_component(comp: dict) -> bool:
    coords = comp["coords"]
    rows = sorted({r for r, _ in coords})
    cols = sorted({c for _, c in coords})
    if len(rows) == 1:
        ordered = sorted(c for _, c in coords)
        return ordered == list(range(ordered[0], ordered[-1] + 1))
    if len(cols) == 1:
        ordered = sorted(r for r, _ in coords)
        return ordered == list(range(ordered[0], ordered[-1] + 1))
    return False


def _parse_graph(grid: np.ndarray) -> Tuple[List[dict], List[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert int(g.min()) >= 0

    components = _monochrome_components(g)
    assert components, "expected at least one non-background component"

    nodes: List[dict] = []
    paths: List[dict] = []
    node_by_cell: Dict[Tuple[int, int], int] = {}

    for comp in components:
        if _is_node_component(g, comp):
            node_id = len(nodes)
            node = dict(comp)
            node["id"] = node_id
            nodes.append(node)
            for cell in comp["coords"]:
                node_by_cell[cell] = node_id
        else:
            assert _is_straight_path_component(comp), "every non-node component must be a straight path"
            path = dict(comp)
            path["id"] = len(paths)
            paths.append(path)

    assert nodes, "expected at least one 2x2 node"
    assert paths, "expected at least one path"
    node_colors = [int(node["color"]) for node in nodes]
    assert len(set(node_colors)) == len(node_colors), "node colors must be distinct"

    node_to_paths: List[List[int]] = [[] for _ in nodes]
    for path in paths:
        touching = set()
        for r, c in path["coords"]:
            for dr, dc in DIRS4:
                nbr = (r + dr, c + dc)
                if nbr in node_by_cell:
                    touching.add(node_by_cell[nbr])
        assert len(touching) == 2, "each path must connect exactly two nodes"
        a, b = sorted(touching)
        path["nodes"] = (a, b)
        node_to_paths[a].append(int(path["id"]))
        node_to_paths[b].append(int(path["id"]))
        assert int(path["color"]) in PATH_RANK, "paths must use the ordered path palette"

    for node, incident in zip(nodes, node_to_paths):
        degree = len(incident)
        assert degree in (2, 3), "every node must have degree 2 or 3"
        node["paths"] = tuple(incident)

    return nodes, paths


def _select_node_color(path_colors: List[int]) -> int:
    assert len(path_colors) in (2, 3)
    ordered = sorted(path_colors, key=lambda color: PATH_RANK[int(color)])
    if len(ordered) == 2:
        return int(ordered[0])
    return int(ordered[1])


def _later_color(a: int, b: int) -> int:
    return int(a) if PATH_RANK[int(a)] >= PATH_RANK[int(b)] else int(b)


def _abstract_nodes(rows: int, cols: int) -> List[Node]:
    return [(r, c) for r in range(rows) for c in range(cols)]


def _abstract_edges(rows: int, cols: int) -> List[Edge]:
    edges: List[Edge] = []
    for r in range(rows):
        for c in range(cols):
            if r + 1 < rows:
                edges.append(_normalize_edge((r, c), (r + 1, c)))
            if c + 1 < cols:
                edges.append(_normalize_edge((r, c), (r, c + 1)))
    return edges


def _incident_edges(nodes: List[Node], edges: List[Edge]) -> Dict[Node, List[Edge]]:
    incident = {node: [] for node in nodes}
    for edge in edges:
        a, b = edge
        incident[a].append(edge)
        incident[b].append(edge)
    return incident


def _sample_edge_colors(nodes: List[Node], edges: List[Edge], incident: Dict[Node, List[Edge]]) -> Dict[Edge, int] | None:
    degree2_nodes = [node for node in nodes if len(incident[node]) == 2]
    degree3_nodes = [node for node in nodes if len(incident[node]) == 3]
    assert degree2_nodes and degree3_nodes

    for _ in range(EDGE_COLOR_ATTEMPTS):
        edge_colors = {edge: random.choice(PATH_ORDER) for edge in edges}
        if set(edge_colors.values()) != set(PATH_ORDER):
            continue

        node_outputs = {}
        for node in nodes:
            incident_colors = [edge_colors[edge] for edge in incident[node]]
            node_outputs[node] = _select_node_color(incident_colors)

        if set(node_outputs.values()) != set(PATH_ORDER):
            continue
        if len({node_outputs[node] for node in degree2_nodes}) < 2:
            continue
        if len({tuple(sorted(edge_colors[edge] for edge in incident[node])) for node in degree3_nodes}) < 2:
            continue

        path_outputs = {}
        same_count = 0
        diff_count = 0
        for edge in edges:
            a, b = edge
            ca, cb = node_outputs[a], node_outputs[b]
            if ca == cb:
                same_count += 1
                path_outputs[edge] = ca
            else:
                diff_count += 1
                path_outputs[edge] = _later_color(ca, cb)

        if same_count == 0 or diff_count == 0:
            continue
        if len(set(path_outputs.values())) < 2:
            continue
        return edge_colors
    return None


def _sample_axis_positions(count: int) -> List[int]:
    coords = [0]
    for _ in range(count - 1):
        coords.append(coords[-1] + random.randint(MIN_STEP, MAX_STEP))
    return coords


def _sample_margin(extent: int) -> Tuple[int, int]:
    available = MAX_SIDE - extent
    assert available >= 2
    lead_max = min(MAX_MARGIN, available - 1)
    lead = random.randint(MIN_MARGIN, lead_max)
    tail_max = min(MAX_MARGIN, MAX_SIDE - extent - lead)
    tail = random.randint(MIN_MARGIN, tail_max)
    return lead, tail


def _draw_graph(rows: int, cols: int, edge_colors: Dict[Edge, int]) -> np.ndarray:
    row_offsets = _sample_axis_positions(rows)
    col_offsets = _sample_axis_positions(cols)
    graph_h = row_offsets[-1] + 2
    graph_w = col_offsets[-1] + 2
    top, bottom = _sample_margin(graph_h)
    left, right = _sample_margin(graph_w)

    h = graph_h + top + bottom
    w = graph_w + left + right
    assert max(h, w) <= MAX_SIDE

    grid = np.full((h, w), BG, dtype=int)
    node_positions = {
        (r, c): (top + row_offsets[r], left + col_offsets[c])
        for r in range(rows)
        for c in range(cols)
    }
    node_colors = {
        node: color for node, color in zip(sorted(node_positions), random.sample(list(NODE_COLOR_POOL), rows * cols))
    }

    for node, color in node_colors.items():
        rr, cc = node_positions[node]
        grid[rr : rr + 2, cc : cc + 2] = int(color)

    for edge, color in edge_colors.items():
        (r0, c0), (r1, c1) = edge
        if r0 == r1:
            row = node_positions[(r0, c0)][0] + (r0 % 2)
            start = node_positions[(r0, c0)][1] + 2
            stop = node_positions[(r1, c1)][1] - 1
            grid[row, start : stop + 1] = int(color)
        else:
            col = node_positions[(r0, c0)][1] + (c0 % 2)
            start = node_positions[(r0, c0)][0] + 2
            stop = node_positions[(r1, c1)][0] - 1
            grid[start : stop + 1, col] = int(color)

    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each 2x2 node from the ordered colors of its incident straight paths, then recolor every path from the updated endpoint nodes.

    Input structure:
    - The background is black.
    - Foreground objects form an orthogonal node-and-path graph.
    - Nodes are solid 2x2 monochrome blocks with distinct original colors.
    - Paths are one-cell-wide straight horizontal or vertical segments, and each node touches exactly 2 or 3 paths from the same fixed three-color path palette.

    Rule:
    - First update every node from the colors of the paths that touch it.
    - A degree-2 node takes the earlier-ranked of its two incident path colors, while a degree-3 node takes the middle-ranked of its three incident path colors.
    - Then update every path from the new colors of its two endpoint nodes: equal endpoints keep that color, otherwise the path takes the later-ranked endpoint color.
    - Shapes, positions, and the black background stay unchanged.

    Key insight:
    - The recoloring is a dependency chain on the graph topology: you must parse the 2x2 nodes and straight path components, resolve all node colors first, and only then can you determine each path color from the two updated endpoints.
    """
    g = np.asarray(grid, dtype=int)
    nodes, paths = _parse_graph(g)

    node_colors = {}
    for node in nodes:
        incident_colors = [int(paths[path_id]["color"]) for path_id in node["paths"]]
        node_colors[int(node["id"])] = _select_node_color(incident_colors)

    out = g.copy()
    for node in nodes:
        color = node_colors[int(node["id"])]
        for r, c in node["coords"]:
            out[r, c] = int(color)

    for path in paths:
        a, b = path["nodes"]
        ca = node_colors[int(a)]
        cb = node_colors[int(b)]
        color = int(ca) if ca == cb else _later_color(ca, cb)
        for r, c in path["coords"]:
            out[r, c] = int(color)

    return out


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        rows, cols = random.choice(LADDER_SHAPES)
        nodes = _abstract_nodes(rows, cols)
        edges = _abstract_edges(rows, cols)
        incident = _incident_edges(nodes, edges)
        edge_colors = _sample_edge_colors(nodes, edges, incident)
        if edge_colors is None:
            continue

        grid = _draw_graph(rows, cols, edge_colors)
        try:
            parsed_nodes, parsed_paths = _parse_graph(grid)
            assert len(parsed_nodes) == len(nodes)
            assert len(parsed_paths) == len(edges)
            out = main(grid)
            assert out.shape == grid.shape
            assert max(grid.shape) <= MAX_SIDE and max(out.shape) <= MAX_SIDE
            return grid
        except AssertionError:
            continue

    raise RuntimeError(f"Failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")
