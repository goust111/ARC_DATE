from __future__ import annotations

import random

import numpy as np


BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

DEGREE_TO_COLOR = {1: RED, 2: BLUE, 3: GREEN, 4: YELLOW}
SAFE_BACKGROUNDS = (0, 6, 8, 9)

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
PERPENDICULARS = {
    "up": ((0, -1), (0, 1)),
    "down": ((0, -1), (0, 1)),
    "left": ((-1, 0), (1, 0)),
    "right": ((-1, 0), (1, 0)),
}
SIDE_CENTER_OFFSET = {
    "up": (0, 1),
    "down": (2, 1),
    "left": (1, 0),
    "right": (1, 2),
}
OUTSIDE_OFFSET = {
    "up": (-1, 1),
    "down": (3, 1),
    "left": (1, -1),
    "right": (1, 3),
}
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}

MAX_GENERATION_ATTEMPTS = 400
TEMPLATE_KINDS = ("walk", "branch", "cycle", "t", "plus")


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _foreground_color(grid: np.ndarray, bg: int) -> int:
    colors = [int(color) for color in np.unique(grid) if int(color) != bg]
    assert len(colors) == 1, "expected exactly one foreground color in the input"
    return colors[0]


def _find_nodes(grid: np.ndarray, structure: int) -> tuple[list[tuple[int, int]], dict[tuple[int, int], int]]:
    h, w = grid.shape
    nodes: list[tuple[int, int]] = []
    cell_to_node: dict[tuple[int, int], int] = {}

    for r0 in range(h - 2):
        for c0 in range(w - 2):
            if not bool(np.all(grid[r0 : r0 + 3, c0 : c0 + 3] == structure)):
                continue
            idx = len(nodes)
            nodes.append((r0, c0))
            for rr in range(r0, r0 + 3):
                for cc in range(c0, c0 + 3):
                    key = (rr, cc)
                    assert key not in cell_to_node, "detected overlapping 3x3 nodes"
                    cell_to_node[key] = idx

    assert nodes, "expected at least one node"
    return nodes, cell_to_node


def _side_center(node: tuple[int, int], direction: str) -> tuple[int, int]:
    r0, c0 = node
    dr, dc = SIDE_CENTER_OFFSET[direction]
    return r0 + dr, c0 + dc


def _outside_cell(node: tuple[int, int], direction: str) -> tuple[int, int]:
    r0, c0 = node
    dr, dc = OUTSIDE_OFFSET[direction]
    return r0 + dr, c0 + dc


def _trace_edge(
    grid: np.ndarray,
    structure: int,
    nodes: list[tuple[int, int]],
    cell_to_node: dict[tuple[int, int], int],
    start_idx: int,
    direction: str,
) -> tuple[int, list[tuple[int, int]]] | None:
    start = _outside_cell(nodes[start_idx], direction)
    if not _in_bounds(grid.shape, start[0], start[1]) or int(grid[start]) != structure:
        return None
    assert start not in cell_to_node, "nodes must not touch directly without a line gap"

    dr, dc = DIRS[direction]
    line_cells: list[tuple[int, int]] = []
    r, c = start
    while True:
        assert _in_bounds(grid.shape, r, c), "line ran out of bounds"
        assert int(grid[r, c]) == structure, "line broke before reaching another node"
        assert (r, c) not in cell_to_node, "line cell unexpectedly entered a node"

        for pdr, pdc in PERPENDICULARS[direction]:
            rr, cc = r + pdr, c + pdc
            if _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == structure and (rr, cc) not in cell_to_node:
                raise AssertionError("lines may not branch away from the node network")

        line_cells.append((r, c))
        nr, nc = r + dr, c + dc
        assert _in_bounds(grid.shape, nr, nc), "line must terminate at another node"
        if (nr, nc) in cell_to_node:
            target_idx = cell_to_node[(nr, nc)]
            assert target_idx != start_idx, "line may not re-enter its source node"
            expected = _side_center(nodes[target_idx], OPPOSITE[direction])
            assert (nr, nc) == expected, "line must enter the far node through its side center"
            return target_idx, line_cells
        assert int(grid[nr, nc]) == structure, "line must be contiguous"
        r, c = nr, nc


def _parse_graph(
    grid: np.ndarray,
) -> tuple[int, int, list[tuple[int, int]], dict[tuple[int, int], int], dict[tuple[int, int], list[tuple[int, int]]], list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _dominant_color(g)
    structure = _foreground_color(g, bg)
    nodes, cell_to_node = _find_nodes(g, structure)

    covered = set(cell_to_node)
    degrees = [0] * len(nodes)
    edges: dict[tuple[int, int], list[tuple[int, int]]] = {}
    seen_half_edges: set[tuple[int, str]] = set()

    for node_idx, node in enumerate(nodes):
        neighbors_for_node: set[int] = set()
        for direction in ("up", "down", "left", "right"):
            seen_half_edges.add((node_idx, direction))
            traced = _trace_edge(g, structure, nodes, cell_to_node, node_idx, direction)
            if traced is None:
                continue
            target_idx, line_cells = traced
            neighbors_for_node.add(target_idx)
            degrees[node_idx] += 1

            edge_key = (min(node_idx, target_idx), max(node_idx, target_idx))
            if edge_key in edges:
                assert edges[edge_key] == line_cells or edges[edge_key] == list(reversed(line_cells))
            else:
                edges[edge_key] = line_cells
                covered.update(line_cells)

        assert len(neighbors_for_node) == degrees[node_idx], "a node cannot connect twice to the same neighbor"

    assert edges, "expected at least one connecting line"
    assert all(1 <= degree <= 4 for degree in degrees), "node degrees must be between 1 and 4"

    structure_cells = {
        (int(r), int(c))
        for r, c in np.argwhere(g == structure)
    }
    assert covered == structure_cells, "all foreground cells must belong to a node or a line"

    for a, b in edges:
        assert degrees[a] >= 1 and degrees[b] >= 1

    return bg, structure, nodes, cell_to_node, edges, degrees


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a grey node-and-line network by node degree, then color each line
    from the higher-degree endpoint or orange when the two endpoint degrees tie.
    """
    g = np.asarray(grid, dtype=int)
    bg, _structure, nodes, _cell_to_node, edges, degrees = _parse_graph(g)

    out = np.full_like(g, bg)
    for node_idx, (r0, c0) in enumerate(nodes):
        node_color = DEGREE_TO_COLOR[degrees[node_idx]]
        out[r0 : r0 + 3, c0 : c0 + 3] = node_color

    for (a, b), cells in edges.items():
        if degrees[a] == degrees[b]:
            color = ORANGE
        else:
            color = DEGREE_TO_COLOR[max(degrees[a], degrees[b])]
        for r, c in cells:
            out[r, c] = color

    return out


def _normalize_coords(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return [(r - min_r, c - min_c) for r, c in coords]


def _transform_coords(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    transformed = list(coords)
    if random.random() < 0.5:
        transformed = [(r, -c) for r, c in transformed]
    for _ in range(random.randrange(4)):
        transformed = [(-c, r) for r, c in transformed]
    return _normalize_coords(transformed)


def _build_graph_from_edges(
    coord_edges: list[tuple[tuple[int, int], tuple[int, int]]]
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    coord_to_idx: dict[tuple[int, int], int] = {}
    coords: list[tuple[int, int]] = []
    edges: list[tuple[int, int]] = []

    for a, b in coord_edges:
        for coord in (a, b):
            if coord not in coord_to_idx:
                coord_to_idx[coord] = len(coords)
                coords.append(coord)
        edges.append((coord_to_idx[a], coord_to_idx[b]))

    coords = _normalize_coords(coords)
    old_to_new = {old: i for i, old in enumerate(range(len(coords)))}
    return coords, [(old_to_new[a], old_to_new[b]) for a, b in edges]


def _coord_graph(coords: list[tuple[int, int]], edges: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    transformed = _transform_coords(coords)
    return transformed, edges


def _template_walk() -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    length = random.randint(2, 7)
    for _ in range(80):
        coords = [(0, 0)]
        used = {coords[0]}
        while len(coords) < length:
            r, c = coords[-1]
            options = [
                (r + dr, c + dc)
                for dr, dc in DIRS.values()
                if (r + dr, c + dc) not in used
            ]
            if not options:
                break
            nxt = random.choice(options)
            coords.append(nxt)
            used.add(nxt)
            span_r = max(rr for rr, _ in coords) - min(rr for rr, _ in coords)
            span_c = max(cc for _, cc in coords) - min(cc for _, cc in coords)
            if span_r > 4 or span_c > 4:
                break
        if len(coords) == length:
            edges = [(i, i + 1) for i in range(length - 1)]
            return _coord_graph(coords, edges)
    raise RuntimeError("failed to generate walk template")


def _template_branch() -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    for _ in range(80):
        coords, edges = _template_walk()
        degrees = [0] * len(coords)
        for a, b in edges:
            degrees[a] += 1
            degrees[b] += 1
        branch_candidates = [i for i, degree in enumerate(degrees) if degree == 2]
        if not branch_candidates:
            continue
        parent = random.choice(branch_candidates)
        pr, pc = coords[parent]
        neighbors = {coords[j] for a, b in edges for j in ((b,) if a == parent else (a,) if b == parent else ())}
        options = []
        for dr, dc in DIRS.values():
            nxt = (pr + dr, pc + dc)
            if nxt in coords:
                continue
            if nxt in neighbors:
                continue
            options.append(nxt)
        if not options:
            continue
        coords2 = list(coords) + [random.choice(options)]
        edges2 = list(edges) + [(parent, len(coords2) - 1)]
        return _coord_graph(coords2, edges2)
    raise RuntimeError("failed to generate branch template")


def _template_cycle() -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h = random.randint(1, 3)
    w = random.randint(1, 3)
    coords = [(0, 0), (0, w), (h, 0), (h, w)]
    edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
    return _coord_graph(coords, edges)


def _template_t() -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    missing = random.choice(("up", "down", "left", "right"))
    coords = [(0, 0)]
    edges: list[tuple[int, int]] = []
    coord_to_idx = {(0, 0): 0}
    for direction in ("up", "down", "left", "right"):
        if direction == missing:
            continue
        arm_len = random.randint(1, 3)
        prev = 0
        r, c = 0, 0
        dr, dc = DIRS[direction]
        for _ in range(arm_len):
            r += dr
            c += dc
            coord = (r, c)
            idx = coord_to_idx.setdefault(coord, len(coords))
            if idx == len(coords):
                coords.append(coord)
            edges.append((prev, idx))
            prev = idx
    return _coord_graph(coords, edges)


def _template_plus() -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    coords = [(0, 0)]
    edges: list[tuple[int, int]] = []
    coord_to_idx = {(0, 0): 0}
    for direction in ("up", "down", "left", "right"):
        arm_len = random.randint(1, 3)
        prev = 0
        r, c = 0, 0
        dr, dc = DIRS[direction]
        for _ in range(arm_len):
            r += dr
            c += dc
            coord = (r, c)
            idx = coord_to_idx.setdefault(coord, len(coords))
            if idx == len(coords):
                coords.append(coord)
            edges.append((prev, idx))
            prev = idx
    return _coord_graph(coords, edges)


def _sample_graph() -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    kind = random.choice(TEMPLATE_KINDS)
    if kind == "walk":
        return _template_walk()
    if kind == "branch":
        return _template_branch()
    if kind == "cycle":
        return _template_cycle()
    if kind == "t":
        return _template_t()
    return _template_plus()


def _render_graph(
    coords: list[tuple[int, int]],
    edges: list[tuple[int, int]],
    bg: int,
    row_pitch: int,
    col_pitch: int,
    top_margin: int,
    left_margin: int,
) -> tuple[np.ndarray, np.ndarray]:
    max_r = max(r for r, _ in coords)
    max_c = max(c for _, c in coords)
    height = top_margin + max_r * row_pitch + 3 + random.randint(1, 3)
    width = left_margin + max_c * col_pitch + 3 + random.randint(1, 3)
    inp = np.full((height, width), bg, dtype=int)
    out = np.full((height, width), bg, dtype=int)

    centers = []
    for r, c in coords:
        cr = top_margin + 1 + r * row_pitch
        cc = left_margin + 1 + c * col_pitch
        centers.append((cr, cc))
        inp[cr - 1 : cr + 2, cc - 1 : cc + 2] = GREY

    degrees = [0] * len(coords)
    for a, b in edges:
        degrees[a] += 1
        degrees[b] += 1

    for idx, (cr, cc) in enumerate(centers):
        out[cr - 1 : cr + 2, cc - 1 : cc + 2] = DEGREE_TO_COLOR[degrees[idx]]

    for a, b in edges:
        ra, ca = centers[a]
        rb, cb = centers[b]
        if ra == rb:
            c0, c1 = sorted((ca, cb))
            inp[ra, c0 + 2 : c1 - 1] = GREY
            color = ORANGE if degrees[a] == degrees[b] else DEGREE_TO_COLOR[max(degrees[a], degrees[b])]
            out[ra, c0 + 2 : c1 - 1] = color
        else:
            r0, r1 = sorted((ra, rb))
            inp[r0 + 2 : r1 - 1, ca] = GREY
            color = ORANGE if degrees[a] == degrees[b] else DEGREE_TO_COLOR[max(degrees[a], degrees[b])]
            out[r0 + 2 : r1 - 1, ca] = color

    return inp, out


def _coarse_edge_clear(coords: list[tuple[int, int]], edges: list[tuple[int, int]]) -> bool:
    coord_set = set(coords)
    seen_line_cells: set[tuple[int, int]] = set()
    for a, b in edges:
        ra, ca = coords[a]
        rb, cb = coords[b]
        if not (ra == rb or ca == cb):
            return False
        dr = 0 if ra == rb else (1 if rb > ra else -1)
        dc = 0 if ca == cb else (1 if cb > ca else -1)
        r, c = ra + dr, ca + dc
        path_cells: list[tuple[int, int]] = []
        while (r, c) != (rb, cb):
            if (r, c) in coord_set:
                return False
            path_cells.append((r, c))
            r += dr
            c += dc
        for cell in path_cells:
            if cell in seen_line_cells:
                return False
            seen_line_cells.add(cell)
    return True


def _gen_attempt() -> np.ndarray | None:
    try:
        coords, edges = _sample_graph()
    except RuntimeError:
        return None

    if not _coarse_edge_clear(coords, edges):
        return None

    max_r = max(r for r, _ in coords)
    max_c = max(c for _, c in coords)
    row_pitch_options = [
        pitch
        for pitch in (4, 5, 6)
        if 3 + max_r * pitch + 6 <= 30
    ]
    col_pitch_options = [
        pitch
        for pitch in (4, 5, 6)
        if 3 + max_c * pitch + 6 <= 30
    ]
    if not row_pitch_options or not col_pitch_options:
        return None

    row_pitch = random.choice(row_pitch_options)
    col_pitch = random.choice(col_pitch_options)
    top_margin = random.randint(1, 4)
    left_margin = random.randint(1, 4)
    bg = random.choice(SAFE_BACKGROUNDS)

    inp, expected = _render_graph(coords, edges, bg, row_pitch, col_pitch, top_margin, left_margin)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid node-network input")
