from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np


Coord = tuple[int, int]
Node = dict[str, object]


MAX_GRID_SIZE = 30
GENERATION_ATTEMPTS = 400

BG_COLOR = 0
LINE_COLOR = 5
NODE_COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
MARKER_COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

NODE_SIDE_CHOICES = (2, 3)
LATTICE_DIM_CHOICES = (2, 3, 4)
NODE_COUNT_MIN = 4
NODE_COUNT_MAX = 8
GAP_MIN = 3
GAP_MAX = 6
MARGIN_CHOICES = (1, 2)
EXTRA_EDGE_MAX = 2
PARTIAL_REACHABILITY_PROB = 0.5


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _dominant_color(grid: np.ndarray) -> int:
    counts = Counter(int(value) for value in np.asarray(grid).ravel())
    return int(max(counts.items(), key=lambda item: (item[1], -item[0]))[0])


def _find_monochrome_components(grid: np.ndarray, background: int) -> list[Node]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[Node] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == background or visited[r, c]:
                continue

            queue = deque([(r, c)])
            visited[r, c] = True
            coords: list[Coord] = []
            r0 = r1 = r
            c0 = c1 = c

            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                r0 = min(r0, cr)
                r1 = max(r1, cr)
                c0 = min(c0, cc)
                c1 = max(c1, cc)

                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        queue.append((nr, nc))

            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "coord_set": set(coords),
                    "bbox": (r0, c0, r1, c1),
                    "area": len(coords),
                }
            )

    return components


def _solid_square_side(component: Node) -> int | None:
    r0, c0, r1, c1 = component["bbox"]
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    if height != width or height < 2:
        return None
    if int(component["area"]) != height * width:
        return None
    return height


def _node_boundary_rays(node: Node) -> list[tuple[Coord, int, int]]:
    r0, c0, r1, c1 = node["bbox"]
    rays: list[tuple[Coord, int, int]] = []
    for r in range(r0, r1 + 1):
        rays.append(((r, c0), 0, -1))
        rays.append(((r, c1), 0, 1))
    for c in range(c0, c1 + 1):
        rays.append(((r0, c), -1, 0))
        rays.append(((r1, c), 1, 0))
    return rays


def _trace_segment(
    grid: np.ndarray,
    background: int,
    node_at: dict[Coord, int],
    start_node: int,
    start_cell: Coord,
    dr: int,
    dc: int,
) -> tuple[int, list[Coord]] | None:
    h, w = grid.shape
    r = start_cell[0] + dr
    c = start_cell[1] + dc
    path: list[Coord] = []

    while 0 <= r < h and 0 <= c < w:
        node_id = node_at.get((r, c))
        if node_id is not None:
            if node_id == start_node:
                return None
            return node_id, path
        if int(grid[r, c]) == background:
            return None
        path.append((r, c))
        r += dr
        c += dc
    return None


def _edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _extract_graph(grid: np.ndarray) -> tuple[list[Node], int, int, dict[int, list[int]]]:
    background = _dominant_color(grid)
    components = _find_monochrome_components(grid, background)

    node_candidates: list[Node] = []
    for component in components:
        side = _solid_square_side(component)
        if side is None:
            continue
        component = dict(component)
        component["side"] = side
        node_candidates.append(component)

    assert node_candidates, "expected solid square nodes"
    side_counts = Counter(int(component["side"]) for component in node_candidates)
    node_side = int(max(side_counts.items(), key=lambda item: (item[1], item[0]))[0])
    nodes = [component for component in node_candidates if int(component["side"]) == node_side]
    assert len(nodes) >= 2, "expected at least two nodes"
    assert all(int(component["side"]) == node_side for component in nodes)

    color_counts = Counter(int(component["color"]) for component in nodes)
    assert len(color_counts) == 2, "expected one regular node color and one source color"
    source_colors = [color for color, count in color_counts.items() if count == 1]
    regular_colors = [color for color, count in color_counts.items() if count == len(nodes) - 1]
    assert len(source_colors) == 1 and len(regular_colors) == 1, "expected a unique source node"
    source_color = int(source_colors[0])

    for node_id, component in enumerate(nodes):
        component["id"] = node_id
    source_nodes = [int(component["id"]) for component in nodes if int(component["color"]) == source_color]
    assert len(source_nodes) == 1
    source_id = source_nodes[0]

    node_at: dict[Coord, int] = {}
    for component in nodes:
        node_id = int(component["id"])
        for coord in component["coords"]:
            node_at[coord] = node_id

    edge_specs: dict[tuple[int, int], dict[str, object]] = {}
    for component in nodes:
        start_id = int(component["id"])
        for start_cell, dr, dc in _node_boundary_rays(component):
            outside = (start_cell[0] + dr, start_cell[1] + dc)
            if outside not in node_at and 0 <= outside[0] < grid.shape[0] and 0 <= outside[1] < grid.shape[1]:
                if int(grid[outside]) == background:
                    continue
            traced = _trace_segment(grid, background, node_at, start_id, start_cell, dr, dc)
            if traced is None:
                continue
            end_id, path = traced
            assert len(path) >= 3, "line segments must be long enough for a visible marker"
            key = _edge_key(start_id, end_id)
            if key in edge_specs:
                prev = edge_specs[key]
                same_forward = int(prev["start_id"]) == start_id and int(prev["end_id"]) == end_id and prev["path"] == path
                same_reverse = int(prev["start_id"]) == end_id and int(prev["end_id"]) == start_id and prev["path"] == path[::-1]
                assert same_forward or same_reverse, "duplicate edge traces disagree"
                continue
            edge_specs[key] = {"start_id": start_id, "end_id": end_id, "path": path}

    assert edge_specs, "expected at least one edge"

    non_node_foreground = {
        (r, c)
        for r in range(grid.shape[0])
        for c in range(grid.shape[1])
        if int(grid[r, c]) != background and (r, c) not in node_at
    }
    traced_foreground = {coord for spec in edge_specs.values() for coord in spec["path"]}
    assert non_node_foreground == traced_foreground, "every non-node foreground cell must belong to one straight edge"

    outgoing: dict[int, list[int]] = defaultdict(list)
    line_colors = set()
    for spec in edge_specs.values():
        path = spec["path"]
        colors = [int(grid[r, c]) for r, c in path]
        color_counts = Counter(colors)
        line_color, line_count = max(color_counts.items(), key=lambda item: (item[1], -item[0]))
        assert sum(1 for count in color_counts.values() if count == line_count) == 1, "line color must be unambiguous"
        marker_positions = [index for index, color in enumerate(colors) if color != line_color]
        assert len(marker_positions) == 1, "each edge must have exactly one marker"
        line_colors.add(int(line_color))

        marker_index = marker_positions[0]
        dist_to_start = marker_index + 1
        dist_to_end = len(path) - marker_index
        assert dist_to_start != dist_to_end, "marker must be strictly closer to one node"

        start_id = int(spec["start_id"])
        end_id = int(spec["end_id"])
        if dist_to_start < dist_to_end:
            tail, head = end_id, start_id
        else:
            tail, head = start_id, end_id
        outgoing[tail].append(head)

    assert len(line_colors) == 1, "all edges must share one line color"
    return nodes, source_id, source_color, outgoing


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every node reachable from the unique source node by following marker-directed graph edges.

    Input structure:
    - The background is a uniform color.
    - Several same-sized solid square nodes sit on the grid; all but one share one color, and the unique differently colored square is the source node.
    - Pairs of nodes are connected by one-pixel-wide horizontal or vertical line segments of a shared line color.
    - Each line segment contains exactly one non-background, non-line marker pixel placed closer to one endpoint node than to the other.

    Rule:
    - Identify the solid square nodes and the straight connecting segments between them.
    - On each segment, whichever endpoint is closer to the colored marker is the arrow head, so the directed edge runs from the far endpoint toward the near endpoint.
    - Starting from the unique source node, follow all outgoing directed edges recursively and recolor every reachable node to the source color.
    - Keep the background, line segments, and marker pixels unchanged.

    Key insight:
    - The marker color itself is irrelevant; only its position along each segment encodes edge direction, turning the picture into a directed graph whose reachable nodes must inherit the source color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    nodes, source_id, source_color, outgoing = _extract_graph(g)

    reachable = {source_id}
    queue = deque([source_id])
    while queue:
        node_id = queue.popleft()
        for neighbor in outgoing.get(node_id, []):
            if neighbor not in reachable:
                reachable.add(neighbor)
                queue.append(neighbor)

    out = g.copy()
    for component in nodes:
        if int(component["id"]) not in reachable:
            continue
        for r, c in component["coords"]:
            out[r, c] = source_color
    return out


def _lattice_neighbors(cell: Coord, rows: int, cols: int) -> list[Coord]:
    r, c = cell
    neighbors: list[Coord] = []
    if r > 0:
        neighbors.append((r - 1, c))
    if r + 1 < rows:
        neighbors.append((r + 1, c))
    if c > 0:
        neighbors.append((r, c - 1))
    if c + 1 < cols:
        neighbors.append((r, c + 1))
    return neighbors


def _sample_layout(rng: random.Random) -> tuple[int, int, int, list[int], list[int], int, int] | None:
    side = rng.choice(NODE_SIDE_CHOICES)
    rows = rng.choice(LATTICE_DIM_CHOICES)
    cols = rng.choice(LATTICE_DIM_CHOICES)
    margin_r = rng.choice(MARGIN_CHOICES)
    margin_c = rng.choice(MARGIN_CHOICES)

    row_gaps = [rng.randint(GAP_MIN, GAP_MAX) for _ in range(rows - 1)]
    col_gaps = [rng.randint(GAP_MIN, GAP_MAX) for _ in range(cols - 1)]
    height = 2 * margin_r + rows * side + sum(row_gaps)
    width = 2 * margin_c + cols * side + sum(col_gaps)
    if height > MAX_GRID_SIZE or width > MAX_GRID_SIZE:
        return None

    row_starts = [margin_r]
    for gap in row_gaps:
        row_starts.append(row_starts[-1] + side + gap)
    col_starts = [margin_c]
    for gap in col_gaps:
        col_starts.append(col_starts[-1] + side + gap)
    return side, rows, cols, row_starts, col_starts, height, width


def _pick_connected_cells(rows: int, cols: int, count: int, rng: random.Random) -> list[Coord]:
    start = (rng.randrange(rows), rng.randrange(cols))
    picked = {start}
    while len(picked) < count:
        frontier = {
            neighbor
            for cell in picked
            for neighbor in _lattice_neighbors(cell, rows, cols)
            if neighbor not in picked
        }
        if not frontier:
            break
        picked.add(rng.choice(list(frontier)))
    return sorted(picked)


def _adjacency_from_cells(cells: list[Coord]) -> tuple[list[tuple[Coord, Coord]], dict[Coord, set[Coord]]]:
    cell_set = set(cells)
    edges: set[tuple[Coord, Coord]] = set()
    adjacency = {cell: set() for cell in cells}
    for cell in cells:
        for neighbor in _lattice_neighbors(cell, max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1):
            if neighbor not in cell_set:
                continue
            edge = (cell, neighbor) if cell < neighbor else (neighbor, cell)
            edges.add(edge)
            adjacency[cell].add(neighbor)
            adjacency[neighbor].add(cell)
    return sorted(edges), adjacency


def _random_spanning_tree(cells: list[Coord], full_adj: dict[Coord, set[Coord]], rng: random.Random) -> list[tuple[Coord, Coord]] | None:
    root = rng.choice(cells)
    visited = {root}
    tree_edges: list[tuple[Coord, Coord]] = []
    while len(visited) < len(cells):
        frontier = [(u, v) for u in visited for v in full_adj[u] if v not in visited]
        if not frontier:
            return None
        u, v = rng.choice(frontier)
        tree_edges.append((u, v) if u < v else (v, u))
        visited.add(v)
    return tree_edges


def _edge_adjacency(cells: list[Coord], edges: list[tuple[Coord, Coord]]) -> dict[Coord, set[Coord]]:
    adjacency = {cell: set() for cell in cells}
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    return adjacency


def _grow_reachable_subset(source: Coord, adjacency: dict[Coord, set[Coord]], target_size: int, rng: random.Random) -> set[Coord]:
    reachable = {source}
    while len(reachable) < target_size:
        frontier = {neighbor for cell in reachable for neighbor in adjacency[cell] if neighbor not in reachable}
        if not frontier:
            break
        reachable.add(rng.choice(list(frontier)))
    return reachable


def _orient_edges(
    cells: list[Coord],
    edges: list[tuple[Coord, Coord]],
    source: Coord,
    reachable: set[Coord],
    adjacency: dict[Coord, set[Coord]],
    rng: random.Random,
) -> dict[tuple[Coord, Coord], tuple[Coord, Coord]]:
    oriented: dict[tuple[Coord, Coord], tuple[Coord, Coord]] = {}
    visited = {source}
    while visited != reachable:
        frontier = [
            (u, v)
            for u in visited
            for v in adjacency[u]
            if v in reachable and v not in visited
        ]
        assert frontier, "reachable subset must stay connected"
        u, v = rng.choice(frontier)
        key = (u, v) if u < v else (v, u)
        oriented[key] = (u, v)
        visited.add(v)

    for a, b in edges:
        key = (a, b) if a < b else (b, a)
        if key in oriented:
            continue
        a_reachable = a in reachable
        b_reachable = b in reachable
        if a_reachable and not b_reachable:
            oriented[key] = (b, a)
        elif b_reachable and not a_reachable:
            oriented[key] = (a, b)
        elif rng.random() < 0.5:
            oriented[key] = (a, b)
        else:
            oriented[key] = (b, a)
    assert len(oriented) == len(edges)
    return oriented


def _node_positions(cells: list[Coord], row_starts: list[int], col_starts: list[int]) -> dict[Coord, Coord]:
    return {cell: (row_starts[cell[0]], col_starts[cell[1]]) for cell in cells}


def _edge_path(cell_a: Coord, cell_b: Coord, positions: dict[Coord, Coord], side: int, rng: random.Random) -> tuple[Coord, Coord, list[Coord]]:
    top_a, left_a = positions[cell_a]
    top_b, left_b = positions[cell_b]
    if cell_a[0] == cell_b[0]:
        offset = rng.randrange(side)
        if cell_a[1] < cell_b[1]:
            start, end = cell_a, cell_b
            row = top_a + offset
            path = [(row, c) for c in range(left_a + side, left_b)]
        else:
            start, end = cell_b, cell_a
            row = top_b + offset
            path = [(row, c) for c in range(left_b + side, left_a)]
    else:
        offset = rng.randrange(side)
        if cell_a[0] < cell_b[0]:
            start, end = cell_a, cell_b
            col = left_a + offset
            path = [(r, col) for r in range(top_a + side, top_b)]
        else:
            start, end = cell_b, cell_a
            col = left_b + offset
            path = [(r, col) for r in range(top_b + side, top_a)]
    assert len(path) >= GAP_MIN
    return start, end, path


def _render_graph(
    cells: list[Coord],
    edges: list[tuple[Coord, Coord]],
    source: Coord,
    reachable: set[Coord],
    oriented: dict[tuple[Coord, Coord], tuple[Coord, Coord]],
    side: int,
    positions: dict[Coord, Coord],
    height: int,
    width: int,
    regular_color: int,
    source_color: int,
    marker_palette: list[int],
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray] | None:
    grid = np.full((height, width), BG_COLOR, dtype=int)
    output = np.full((height, width), BG_COLOR, dtype=int)

    for cell in cells:
        top, left = positions[cell]
        node_color = source_color if cell == source else regular_color
        grid[top : top + side, left : left + side] = node_color
        output[top : top + side, left : left + side] = source_color if cell in reachable else regular_color

    path_occupancy: set[Coord] = set()
    for edge in edges:
        start, end, path = _edge_path(edge[0], edge[1], positions, side, rng)
        for coord in path:
            if coord in path_occupancy or int(grid[coord]) != BG_COLOR:
                return None
        for coord in path:
            grid[coord] = LINE_COLOR
            output[coord] = LINE_COLOR
            path_occupancy.add(coord)

        tail, head = oriented[(edge[0], edge[1]) if edge[0] < edge[1] else (edge[1], edge[0])]
        max_head_distance = min(2, len(path) // 2)
        marker_distance = rng.randint(1, max_head_distance)
        if head == start:
            marker_index = marker_distance - 1
        elif head == end:
            marker_index = len(path) - marker_distance
        else:
            return None
        marker_color = rng.choice(marker_palette)
        mr, mc = path[marker_index]
        grid[mr, mc] = marker_color
        output[mr, mc] = marker_color

    return grid, output


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    layout = _sample_layout(rng)
    if layout is None:
        return None
    side, rows, cols, row_starts, col_starts, height, width = layout

    max_nodes = min(NODE_COUNT_MAX, rows * cols)
    if max_nodes < NODE_COUNT_MIN:
        return None
    node_count = rng.randint(NODE_COUNT_MIN, max_nodes)
    cells = _pick_connected_cells(rows, cols, node_count, rng)
    if len(cells) != node_count:
        return None

    full_edges, full_adj = _adjacency_from_cells(cells)
    if not full_edges:
        return None
    tree_edges = _random_spanning_tree(cells, full_adj, rng)
    if tree_edges is None:
        return None

    remaining_edges = [edge for edge in full_edges if edge not in tree_edges]
    extra_count = rng.randint(0, min(EXTRA_EDGE_MAX, len(remaining_edges)))
    extra_edges = rng.sample(remaining_edges, extra_count)
    edges = sorted(tree_edges + extra_edges)
    adjacency = _edge_adjacency(cells, edges)

    source = rng.choice(cells)
    if len(cells) > 1 and rng.random() < PARTIAL_REACHABILITY_PROB:
        target_size = rng.randint(1, len(cells) - 1)
        reachable = _grow_reachable_subset(source, adjacency, target_size, rng)
    else:
        reachable = set(cells)

    oriented = _orient_edges(cells, edges, source, reachable, adjacency, rng)
    positions = _node_positions(cells, row_starts, col_starts)

    regular_color, source_color = rng.sample(list(NODE_COLOR_POOL), 2)
    marker_palette = [color for color in MARKER_COLOR_POOL if color not in (regular_color, source_color, LINE_COLOR)]
    marker_palette = rng.sample(marker_palette, rng.randint(2, min(4, len(marker_palette))))

    rendered = _render_graph(
        cells=cells,
        edges=edges,
        source=source,
        reachable=reachable,
        oriented=oriented,
        side=side,
        positions=positions,
        height=height,
        width=width,
        regular_color=regular_color,
        source_color=source_color,
        marker_palette=marker_palette,
        rng=rng,
    )
    if rendered is None:
        return None
    grid, expected = rendered

    if _dominant_color(grid) != BG_COLOR:
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GENERATION_ATTEMPTS} attempts")
