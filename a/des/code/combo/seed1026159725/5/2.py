import numpy as np


PATH_COLOR = 5
SOLUTION_COLOR = 3
JUNCTION_COLOR = 4
RESERVED_COLORS = {PATH_COLOR, SOLUTION_COLOR, JUNCTION_COLOR}
MAX_GENERATION_ATTEMPTS = 400


def _neighbors(coord: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = coord
    h, w = shape
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _analyze_input(grid: np.ndarray) -> tuple[int, int, list[tuple[int, int]], dict[tuple[int, int], list[tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    colors, counts = np.unique(g[g != bg], return_counts=True)
    assert len(colors) == 3, "expected one path color and two endpoint colors"

    count_by_color = {int(color): int(count) for color, count in zip(colors, counts)}
    endpoint_colors = [color for color, count in count_by_color.items() if count == 1]
    assert len(endpoint_colors) == 2, "expected exactly two singleton endpoint colors"
    path_colors = [color for color in count_by_color if color not in endpoint_colors]
    assert len(path_colors) == 1, "expected exactly one path color"
    path_color = path_colors[0]

    endpoints = [tuple(map(int, np.argwhere(g == color)[0])) for color in endpoint_colors]
    node_set = {tuple(map(int, rc)) for rc in np.argwhere((g == path_color) | np.isin(g, endpoint_colors))}
    assert len(node_set) == count_by_color[path_color] + 2

    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for node in node_set:
        adjacency[node] = [nbr for nbr in _neighbors(node, g.shape) if nbr in node_set]

    for endpoint in endpoints:
        assert len(adjacency[endpoint]) == 1, "each endpoint must touch the network at one side"

    seen = set()
    stack = [endpoints[0]]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(adjacency[node])
    assert endpoints[1] in seen and len(seen) == len(node_set), "network must be connected"

    edge_count = sum(len(nbrs) for nbrs in adjacency.values()) // 2
    assert edge_count == len(node_set) - 1, "network must be a tree to guarantee a unique path"
    return bg, path_color, endpoints, adjacency


def _path_between(
    start: tuple[int, int],
    goal: tuple[int, int],
    adjacency: dict[tuple[int, int], list[tuple[int, int]]],
) -> list[tuple[int, int]]:
    parent = {start: None}
    queue = [start]
    head = 0
    while head < len(queue):
        node = queue[head]
        head += 1
        if node == goal:
            break
        for nbr in adjacency[node]:
            if nbr not in parent:
                parent[nbr] = node
                queue.append(nbr)
    assert goal in parent, "endpoints must be connected"

    path: list[tuple[int, int]] = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the unique route between the two colored endpoints on a thin orthogonal
    network, recolor the route green, and mark route junctions yellow.
    """
    g = np.asarray(grid, dtype=int)
    _bg, path_color, endpoints, adjacency = _analyze_input(g)
    route = _path_between(endpoints[0], endpoints[1], adjacency)

    out = g.copy()
    endpoint_set = set(endpoints)
    for cell in route:
        if cell in endpoint_set:
            continue
        if g[cell] != path_color:
            continue
        out[cell] = SOLUTION_COLOR

    for cell in route:
        if cell in endpoint_set:
            continue
        if g[cell] != path_color:
            continue
        path_neighbors = sum(1 for nbr in _neighbors(cell, g.shape) if g[nbr] == path_color)
        if path_neighbors > 2:
            out[cell] = JUNCTION_COLOR

    return out


def _coarse_neighbors(node: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = node
    h, w = shape
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _build_main_path(
    rng: np.random.Generator,
    coarse_shape: tuple[int, int],
    length: int,
) -> list[tuple[int, int]] | None:
    for _ in range(200):
        start = (int(rng.integers(coarse_shape[0])), int(rng.integers(coarse_shape[1])))
        path = [start]
        used = {start}
        while len(path) < length:
            current = path[-1]
            options = [nbr for nbr in _coarse_neighbors(current, coarse_shape) if nbr not in used]
            if not options:
                break
            if len(path) >= 2:
                prev = path[-2]
                dr = current[0] - prev[0]
                dc = current[1] - prev[1]
                straight = (current[0] + dr, current[1] + dc)
                if straight in options and rng.random() < 0.55:
                    nxt = straight
                else:
                    nxt = options[int(rng.integers(len(options)))]
            else:
                nxt = options[int(rng.integers(len(options)))]
            path.append(nxt)
            used.add(nxt)
        if len(path) == length:
            return path
    return None


def _add_branch(
    rng: np.random.Generator,
    coarse_shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    anchors: list[tuple[int, int]],
    length_range: tuple[int, int],
) -> list[tuple[int, int]] | None:
    if not anchors:
        return None
    for _ in range(120):
        anchor = anchors[int(rng.integers(len(anchors)))]
        starts = [nbr for nbr in _coarse_neighbors(anchor, coarse_shape) if nbr not in occupied]
        if not starts:
            continue
        target_len = int(rng.integers(length_range[0], length_range[1] + 1))
        branch = [starts[int(rng.integers(len(starts)))]]
        used = set(branch)
        while len(branch) < target_len:
            current = branch[-1]
            options = [
                nbr
                for nbr in _coarse_neighbors(current, coarse_shape)
                if nbr not in occupied and nbr not in used
            ]
            if not options:
                break
            branch.append(options[int(rng.integers(len(options)))])
            used.add(branch[-1])
        if branch:
            return [anchor] + branch
    return None


def _edges_from_walk(walk: list[tuple[int, int]]) -> set[tuple[tuple[int, int], tuple[int, int]]]:
    edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for a, b in zip(walk, walk[1:]):
        edges.add(tuple(sorted((a, b))))
    return edges


def _choose_endpoint_colors(rng: np.random.Generator, bg: int) -> tuple[int, int]:
    options = [color for color in range(10) if color not in RESERVED_COLORS | {bg}]
    rng.shuffle(options)
    return int(options[0]), int(options[1])


def _render_tree(
    nodes: set[tuple[int, int]],
    edges: set[tuple[tuple[int, int], tuple[int, int]]],
    endpoints: tuple[tuple[int, int], tuple[int, int]],
    endpoint_colors: tuple[int, int],
    bg: int,
) -> np.ndarray:
    rows = [r for r, _ in nodes]
    cols = [c for _, c in nodes]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    cropped_nodes = {(r - min_r, c - min_c) for r, c in nodes}
    cropped_edges = {
        tuple(sorted(((a[0] - min_r, a[1] - min_c), (b[0] - min_r, b[1] - min_c))))
        for a, b in edges
    }
    ep0 = (endpoints[0][0] - min_r, endpoints[0][1] - min_c)
    ep1 = (endpoints[1][0] - min_r, endpoints[1][1] - min_c)

    coarse_h = max(r for r, _ in cropped_nodes) + 1
    coarse_w = max(c for _, c in cropped_nodes) + 1
    footprint_h = 2 * coarse_h - 1
    footprint_w = 2 * coarse_w - 1

    rng = np.random.default_rng()
    top = int(rng.integers(1, 4))
    bottom = int(rng.integers(1, 4))
    left = int(rng.integers(1, 4))
    right = int(rng.integers(1, 4))
    grid = np.full((footprint_h + top + bottom, footprint_w + left + right), bg, dtype=int)

    def to_grid(node: tuple[int, int]) -> tuple[int, int]:
        return top + 2 * node[0], left + 2 * node[1]

    for node in cropped_nodes:
        rr, cc = to_grid(node)
        grid[rr, cc] = PATH_COLOR
    for a, b in cropped_edges:
        ra, ca = to_grid(a)
        rb, cb = to_grid(b)
        grid[(ra + rb) // 2, (ca + cb) // 2] = PATH_COLOR

    r0, c0 = to_grid(ep0)
    r1, c1 = to_grid(ep1)
    grid[r0, c0] = endpoint_colors[0]
    grid[r1, c1] = endpoint_colors[1]
    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    coarse_h = int(rng.integers(4, 9))
    coarse_w = int(rng.integers(4, 10))
    coarse_shape = (coarse_h, coarse_w)
    max_main = min(coarse_h * coarse_w - 2, 11)
    main_len = int(rng.integers(4, max_main + 1))

    main_path = _build_main_path(rng, coarse_shape, main_len)
    if main_path is None:
        return None

    nodes = set(main_path)
    edges = _edges_from_walk(main_path)
    branch_mode = int(rng.integers(0, 3))
    branch_count = [0, int(rng.integers(1, 3)), int(rng.integers(2, 5))][branch_mode]
    internal_nodes = main_path[1:-1]

    for _ in range(branch_count):
        branch = _add_branch(rng, coarse_shape, nodes, internal_nodes, (1, 4))
        if branch is None:
            continue
        branch_nodes = branch[1:]
        branch_edges = _edges_from_walk(branch)
        nodes.update(branch_nodes)
        edges.update(branch_edges)
        if len(branch_nodes) >= 2 and rng.random() < 0.35:
            internal_nodes.extend(branch_nodes[:-1])

    degrees = {node: 0 for node in nodes}
    for a, b in edges:
        degrees[a] += 1
        degrees[b] += 1

    endpoint_a = main_path[0]
    endpoint_b = main_path[-1]
    if degrees[endpoint_a] != 1 or degrees[endpoint_b] != 1:
        return None
    if any(degrees[node] == 0 for node in nodes):
        return None

    bg_options = [color for color in range(10) if color not in RESERVED_COLORS]
    bg = int(bg_options[int(rng.integers(len(bg_options)))])
    endpoint_colors = _choose_endpoint_colors(rng, bg)
    grid = _render_tree(nodes, edges, (endpoint_a, endpoint_b), endpoint_colors, bg)

    try:
        _bg, _path_color, endpoints, adjacency = _analyze_input(grid)
        route = _path_between(endpoints[0], endpoints[1], adjacency)
    except AssertionError:
        return None

    route_core = [cell for cell in route if grid[cell] == PATH_COLOR]
    junction_count = sum(
        1
        for cell in route_core
        if sum(1 for nbr in _neighbors(cell, grid.shape) if grid[nbr] == PATH_COLOR) > 2
    )

    if branch_count == 0 and junction_count != 0:
        return None
    if branch_count > 0 and len(route_core) < 3:
        return None
    if branch_mode == 2 and junction_count == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
