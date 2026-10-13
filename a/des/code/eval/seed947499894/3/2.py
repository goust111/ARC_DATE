import random
from collections import Counter, defaultdict

import numpy as np


BG_COLOR = 0
WALL_COLOR = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 300
ROOM_HEIGHT_RANGE = (2, 6)
ROOM_WIDTH_RANGE = (2, 6)
_BALANCED_TOPOLOGY_CACHE = {}


def _mode(values) -> int:
    flat = [int(v) for v in np.ravel(values)]
    return int(Counter(flat).most_common(1)[0][0])


def _canonical_edge(a, b):
    return tuple(sorted((tuple(a), tuple(b))))


def _wall_line_indices(counts: np.ndarray) -> list[int]:
    counts = np.asarray(counts, dtype=int)
    low = int(counts.min())
    high = int(counts.max())
    assert high > low, "expected both wall rows/cols and interior rows/cols"
    threshold = (low + high) / 2.0
    raw = [int(i) for i, value in enumerate(counts) if int(value) >= threshold]
    assert raw, "failed to locate wall rows/cols"

    groups = []
    start = raw[0]
    prev = raw[0]
    for idx in raw[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        groups.append((start, prev))
        start = idx
        prev = idx
    groups.append((start, prev))
    return [int((a + b) // 2) for a, b in groups]


def _parse_structure(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    border = np.concatenate((g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]))
    wall = _mode(border)
    wall_mask = g == wall

    non_wall = g[~wall_mask]
    assert non_wall.size > 0, "grid must contain room interiors"
    bg = _mode(non_wall)

    wall_rows = _wall_line_indices(wall_mask.sum(axis=1))
    wall_cols = _wall_line_indices(wall_mask.sum(axis=0))
    assert wall_rows[0] == 0 and wall_rows[-1] == h - 1
    assert wall_cols[0] == 0 and wall_cols[-1] == w - 1

    room_rows = len(wall_rows) - 1
    room_cols = len(wall_cols) - 1
    assert room_rows >= 1 and room_cols >= 1

    rooms = {}
    for rr in range(room_rows):
        r0 = int(wall_rows[rr] + 1)
        r1 = int(wall_rows[rr + 1])
        assert r0 < r1, "empty room height"
        for cc in range(room_cols):
            c0 = int(wall_cols[cc] + 1)
            c1 = int(wall_cols[cc + 1])
            assert c0 < c1, "empty room width"
            patch = g[r0:r1, c0:c1]
            assert not bool((patch == wall).any()), "room interior cannot contain walls"

            seed_values = np.unique(patch[(patch != wall) & (patch != bg)])
            assert len(seed_values) == 1, "each room must contain exactly one seed color"
            rooms[(rr, cc)] = {
                "bounds": (r0, r1, c0, c1),
                "seed": int(seed_values[0]),
            }

    adjacency = defaultdict(set)
    for rr in range(room_rows):
        r0 = int(wall_rows[rr] + 1)
        r1 = int(wall_rows[rr + 1])
        for cc in range(room_cols - 1):
            shared_c = int(wall_cols[cc + 1])
            if bool(np.any(g[r0:r1, shared_c] != wall)):
                left = (rr, cc)
                right = (rr, cc + 1)
                adjacency[left].add(right)
                adjacency[right].add(left)

    for rr in range(room_rows - 1):
        shared_r = int(wall_rows[rr + 1])
        for cc in range(room_cols):
            c0 = int(wall_cols[cc] + 1)
            c1 = int(wall_cols[cc + 1])
            if bool(np.any(g[shared_r, c0:c1] != wall)):
                top = (rr, cc)
                bottom = (rr + 1, cc)
                adjacency[top].add(bottom)
                adjacency[bottom].add(top)

    for room in rooms:
        adjacency[room]

    return {
        "wall": int(wall),
        "bg": int(bg),
        "wall_rows": wall_rows,
        "wall_cols": wall_cols,
        "room_shape": (room_rows, room_cols),
        "rooms": rooms,
        "adjacency": {key: set(value) for key, value in adjacency.items()},
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each rectangular room using the direct-neighbor seed rule induced by wall gaps.

    Rooms are separated by global gray wall rows/columns. A single missing wall pixel on a
    shared wall segment creates a direct connection between two adjacent rooms. Each room
    keeps its own seed color unless it has exactly one direct neighbor, in which case the
    whole room is recolored to that neighbor's seed color.
    """
    g = np.asarray(grid, dtype=int)
    parsed = _parse_structure(g)
    out = g.copy()

    rooms = parsed["rooms"]
    adjacency = parsed["adjacency"]
    for room, info in rooms.items():
        neighbors = sorted(adjacency[room])
        if len(neighbors) == 1:
            fill_color = int(rooms[neighbors[0]]["seed"])
        else:
            fill_color = int(info["seed"])
        r0, r1, c0, c1 = info["bounds"]
        out[r0:r1, c0:c1] = fill_color

    return out


def _grid_neighbors(room_shape: tuple[int, int], node: tuple[int, int]) -> list[tuple[int, int]]:
    room_rows, room_cols = room_shape
    rr, cc = node
    result = []
    if rr > 0:
        result.append((rr - 1, cc))
    if rr + 1 < room_rows:
        result.append((rr + 1, cc))
    if cc > 0:
        result.append((rr, cc - 1))
    if cc + 1 < room_cols:
        result.append((rr, cc + 1))
    return result


def _all_nodes(room_shape: tuple[int, int]) -> list[tuple[int, int]]:
    room_rows, room_cols = room_shape
    return [(rr, cc) for rr in range(room_rows) for cc in range(room_cols)]


def _all_potential_edges(room_shape: tuple[int, int]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    room_rows, room_cols = room_shape
    edges = []
    for rr in range(room_rows):
        for cc in range(room_cols):
            if rr + 1 < room_rows:
                edges.append(((rr, cc), (rr + 1, cc)))
            if cc + 1 < room_cols:
                edges.append(((rr, cc), (rr, cc + 1)))
    return edges


def _choose_path_edges(rng: random.Random, room_shape: tuple[int, int]) -> set | None:
    nodes = _all_nodes(room_shape)
    if len(nodes) < 3:
        return None

    max_edges = min(4, len(nodes) - 1)
    min_edges = 2
    if max_edges < min_edges:
        return None

    for _ in range(100):
        target_edges = rng.randint(min_edges, max_edges)
        start = rng.choice(nodes)
        path = [start]
        seen = {start}
        while len(path) - 1 < target_edges:
            choices = [n for n in _grid_neighbors(room_shape, path[-1]) if n not in seen]
            if not choices:
                break
            nxt = rng.choice(choices)
            path.append(nxt)
            seen.add(nxt)
        if len(path) - 1 == target_edges:
            return {_canonical_edge(path[i], path[i + 1]) for i in range(len(path) - 1)}
    return None


def _choose_star_edges(rng: random.Random, room_shape: tuple[int, int]) -> set | None:
    nodes = _all_nodes(room_shape)
    centers = [node for node in nodes if len(_grid_neighbors(room_shape, node)) >= 2]
    if not centers:
        return None

    center = rng.choice(centers)
    neighbors = _grid_neighbors(room_shape, center)
    degree = rng.randint(2, min(3, len(neighbors)))
    leaves = rng.sample(neighbors, degree)
    return {_canonical_edge(center, leaf) for leaf in leaves}


def _choose_mixed_edges(rng: random.Random, room_shape: tuple[int, int]) -> set | None:
    base = _choose_star_edges(rng, room_shape)
    if base is None:
        base = _choose_path_edges(rng, room_shape)
    if base is None:
        return None

    used = set()
    for a, b in base:
        used.add(a)
        used.add(b)

    leftovers = [
        _canonical_edge(a, b)
        for a, b in _all_potential_edges(room_shape)
        if a not in used and b not in used
    ]
    if leftovers:
        base = set(base)
        base.add(rng.choice(leftovers))
    return set(base)


def _choose_topology(rng: random.Random, room_shape: tuple[int, int]) -> set:
    potential_edges = [_canonical_edge(a, b) for a, b in _all_potential_edges(room_shape)]
    assert potential_edges, "need at least one possible room adjacency"

    mode_builders = [("isolated", lambda: set())]
    mode_builders.append(("pair", lambda: {rng.choice(potential_edges)}))

    if _choose_path_edges(rng, room_shape) is not None:
        mode_builders.append(("path", lambda: _choose_path_edges(rng, room_shape)))
    if _choose_star_edges(rng, room_shape) is not None:
        mode_builders.append(("star", lambda: _choose_star_edges(rng, room_shape)))
    if len(_all_nodes(room_shape)) >= 4 and _choose_mixed_edges(rng, room_shape) is not None:
        mode_builders.append(("mixed", lambda: _choose_mixed_edges(rng, room_shape)))

    while True:
        _mode_name, builder = rng.choice(mode_builders)
        edges = builder()
        if edges is not None:
            return set(edges)


def _degree_counts(room_shape: tuple[int, int], edges: set) -> Counter:
    degrees = {node: 0 for node in _all_nodes(room_shape)}
    for a, b in edges:
        degrees[a] += 1
        degrees[b] += 1
    return Counter(degrees.values())


def _degree_balanced(room_shape: tuple[int, int], edges: set) -> bool:
    max_degree = max(len(_grid_neighbors(room_shape, node)) for node in _all_nodes(room_shape))
    counts = _degree_counts(room_shape, edges)
    buckets = [int(counts.get(degree, 0)) for degree in range(max_degree + 1)]
    return min(buckets) > 0 and max(buckets) - min(buckets) <= 1


def _choose_degree_balanced_topology(rng: random.Random, room_shape: tuple[int, int]) -> set | None:
    candidates = _BALANCED_TOPOLOGY_CACHE.get(room_shape)
    if candidates is None:
        potential_edges = [_canonical_edge(a, b) for a, b in _all_potential_edges(room_shape)]
        candidates = []
        for mask in range(1 << len(potential_edges)):
            edges = {potential_edges[idx] for idx in range(len(potential_edges)) if mask & (1 << idx)}
            if _degree_balanced(room_shape, edges):
                candidates.append(edges)
        _BALANCED_TOPOLOGY_CACHE[room_shape] = candidates
    if candidates:
        return set(rng.choice(candidates))
    return None


def _line_positions(lengths: list[int]) -> list[int]:
    positions = [0]
    cursor = 0
    for size in lengths:
        cursor += int(size) + 1
        positions.append(cursor)
    return positions


def _adjacency_edge_set(adjacency: dict) -> set:
    edges = set()
    for room, neighbors in adjacency.items():
        for other in neighbors:
            edges.add(_canonical_edge(room, other))
    return edges


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    room_shape_options = [
        (rows, cols)
        for rows in range(1, 5)
        for cols in range(1, 5)
        if 2 <= rows * cols <= len(SEED_COLORS)
        and max(len(_grid_neighbors((rows, cols), node)) for node in _all_nodes((rows, cols))) >= 3
    ]
    room_rows, room_cols = rng.choice(room_shape_options)
    if room_rows == 1 and room_cols == 1:
        return None

    room_heights = [rng.randint(*ROOM_HEIGHT_RANGE) for _ in range(room_rows)]
    room_widths = [rng.randint(*ROOM_WIDTH_RANGE) for _ in range(room_cols)]
    row_lines = _line_positions(room_heights)
    col_lines = _line_positions(room_widths)
    height = int(row_lines[-1] + 1)
    width = int(col_lines[-1] + 1)

    grid = np.full((height, width), BG_COLOR, dtype=int)
    grid[row_lines, :] = WALL_COLOR
    grid[:, col_lines] = WALL_COLOR

    topology = _choose_degree_balanced_topology(rng, (room_rows, room_cols))
    if topology is None:
        return None
    for a, b in topology:
        if a[0] == b[0]:
            rr = a[0]
            shared_c = int(col_lines[max(a[1], b[1])])
            gap_r = rng.randrange(int(row_lines[rr] + 1), int(row_lines[rr + 1]))
            grid[gap_r, shared_c] = BG_COLOR
        else:
            cc = a[1]
            shared_r = int(row_lines[max(a[0], b[0])])
            gap_c = rng.randrange(int(col_lines[cc] + 1), int(col_lines[cc + 1]))
            grid[shared_r, gap_c] = BG_COLOR

    seed_colors = rng.sample(list(SEED_COLORS), room_rows * room_cols)
    color_iter = iter(seed_colors)
    for rr in range(room_rows):
        for cc in range(room_cols):
            r0 = int(row_lines[rr] + 1)
            r1 = int(row_lines[rr + 1])
            c0 = int(col_lines[cc] + 1)
            c1 = int(col_lines[cc + 1])
            seed_r = rng.randrange(r0, r1)
            seed_c = rng.randrange(c0, c1)
            grid[seed_r, seed_c] = next(color_iter)

    try:
        parsed = _parse_structure(grid)
        if parsed["room_shape"] != (room_rows, room_cols):
            return None
        if _adjacency_edge_set(parsed["adjacency"]) != set(topology):
            return None
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-propagation input")
