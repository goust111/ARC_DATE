import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8
MAROON = 9

MARKER_COLOR = BLUE
PATH_COLORS = (MAGENTA, ORANGE, CYAN)
ARRIVAL_COLORS = (RED, BLUE, YELLOW)

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
MARKER_TO_DIR = {
    (0, 0): (-1, 0),
    (0, 1): (-1, 0),
    (0, 2): (0, 1),
    (1, 0): (0, -1),
    (1, 2): (0, 1),
    (2, 0): (0, -1),
    (2, 1): (1, 0),
    (2, 2): (1, 0),
}
DIR_TO_MARKERS = {
    (-1, 0): ((0, 0), (0, 1)),
    (0, 1): ((0, 2), (1, 2)),
    (1, 0): ((2, 1), (2, 2)),
    (0, -1): ((1, 0), (2, 0)),
}

MAX_GENERATION_ATTEMPTS = 300


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _is_border(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _delta(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    if dr != 0 and dc != 0:
        raise AssertionError("vertices must be orthogonally aligned")
    if dr > 0:
        return (1, 0)
    if dr < 0:
        return (-1, 0)
    if dc > 0:
        return (0, 1)
    if dc < 0:
        return (0, -1)
    raise AssertionError("distinct vertices required")


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _find_nodes(grid: np.ndarray) -> dict[tuple[int, int], tuple[int, int]]:
    h, w = grid.shape
    nodes: dict[tuple[int, int], tuple[int, int]] = {}
    used: set[tuple[int, int]] = set()
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            patch = grid[r - 1 : r + 2, c - 1 : c + 2]
            if int(np.count_nonzero(patch == GREY)) != 8:
                continue
            marker_cells = np.argwhere(patch != GREY)
            if len(marker_cells) != 1:
                continue
            mr, mc = map(int, marker_cells[0])
            if (mr, mc) == (1, 1):
                continue
            coords = {(rr, cc) for rr in range(r - 1, r + 2) for cc in range(c - 1, c + 2)}
            if coords & used:
                continue
            nodes[(r, c)] = MARKER_TO_DIR[(mr, mc)]
            used |= coords
    return nodes


def _ray_target(
    grid: np.ndarray,
    start: tuple[int, int],
    step: tuple[int, int],
    vertices: set[tuple[int, int]],
) -> tuple[int, int] | None:
    h, w = grid.shape
    r, c = start
    dr, dc = step
    r += dr
    c += dc
    while 0 <= r < h and 0 <= c < w:
        if int(grid[r, c]) == BG:
            return None
        if (r, c) in vertices and (r, c) != start:
            return (r, c)
        r += dr
        c += dc
    return None


def _adjacency(
    grid: np.ndarray,
    vertices: set[tuple[int, int]],
) -> dict[tuple[int, int], dict[tuple[int, int], tuple[int, int]]]:
    adj: dict[tuple[int, int], dict[tuple[int, int], tuple[int, int]]] = {}
    for vertex in vertices:
        rays: dict[tuple[int, int], tuple[int, int]] = {}
        for step in DIRS:
            target = _ray_target(grid, vertex, step, vertices)
            if target is not None:
                rays[step] = target
        adj[vertex] = rays
    return adj


def _follow_signal(
    source: tuple[int, int],
    node_dirs: dict[tuple[int, int], tuple[int, int]],
    terminals: set[tuple[int, int]],
    adj: dict[tuple[int, int], dict[tuple[int, int], tuple[int, int]]],
) -> tuple[int, tuple[int, int]]:
    time = 0
    current = source
    seen = {source}
    while True:
        if current in terminals:
            return time, current
        if current in node_dirs:
            exit_dir = node_dirs[current]
            nxt = adj[current].get(exit_dir)
            assert nxt is not None, "node exit direction must connect to a path"
        else:
            choices = list(adj[current].values())
            assert len(choices) == 1, "sources and terminals must be simple endpoints"
            nxt = choices[0]
        time += _manhattan(current, nxt)
        current = nxt
        if current in seen:
            raise AssertionError("signal routing must not create a cycle")
        seen.add(current)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow each edge source through 3x3 direction nodes and recolor reachable terminals by arrival order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    node_dirs = _find_nodes(g)
    assert node_dirs, "expected at least one 3x3 grey node"

    sources = {
        tuple(map(int, coord))
        for coord in np.argwhere(g == GREEN)
        if _is_border(g.shape, int(coord[0]), int(coord[1]))
    }
    terminals = {tuple(map(int, coord)) for coord in np.argwhere(g == MAROON)}
    assert sources, "expected at least one green edge source"
    assert terminals, "expected at least one maroon terminal"

    vertices = set(node_dirs) | sources | terminals
    adj = _adjacency(g, vertices)

    earliest: dict[tuple[int, int], int] = {}
    for source in sorted(sources):
        time, terminal = _follow_signal(source, node_dirs, terminals, adj)
        prev = earliest.get(terminal)
        if prev is None or time < prev:
            earliest[terminal] = time

    ordered = sorted(earliest.items(), key=lambda item: (item[1], item[0][0], item[0][1]))
    assert len(ordered) <= len(ARRIVAL_COLORS), "generator uses at most three reachable terminals"

    out = g.copy()
    for idx, (terminal, _time) in enumerate(ordered):
        out[terminal] = ARRIVAL_COLORS[idx]
    return out


def _sample_positions(
    rng: random.Random,
    count: int,
    lo: int,
    hi: int,
    min_gap: int,
) -> list[int] | None:
    if count <= 0:
        return []
    if lo + (count - 1) * min_gap > hi:
        return None
    positions: list[int] = []
    next_lo = lo
    for idx in range(count):
        remaining = count - idx - 1
        max_pos = hi - remaining * min_gap
        if next_lo > max_pos:
            return None
        pos = rng.randint(next_lo, max_pos)
        positions.append(pos)
        next_lo = pos + min_gap
    return positions


def _paint_segment(
    grid: np.ndarray,
    a: tuple[int, int],
    b: tuple[int, int],
    color: int,
) -> None:
    dr, dc = _delta(a, b)
    steps = _manhattan(a, b)
    for t in range(1, steps):
        r = a[0] + dr * t
        c = a[1] + dc * t
        grid[r, c] = color


def _transform_grid(grid: np.ndarray, name: str) -> np.ndarray:
    if name == "identity":
        return grid.copy()
    if name == "flip_ud":
        return np.flipud(grid).copy()
    if name == "flip_lr":
        return np.fliplr(grid).copy()
    if name == "rot180":
        return np.rot90(grid, 2).copy()
    if name == "rot90":
        return np.rot90(grid, 1).copy()
    if name == "rot270":
        return np.rot90(grid, 3).copy()
    raise ValueError(f"unknown transform {name}")


def _choose_marker_offsets(
    rng: random.Random,
    exit_dirs: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = [(-1, -1)] * len(exit_dirs)
    order = list(range(len(exit_dirs)))
    rng.shuffle(order)

    corner_idx = order[0]
    edge_idx = order[1 if len(order) > 1 else 0]

    for idx, exit_dir in enumerate(exit_dirs):
        corner_offset, edge_offset = DIR_TO_MARKERS[exit_dir]
        if idx == corner_idx and idx == edge_idx:
            offsets[idx] = corner_offset if rng.random() < 0.5 else edge_offset
        elif idx == corner_idx:
            offsets[idx] = corner_offset
        elif idx == edge_idx:
            offsets[idx] = edge_offset
        else:
            offsets[idx] = corner_offset if rng.random() < 0.5 else edge_offset
    return offsets


def _route_nodes(
    start_row: int,
    low_row: int,
    high_row: int,
    columns: list[int],
    vertical_moves: int,
    tail_extra: bool,
) -> list[tuple[int, int]]:
    nodes = [(start_row, columns[0])]
    current_row = start_row
    col_index = 1
    for _ in range(vertical_moves):
        current_row = high_row if current_row == low_row else low_row
        nodes.append((current_row, columns[col_index - 1]))
        nodes.append((current_row, columns[col_index]))
        col_index += 1
    if tail_extra:
        nodes.append((current_row, columns[col_index]))
    return nodes


def _route_vertical_counts(rng: random.Random, n_routes: int, tie_mode: bool) -> list[int]:
    choices = [0, 1, 2, 3]
    if n_routes == 2:
        if tie_mode:
            repeated = rng.choice(choices)
            return [repeated, repeated]
        return rng.sample(choices, 2)

    assert n_routes == 3
    if tie_mode:
        repeated = rng.choice(choices)
        other_choices = [value for value in choices if value != repeated]
        values = [repeated, repeated, rng.choice(other_choices)]
        rng.shuffle(values)
        return values
    return rng.sample(choices, 3)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_routes = rng.choice([2, 3])
    tie_mode = rng.random() < 0.5

    band_height = 7
    band_gap = 2
    top_margin = 2
    bottom_margin = 2
    min_height = top_margin + bottom_margin + n_routes * band_height + (n_routes - 1) * band_gap
    height = min_height + rng.randint(0, 5)
    width = rng.randint(23, 31)

    top_shift = rng.randint(0, height - min_height)
    band_tops = [top_margin + top_shift + idx * (band_height + band_gap) for idx in range(n_routes)]

    route_colors = list(PATH_COLORS)
    rng.shuffle(route_colors)
    vertical_counts = _route_vertical_counts(rng, n_routes, tie_mode)

    grid = np.full((height, width), BG, dtype=int)
    all_nodes: list[tuple[int, int]] = []
    all_exit_dirs: list[tuple[int, int]] = []
    sources: list[tuple[int, int]] = []
    terminals: list[tuple[int, int]] = []

    for idx in range(n_routes):
        low_row = band_tops[idx] + 1
        high_row = band_tops[idx] + 5
        start_row = rng.choice([low_row, high_row])
        vertical_moves = vertical_counts[idx]
        tail_extra = rng.random() < 0.5
        column_count = vertical_moves + 1 + int(tail_extra)
        columns = _sample_positions(rng, column_count, 3, width - 4, 4)
        if columns is None:
            return None

        nodes = _route_nodes(start_row, low_row, high_row, columns, vertical_moves, tail_extra)
        source = (start_row, 0)
        terminal = (nodes[-1][0], width - 1)
        route_vertices = [source] + nodes + [terminal]

        for a, b in zip(route_vertices, route_vertices[1:]):
            _paint_segment(grid, a, b, route_colors[idx])

        all_nodes.extend(nodes)
        sources.append(source)
        terminals.append(terminal)
        for node, nxt in zip(nodes, route_vertices[2:]):
            all_exit_dirs.append(_delta(node, nxt))

    marker_offsets = _choose_marker_offsets(rng, all_exit_dirs)

    for (r, c), offset in zip(all_nodes, marker_offsets):
        if not _in_bounds(grid.shape, r - 1, c - 1) or not _in_bounds(grid.shape, r + 1, c + 1):
            return None
        grid[r - 1 : r + 2, c - 1 : c + 2] = GREY
        grid[r - 1 + offset[0], c - 1 + offset[1]] = MARKER_COLOR

    for source in sources:
        grid[source] = GREEN
    for terminal in terminals:
        grid[terminal] = MAROON

    transforms = ["identity", "flip_ud", "flip_lr", "rot180"]
    if not tie_mode:
        transforms += ["rot90", "rot270"]
    transformed = _transform_grid(grid, rng.choice(transforms))

    try:
        out = main(transformed)
    except AssertionError:
        return None

    changed = np.argwhere((transformed == MAROON) & (out != MAROON))
    if len(changed) != n_routes:
        return None

    return transformed


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exceeded retry budget")
