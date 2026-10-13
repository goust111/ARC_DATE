from __future__ import annotations

import itertools
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
MAGENTA = 6
ORANGE = 7

SOURCE_COLORS = (GREEN, YELLOW, MAGENTA, ORANGE)
INSTRUCTION_COLORS = (RED, BLUE)

GRID_HEIGHT_RANGE = (11, 19)
GRID_WIDTH_RANGE = (11, 19)
MIN_AXIS_COUNT = 3
MAX_AXIS_COUNT = 5
MIN_AXIS_GAP = 3
MIN_TREE_NODES = 6
MAX_TREE_NODES = 10
MAX_GENERATION_ATTEMPTS = 400
Coord = tuple[int, int]
Lattice = tuple[int, int]


def _neighbors(shape: tuple[int, int], r: int, c: int) -> list[Coord]:
    out: list[Coord] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < shape[0]:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < shape[1]:
        out.append((r, c + 1))
    return out


def _degree(mask: np.ndarray, cell: Coord) -> int:
    r, c = cell
    return sum(bool(mask[nr, nc]) for nr, nc in _neighbors(mask.shape, r, c))


def _cell_neighbors(mask: np.ndarray, cell: Coord) -> list[Coord]:
    r, c = cell
    return [(nr, nc) for nr, nc in _neighbors(mask.shape, r, c) if bool(mask[nr, nc])]


def _left(direction: Coord) -> Coord:
    dr, dc = direction
    return (-dc, dr)


def _right(direction: Coord) -> Coord:
    dr, dc = direction
    return (dc, -dr)


def _opposite(direction: Coord) -> Coord:
    dr, dc = direction
    return (-dr, -dc)


def _is_endpoint(mask: np.ndarray, cell: Coord) -> bool:
    return _degree(mask, cell) == 1


def _source_endpoints(grid: np.ndarray) -> list[Coord]:
    mask = grid != BLACK
    sources: list[Coord] = []
    for r, c in np.argwhere(mask):
        cell = (int(r), int(c))
        color = int(grid[cell])
        if color not in (BLACK, GRAY, RED, BLUE) and _is_endpoint(mask, cell):
            sources.append(cell)
    return sources


def _receiver_endpoints(grid: np.ndarray) -> list[Coord]:
    mask = grid != BLACK
    receivers: list[Coord] = []
    for r, c in np.argwhere(grid == GRAY):
        cell = (int(r), int(c))
        if _is_endpoint(mask, cell):
            receivers.append(cell)
    return receivers


def _trace_signal(grid: np.ndarray, start: Coord) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    mask = g != BLACK
    assert bool(mask[start]), "signal must start on the network"
    assert _is_endpoint(mask, start), "signal source must be an endpoint"

    neighbors = _cell_neighbors(mask, start)
    assert len(neighbors) == 1
    prev = start
    current = neighbors[0]

    used_preferred_turn = False
    used_fallback = False
    stopped = False
    seen: set[tuple[Coord, Coord]] = set()

    while True:
        state = (prev, current)
        if state in seen:
            stopped = True
            return {
                "dest": None,
                "used_preferred_turn": used_preferred_turn,
                "used_fallback": used_fallback,
                "stopped": stopped,
            }
        seen.add(state)

        if _is_endpoint(mask, current):
            return {
                "dest": current,
                "used_preferred_turn": used_preferred_turn,
                "used_fallback": used_fallback,
                "stopped": stopped,
            }

        incoming = (current[0] - prev[0], current[1] - prev[1])
        nbrs = _cell_neighbors(mask, current)
        color = int(g[current])

        nxt: Coord | None = None
        if color in INSTRUCTION_COLORS:
            preferred_dir = _right(incoming) if color == RED else _left(incoming)
            straight_dir = incoming
            preferred = (current[0] + preferred_dir[0], current[1] + preferred_dir[1])
            straight = (current[0] + straight_dir[0], current[1] + straight_dir[1])
            if preferred in nbrs:
                nxt = preferred
                used_preferred_turn = True
            elif straight in nbrs:
                nxt = straight
                used_fallback = True
            else:
                stopped = True
                return {
                    "dest": None,
                    "used_preferred_turn": used_preferred_turn,
                    "used_fallback": used_fallback,
                    "stopped": stopped,
                }
        else:
            non_back = [cell for cell in nbrs if cell != prev]
            if len(non_back) == 1:
                nxt = non_back[0]
            elif len(non_back) > 1:
                straight = (current[0] + incoming[0], current[1] + incoming[1])
                if straight in non_back:
                    nxt = straight
                else:
                    stopped = True
                    return {
                        "dest": None,
                        "used_preferred_turn": used_preferred_turn,
                        "used_fallback": used_fallback,
                        "stopped": stopped,
                    }
            else:
                stopped = True
                return {
                    "dest": None,
                    "used_preferred_turn": used_preferred_turn,
                    "used_fallback": used_fallback,
                    "stopped": stopped,
                }

        prev, current = current, nxt


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate signals along a colored path network and mark the reached receiver endpoints.

    Input structure
    - The background is black.
    - A one-pixel-wide orthogonal network is drawn in gray.
    - Some junction-center pixels are recolored red or blue to encode turning rules.
    - Some degree-1 endpoints are recolored with source colors.

    Rule
    - Each colored source endpoint emits a signal into the network.
    - At a red instruction cell, the signal turns right relative to its incoming
      direction if possible; otherwise it goes straight; if neither exists it stops.
    - At a blue instruction cell, the same rule applies with a left turn.
    - Signals ignore one another and only recolor gray receiver endpoints.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    sources = _source_endpoints(g)
    for source in sources:
        source_color = int(g[source])
        trace = _trace_signal(g, source)
        dest = trace["dest"]
        if dest is None:
            continue
        dest = tuple(dest)
        if int(g[dest]) != GRAY:
            continue
        previous = int(out[dest])
        assert previous in (GRAY, source_color), "conflicting signals must not target one receiver"
        out[dest] = source_color
    return out


def _sample_axis_positions(limit: int) -> list[int]:
    feasible = [
        count
        for count in range(MIN_AXIS_COUNT, MAX_AXIS_COUNT + 1)
        if 1 + (count - 1) * MIN_AXIS_GAP <= limit - 2
    ]
    assert feasible, "grid too small for requested lattice spacing"
    count = random.choice(feasible)
    slack = (limit - 2) - (1 + (count - 1) * MIN_AXIS_GAP)
    extras = [0] * (count + 1)
    for _ in range(slack):
        extras[random.randrange(count + 1)] += 1

    vals: list[int] = []
    prefix = extras[0]
    for i in range(count):
        vals.append(1 + prefix + i * MIN_AXIS_GAP)
        prefix += extras[i + 1]
    return vals


def _lattice_neighbors(node: Lattice, rows_n: int, cols_n: int) -> list[Lattice]:
    i, j = node
    out: list[Lattice] = []
    if i > 0:
        out.append((i - 1, j))
    if i + 1 < rows_n:
        out.append((i + 1, j))
    if j > 0:
        out.append((i, j - 1))
    if j + 1 < cols_n:
        out.append((i, j + 1))
    return out


def _grow_tree(rows_n: int, cols_n: int) -> tuple[set[Lattice], list[tuple[Lattice, Lattice]]]:
    total = rows_n * cols_n
    low = min(MIN_TREE_NODES, total)
    high = min(MAX_TREE_NODES, total)
    if low > high:
        low = high
    target = random.randint(low, high)

    root = (random.randrange(rows_n), random.randrange(cols_n))
    nodes = {root}
    edges: list[tuple[Lattice, Lattice]] = []

    while len(nodes) < target:
        frontier: list[tuple[Lattice, Lattice]] = []
        for node in nodes:
            for nb in _lattice_neighbors(node, rows_n, cols_n):
                if nb not in nodes:
                    frontier.append((node, nb))
        if not frontier:
            break
        parent, child = random.choice(frontier)
        nodes.add(child)
        edges.append((parent, child))

    return nodes, edges


def _build_degree_map(nodes: set[Lattice], edges: list[tuple[Lattice, Lattice]]) -> dict[Lattice, int]:
    degree = {node: 0 for node in nodes}
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1
    return degree


def _incident_dirs(node: Lattice, edges: list[tuple[Lattice, Lattice]]) -> set[Coord]:
    out: set[Coord] = set()
    for a, b in edges:
        if a == node:
            di = b[0] - a[0]
            dj = b[1] - a[1]
        elif b == node:
            di = a[0] - b[0]
            dj = a[1] - b[1]
        else:
            continue
        if di < 0:
            out.add((-1, 0))
        elif di > 0:
            out.add((1, 0))
        elif dj < 0:
            out.add((0, -1))
        else:
            out.add((0, 1))
    return out


def _is_straight_dirs(dirs: set[Coord]) -> bool:
    return len(dirs) == 2 and any(_opposite(direction) in dirs for direction in dirs)


def _draw_network(
    height: int,
    width: int,
    rows: list[int],
    cols: list[int],
    nodes: set[Lattice],
    edges: list[tuple[Lattice, Lattice]],
) -> tuple[np.ndarray, dict[Lattice, Coord], dict[Lattice, int]]:
    grid = np.full((height, width), BLACK, dtype=int)
    coords = {node: (rows[node[0]], cols[node[1]]) for node in nodes}
    degree = _build_degree_map(nodes, edges)

    for a, b in edges:
        ra, ca = coords[a]
        rb, cb = coords[b]
        if ra == rb:
            c0, c1 = sorted((ca, cb))
            grid[ra, c0 : c1 + 1] = GRAY
        else:
            assert ca == cb
            r0, r1 = sorted((ra, rb))
            grid[r0 : r1 + 1, ca] = GRAY

    instruction_nodes: list[Lattice] = []
    for node in nodes:
        dirs = _incident_dirs(node, edges)
        if degree[node] >= 3:
            instruction_nodes.append(node)
        elif degree[node] == 2 and not _is_straight_dirs(dirs):
            # A degree-2 corner can still encode a turn or a stop.
            instruction_nodes.append(node)

    for node in instruction_nodes:
        grid[coords[node]] = random.choice(INSTRUCTION_COLORS)

    if len(instruction_nodes) >= 2:
        colors = [int(grid[coords[node]]) for node in instruction_nodes]
        if len(set(colors)) == 1:
            flip = BLUE if colors[0] == RED else RED
            chosen = random.choice(instruction_nodes)
            grid[coords[chosen]] = flip

    return grid, coords, degree


def _leaf_cells(coords: dict[Lattice, Coord], degree: dict[Lattice, int]) -> list[Coord]:
    return [coords[node] for node, deg in degree.items() if deg == 1]


def _choose_source_subset(
    base_grid: np.ndarray,
    leaves: list[Coord],
    need_fallback: bool,
    need_stop: bool,
) -> list[Coord] | None:
    traces = {leaf: _trace_signal(base_grid, leaf) for leaf in leaves}
    all_subsets: list[list[Coord]] = []
    for size in range(2, min(len(SOURCE_COLORS), len(leaves)) + 1):
        for combo in itertools.combinations(leaves, size):
            all_subsets.append(list(combo))
    random.shuffle(all_subsets)

    best: list[Coord] | None = None
    best_score: tuple[int, int, int, int] | None = None

    for subset in all_subsets:
        success = [leaf for leaf in subset if traces[leaf]["dest"] is not None]
        if len(success) < 2:
            continue

        dests = [tuple(traces[leaf]["dest"]) for leaf in success]
        if len(set(dests)) != len(dests):
            continue
        if any(dest in subset for dest in dests):
            continue

        has_preferred = any(bool(traces[leaf]["used_preferred_turn"]) for leaf in subset)
        has_fallback = any(bool(traces[leaf]["used_fallback"]) for leaf in subset)
        has_stop = any(bool(traces[leaf]["stopped"]) and traces[leaf]["dest"] is None for leaf in subset)
        if not has_preferred:
            continue
        if need_fallback and not has_fallback:
            continue
        if need_stop and not has_stop:
            continue

        score = (
            int(has_fallback),
            int(has_stop),
            len(success),
            len(subset),
        )
        if best_score is None or score > best_score:
            best = subset
            best_score = score

    return best


def _generate_candidate() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    rows = _sample_axis_positions(height)
    cols = _sample_axis_positions(width)

    nodes, edges = _grow_tree(len(rows), len(cols))
    if len(edges) < 4:
        return None

    degree_map = _build_degree_map(nodes, edges)
    if sum(deg == 1 for deg in degree_map.values()) < 4:
        return None

    grid, coords, degree = _draw_network(height, width, rows, cols, nodes, edges)
    leaves = _leaf_cells(coords, degree)
    if len(leaves) < 4:
        return None

    instruction_count = int(np.count_nonzero((grid == RED) | (grid == BLUE)))
    if instruction_count == 0:
        return None

    need_fallback = bool(random.getrandbits(1))
    need_stop = bool(random.getrandbits(1))
    chosen_sources = _choose_source_subset(grid, leaves, need_fallback, need_stop)
    if chosen_sources is None:
        return None

    colors = random.sample(list(SOURCE_COLORS), len(chosen_sources))
    inp = grid.copy()
    for source, color in zip(chosen_sources, colors):
        inp[source] = color

    out = main(inp)
    if np.array_equal(out, inp):
        return None

    source_count = len(_source_endpoints(inp))
    receiver_count = len(_receiver_endpoints(inp))
    if source_count != len(chosen_sources) or receiver_count < 1:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-routing ARC input")
