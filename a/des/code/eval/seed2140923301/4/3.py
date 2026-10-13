from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
MAGENTA = 6
ORANGE = 7
AZURE = 8

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
TRAVERSABLE = {GREEN, YELLOW, AZURE, MAGENTA, ORANGE, RED}

LOGICAL_SIDE_MIN = 5
LOGICAL_SIDE_MAX = 8
LOGICAL_NODE_COUNT_MIN = 9
LOGICAL_NODE_COUNT_MAX = 18
MAX_LOGICAL_DEGREE = 3
MAX_GATE_COUNT = 3
MAX_GENERATION_ATTEMPTS = 500

MODE_SEQUENCE = ["forward", "reverse"]
_NEXT_MODE_INDEX = random.randrange(len(MODE_SEQUENCE))


Coord = tuple[int, int]
LogicalCoord = tuple[int, int]
Edge = tuple[LogicalCoord, LogicalCoord]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _edge(a: LogicalCoord, b: LogicalCoord) -> Edge:
    return (a, b) if a <= b else (b, a)


def _logical_to_grid(node: LogicalCoord) -> Coord:
    return 1 + 2 * node[0], 1 + 2 * node[1]


def _is_traversable(value: int) -> bool:
    return int(value) in TRAVERSABLE


def _find_source(g: np.ndarray) -> Coord:
    coords = np.argwhere(g == YELLOW)
    assert coords.shape == (1, 2), "expected exactly one yellow source"
    return int(coords[0, 0]), int(coords[0, 1])


def _find_gates(g: np.ndarray) -> tuple[dict[Coord, Coord], dict[Coord, Coord]]:
    gates: dict[Coord, Coord] = {}
    red_to_orange: dict[Coord, Coord] = {}
    h, w = g.shape

    for r, c in map(tuple, np.argwhere(g == ORANGE)):
        r = int(r)
        c = int(c)
        red_neighbors = []
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if _in_bounds((h, w), nr, nc) and int(g[nr, nc]) == RED:
                red_neighbors.append((nr, nc, dr, dc))

        assert len(red_neighbors) == 1, "each orange gate center must have one red arm"
        nr, nc, dr, dc = red_neighbors[0]
        br, bc = r - dr, c - dc
        assert _in_bounds((h, w), br, bc), "gate marker must stay in bounds"
        assert int(g[br, bc]) == BLACK, "the gate's opposite arm must be black"

        center = (r, c)
        red = (int(nr), int(nc))
        gates[center] = red
        red_to_orange[red] = center

    return gates, red_to_orange


def _out_neighbors(g: np.ndarray, gates: dict[Coord, Coord], red_to_orange: dict[Coord, Coord], cell: Coord) -> list[Coord]:
    r, c = cell
    value = int(g[r, c])
    if value not in TRAVERSABLE:
        return []

    if cell in gates:
        red = gates[cell]
        return [red] if _is_traversable(int(g[red])) else []

    h, w = g.shape
    outs = []
    blocked_orange = red_to_orange.get(cell)
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if not _in_bounds((h, w), nr, nc):
            continue
        if not _is_traversable(int(g[nr, nc])):
            continue
        if blocked_orange is not None and (nr, nc) == blocked_orange:
            continue
        outs.append((nr, nc))
    return outs


def _reachable_cells(g: np.ndarray) -> set[Coord]:
    gates, red_to_orange = _find_gates(g)
    source = _find_source(g)
    seen = {source}
    queue = deque([source])

    while queue:
        cell = queue.popleft()
        for nxt in _out_neighbors(g, gates, red_to_orange, cell):
            if nxt in seen:
                continue
            seen.add(nxt)
            queue.append(nxt)
    return seen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate a signal along the path network, respecting one-way gates, and light reachable terminals.

    Input structure:
    - black background.
    - green one-pixel-wide paths with corners and branch points.
    - gate markers consist of an orange center with one adjacent red pixel and the opposite arm black.
    - one yellow source pixel.
    - dead-end terminals are azure.

    Rule:
    - propagate from the yellow source through the path network.
    - a gate can be entered only from non-red sides and can leave only through its red side.
    - any reachable azure terminal becomes magenta.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    reachable = _reachable_cells(g)
    out = g.copy()
    for r, c in reachable:
        if int(out[r, c]) == AZURE:
            out[r, c] = MAGENTA
    return out


def _logical_neighbors(node: LogicalCoord, h: int, w: int) -> list[LogicalCoord]:
    r, c = node
    out = []
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _incident_dirs(node: LogicalCoord, edges: set[Edge], h: int, w: int) -> list[tuple[int, int]]:
    dirs = []
    for dr, dc in DIRS4:
        nbr = (node[0] + dr, node[1] + dc)
        if 0 <= nbr[0] < h and 0 <= nbr[1] < w and _edge(node, nbr) in edges:
            dirs.append((dr, dc))
    return dirs


def _build_logical_tree(log_h: int, log_w: int, target_nodes: int) -> tuple[set[LogicalCoord], set[Edge]] | None:
    start = (random.randrange(log_h), random.randrange(log_w))
    nodes = {start}
    edges: set[Edge] = set()
    degree = {start: 0}

    while len(nodes) < target_nodes:
        parents: list[tuple[LogicalCoord, list[LogicalCoord]]] = []
        weights: list[int] = []
        for node in nodes:
            if degree[node] >= MAX_LOGICAL_DEGREE:
                continue
            free = [nbr for nbr in _logical_neighbors(node, log_h, log_w) if nbr not in nodes]
            if not free:
                continue
            parents.append((node, free))
            deg = degree[node]
            branch_bias = 2 if deg == 0 else 4 if deg == 1 else 2
            weights.append(branch_bias * len(free))

        if not parents:
            return None

        parent, free_neighbors = random.choices(parents, weights=weights, k=1)[0]
        child = random.choice(free_neighbors)
        nodes.add(child)
        edges.add(_edge(parent, child))
        degree[parent] += 1
        degree[child] = 1

    return nodes, edges


def _gate_candidates(nodes: set[LogicalCoord], edges: set[Edge], log_h: int, log_w: int) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for node in nodes:
        dirs = set(_incident_dirs(node, edges, log_h, log_w))
        if len(dirs) != 3:
            continue

        if (0, -1) in dirs and (0, 1) in dirs:
            axis = [(0, -1), (0, 1)]
        elif (-1, 0) in dirs and (1, 0) in dirs:
            axis = [(-1, 0), (1, 0)]
        else:
            continue

        stem = next(direction for direction in dirs if direction not in set(axis))
        candidates.append({"center": node, "axis": axis, "stem": stem})
    return candidates


def _choose_gates(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    random.shuffle(candidates)
    target = random.randint(1, min(MAX_GATE_COUNT, len(candidates)))
    chosen = []
    for candidate in candidates[:target]:
        axis = list(candidate["axis"])
        red_dir = random.choice(axis)
        blocked_dir = axis[0] if axis[1] == red_dir else axis[1]
        chosen.append(
            {
                "center": candidate["center"],
                "stem": candidate["stem"],
                "red_dir": red_dir,
                "blocked_dir": blocked_dir,
            }
        )
    return chosen


def _render_base(nodes: set[LogicalCoord], edges: set[Edge], gates: list[dict[str, object]], log_h: int, log_w: int) -> np.ndarray:
    grid = np.full((2 * log_h + 1, 2 * log_w + 1), BLACK, dtype=int)

    for node in nodes:
        r, c = _logical_to_grid(node)
        grid[r, c] = GREEN

    for a, b in edges:
        ar, ac = _logical_to_grid(a)
        br, bc = _logical_to_grid(b)
        grid[(ar + br) // 2, (ac + bc) // 2] = GREEN

    for gate in gates:
        center = gate["center"]
        red_dir = gate["red_dir"]
        blocked_dir = gate["blocked_dir"]
        cr, cc = _logical_to_grid(center)
        grid[cr, cc] = ORANGE
        grid[cr + red_dir[0], cc + red_dir[1]] = RED
        grid[cr + blocked_dir[0], cc + blocked_dir[1]] = BLACK

    return grid


def _undirected_neighbors(g: np.ndarray, cell: Coord) -> list[Coord]:
    r, c = cell
    if not _is_traversable(int(g[r, c])):
        return []

    h, w = g.shape
    out = []
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if _in_bounds((h, w), nr, nc) and _is_traversable(int(g[nr, nc])):
            out.append((nr, nc))
    return out


def _leaf_cells(g: np.ndarray) -> list[Coord]:
    leaves = []
    for r, c in np.argwhere(np.isin(g, list(TRAVERSABLE))):
        cell = (int(r), int(c))
        if len(_undirected_neighbors(g, cell)) == 1:
            leaves.append(cell)
    return leaves


def _paint_source_and_terminals(base: np.ndarray, source: Coord) -> np.ndarray:
    inp = base.copy()
    leaves = _leaf_cells(base)
    assert source in leaves, "source must be a leaf of the visible network"
    for r, c in leaves:
        inp[r, c] = YELLOW if (r, c) == source else AZURE
    return inp


def _instance_stats(inp: np.ndarray) -> dict[str, int]:
    gates, _red_to_orange = _find_gates(inp)
    reachable = _reachable_cells(inp)
    total_terminals = int(np.count_nonzero(inp == AZURE))
    reached_terminals = sum((r, c) in reachable for r, c in map(tuple, np.argwhere(inp == AZURE)))
    forward_gates = sum(center in reachable for center in gates)
    reverse_gates = sum(red in reachable and center not in reachable for center, red in gates.items())
    return {
        "total_terminals": total_terminals,
        "reached_terminals": int(reached_terminals),
        "forward_gates": int(forward_gates),
        "reverse_gates": int(reverse_gates),
    }


def _valid_instance(inp: np.ndarray, mode: str) -> bool:
    try:
        out = main(inp)
        stats = _instance_stats(inp)
    except AssertionError:
        return False

    if not np.array_equal(out, main(inp.copy())):
        return False

    total_terminals = stats["total_terminals"]
    reached_terminals = stats["reached_terminals"]
    if total_terminals < 2:
        return False
    if reached_terminals <= 0 or reached_terminals >= total_terminals:
        return False
    if mode == "forward" and stats["forward_gates"] <= 0:
        return False
    if mode == "reverse" and stats["reverse_gates"] <= 0:
        return False
    return True


def _pick_source(base: np.ndarray, mode: str) -> np.ndarray | None:
    leaves = _leaf_cells(base)
    random.shuffle(leaves)
    for source in leaves:
        inp = _paint_source_and_terminals(base, source)
        if _valid_instance(inp, mode):
            return inp
    return None


def _gen_attempt(mode: str) -> np.ndarray | None:
    log_h = random.randint(LOGICAL_SIDE_MIN, LOGICAL_SIDE_MAX)
    log_w = random.randint(LOGICAL_SIDE_MIN, LOGICAL_SIDE_MAX)
    target_nodes = random.randint(
        LOGICAL_NODE_COUNT_MIN,
        min(LOGICAL_NODE_COUNT_MAX, log_h * log_w),
    )
    built = _build_logical_tree(log_h, log_w, target_nodes)
    if built is None:
        return None
    nodes, edges = built

    candidates = _gate_candidates(nodes, edges, log_h, log_w)
    if not candidates:
        return None
    gates = _choose_gates(candidates)
    if not gates:
        return None

    base = _render_base(nodes, edges, gates, log_h, log_w)
    leaves = _leaf_cells(base)
    if len(leaves) < 3:
        return None

    return _pick_source(base, mode)


def generate_input() -> np.ndarray:
    global _NEXT_MODE_INDEX
    mode = MODE_SEQUENCE[_NEXT_MODE_INDEX]
    _NEXT_MODE_INDEX = (_NEXT_MODE_INDEX + 1) % len(MODE_SEQUENCE)
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(mode)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gate-routing puzzle")
