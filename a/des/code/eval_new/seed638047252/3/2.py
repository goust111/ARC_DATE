from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]

MAX_PATHS_PER_COLOR = 128

COARSE_ROWS_RANGE = (5, 9)
COARSE_COLS_RANGE = (5, 9)
PAIR_COUNT_SMALL = (2, 3)
PAIR_COUNT_LARGE = (2, 3, 4)
ROUTE_EDGE_MIN = 3
ROUTE_EDGE_MAX = 6
GENERATION_RETRIES = 500
GRAPH_BUILD_RETRIES = 120
ROUTE_BUILD_RETRIES = 120
CONNECT_RETRIES = 160
LOOP_PROBABILITY = 0.5


Coord = tuple[int, int]
CoarseNode = tuple[int, int]
CandidatePath = tuple[tuple[Coord, ...], frozenset[Coord]]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color endpoint pair through the grey corridor network and tint the used corridor cells with that color.

    Input structure:
    - Background is black.
    - Grey cells form a one-pixel-wide 4-connected corridor network with branches and occasional unused side corridors.
    - Every non-black, non-grey cell is a colored endpoint; each such color appears exactly twice and each endpoint sits at a corridor end.

    Rule:
    - Pair the two endpoints of each color.
    - For every pair, choose a simple 4-connected route whose interior visits only grey corridor cells.
    - The chosen routes must be mutually non-overlapping on grey cells.
    - Recolor every grey cell used by a chosen route to that pair's color, while leaving all unused grey cells unchanged.

    Key insight:
    - A single pair can have several local route options, but only one global assignment of non-overlapping corridor paths lets all same-color endpoints connect simultaneously.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"

    solutions = _solve_assignments(g, max_solutions=2)
    assert len(solutions) == 1, "expected a unique non-overlapping path assignment"

    out = g.copy()
    for color, (_cells, grey_cells) in solutions[0].items():
        for r, c in grey_cells:
            out[r, c] = color
    return out


def _neighbors4(cell: Coord, height: int, width: int) -> list[Coord]:
    r, c = cell
    result = []
    if r > 0:
        result.append((r - 1, c))
    if r + 1 < height:
        result.append((r + 1, c))
    if c > 0:
        result.append((r, c - 1))
    if c + 1 < width:
        result.append((r, c + 1))
    return result


def _collect_pairs(grid: np.ndarray) -> tuple[dict[int, tuple[Coord, Coord]], frozenset[Coord]]:
    height, width = grid.shape
    pairs: dict[int, tuple[Coord, Coord]] = {}
    terminals: list[Coord] = []
    for raw_color in np.unique(grid):
        color = int(raw_color)
        if color in (BG, GREY):
            continue
        coords = [tuple(map(int, pt)) for pt in np.argwhere(grid == color)]
        assert len(coords) == 2, "each endpoint color must appear exactly twice"
        for coord in coords:
            corridor_neighbors = [nb for nb in _neighbors4(coord, height, width) if int(grid[nb]) == GREY]
            assert len(corridor_neighbors) == 1, "each endpoint must touch exactly one grey corridor cell"
        pairs[color] = (coords[0], coords[1])
        terminals.extend(coords)
    assert pairs, "expected at least one colored endpoint pair"
    return pairs, frozenset(terminals)


def _build_adjacency(grid: np.ndarray) -> dict[Coord, tuple[Coord, ...]]:
    height, width = grid.shape
    traversable = {(r, c) for r in range(height) for c in range(width) if int(grid[r, c]) != BG}
    adjacency: dict[Coord, tuple[Coord, ...]] = {}
    for cell in traversable:
        neighbors = [nb for nb in _neighbors4(cell, height, width) if nb in traversable]
        adjacency[cell] = tuple(neighbors)
    return adjacency


def _reachable(
    adjacency: dict[Coord, tuple[Coord, ...]],
    start: Coord,
    goal: Coord,
    blocked: set[Coord],
) -> bool:
    queue = deque([start])
    seen = {start}
    while queue:
        cell = queue.popleft()
        if cell == goal:
            return True
        for nb in adjacency[cell]:
            if nb in seen:
                continue
            if nb != goal and nb in blocked:
                continue
            seen.add(nb)
            queue.append(nb)
    return False


def _enumerate_paths(
    grid: np.ndarray,
    adjacency: dict[Coord, tuple[Coord, ...]],
    start: Coord,
    goal: Coord,
    blocked_terminals: frozenset[Coord],
    max_paths: int,
) -> tuple[list[CandidatePath], bool]:
    paths: list[CandidatePath] = []
    path = [start]
    visited = {start}

    def dfs(cell: Coord) -> None:
        if len(paths) > max_paths:
            return
        if cell == goal:
            grey_cells = frozenset(pt for pt in path if int(grid[pt]) == GREY)
            paths.append((tuple(path), grey_cells))
            return

        next_cells = [nb for nb in adjacency[cell] if nb not in visited and nb not in blocked_terminals]
        next_cells.sort(key=lambda nb: (abs(nb[0] - goal[0]) + abs(nb[1] - goal[1]), len(adjacency[nb])))
        for nb in next_cells:
            if len(paths) > max_paths:
                return
            if nb != goal and not _reachable(adjacency, nb, goal, set(visited) | set(blocked_terminals)):
                continue
            visited.add(nb)
            path.append(nb)
            dfs(nb)
            path.pop()
            visited.remove(nb)

    dfs(start)
    truncated = len(paths) > max_paths
    return paths[:max_paths], truncated


def _solve_assignments(grid: np.ndarray, max_solutions: int) -> list[dict[int, CandidatePath]]:
    pairs, terminals = _collect_pairs(grid)
    adjacency = _build_adjacency(grid)

    candidates: dict[int, list[CandidatePath]] = {}
    for color, (start, goal) in pairs.items():
        blocked_terminals = frozenset(terminals - {start, goal})
        color_paths, truncated = _enumerate_paths(
            grid,
            adjacency,
            start,
            goal,
            blocked_terminals,
            max_paths=MAX_PATHS_PER_COLOR,
        )
        assert not truncated, "path search exceeded configured complexity bound"
        assert color_paths, "each endpoint pair must be connectable through the corridor network"
        candidates[color] = color_paths

    solutions: list[dict[int, CandidatePath]] = []
    dead_states: set[tuple[tuple[int, ...], frozenset[Coord]]] = set()

    def backtrack(remaining: tuple[int, ...], used_grey: frozenset[Coord], chosen: dict[int, CandidatePath]) -> None:
        if len(solutions) >= max_solutions:
            return
        state = (remaining, used_grey)
        if state in dead_states:
            return
        if not remaining:
            solutions.append(chosen.copy())
            return

        best_color = None
        feasible_paths: list[CandidatePath] | None = None
        for color in remaining:
            usable = [cand for cand in candidates[color] if cand[1].isdisjoint(used_grey)]
            if not usable:
                dead_states.add(state)
                return
            if feasible_paths is None or len(usable) < len(feasible_paths):
                best_color = color
                feasible_paths = usable

        assert best_color is not None and feasible_paths is not None
        next_remaining = tuple(color for color in remaining if color != best_color)
        feasible_paths.sort(key=lambda cand: len(cand[1]))

        found = False
        for candidate in feasible_paths:
            new_used = frozenset(set(used_grey) | set(candidate[1]))
            if any(not any(path[1].isdisjoint(new_used) for path in candidates[color]) for color in next_remaining):
                continue
            chosen[best_color] = candidate
            before = len(solutions)
            backtrack(next_remaining, new_used, chosen)
            if len(solutions) > before:
                found = True
            chosen.pop(best_color, None)
            if len(solutions) >= max_solutions:
                return

        if not found:
            dead_states.add(state)

    ordered_colors = tuple(sorted(candidates, key=lambda color: len(candidates[color])))
    backtrack(ordered_colors, frozenset(), {})
    return solutions


def _coarse_neighbors(node: CoarseNode, rows: int, cols: int) -> list[CoarseNode]:
    r, c = node
    result = []
    if r > 0:
        result.append((r - 1, c))
    if r + 1 < rows:
        result.append((r + 1, c))
    if c > 0:
        result.append((r, c - 1))
    if c + 1 < cols:
        result.append((r, c + 1))
    return result


def _edge_key(a: CoarseNode, b: CoarseNode) -> tuple[CoarseNode, CoarseNode]:
    return (a, b) if a <= b else (b, a)


def _edge_set(graph: dict[CoarseNode, set[CoarseNode]]) -> set[tuple[CoarseNode, CoarseNode]]:
    return {_edge_key(node, nb) for node, nbs in graph.items() for nb in nbs}


def _add_graph_path(graph: dict[CoarseNode, set[CoarseNode]], path: list[CoarseNode]) -> None:
    for a, b in zip(path, path[1:]):
        graph.setdefault(a, set()).add(b)
        graph.setdefault(b, set()).add(a)


def _graph_components(graph: dict[CoarseNode, set[CoarseNode]]) -> list[set[CoarseNode]]:
    remaining = set(graph)
    components: list[set[CoarseNode]] = []
    while remaining:
        start = remaining.pop()
        queue = [start]
        component = {start}
        while queue:
            node = queue.pop()
            for nb in graph[node]:
                if nb in remaining:
                    remaining.remove(nb)
                    component.add(nb)
                    queue.append(nb)
        components.append(component)
    return components


def _free_degree(
    node: CoarseNode,
    rows: int,
    cols: int,
    blocked: set[CoarseNode],
    visited: set[CoarseNode],
) -> int:
    return sum(1 for nb in _coarse_neighbors(node, rows, cols) if nb not in blocked and nb not in visited)


def _sample_route(rows: int, cols: int, blocked_nodes: set[CoarseNode]) -> list[CoarseNode] | None:
    free_nodes = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in blocked_nodes]
    if not free_nodes:
        return None

    candidate_lengths = list(range(ROUTE_EDGE_MIN, ROUTE_EDGE_MAX + 1))
    random.shuffle(candidate_lengths)

    for target_edges in candidate_lengths:
        for _ in range(ROUTE_BUILD_RETRIES):
            start = random.choice(free_nodes)
            path = [start]
            visited = {start}

            def dfs(node: CoarseNode) -> bool:
                if len(path) - 1 == target_edges:
                    return True
                next_nodes = [nb for nb in _coarse_neighbors(node, rows, cols) if nb not in blocked_nodes and nb not in visited]
                random.shuffle(next_nodes)
                next_nodes.sort(
                    key=lambda nb: _free_degree(nb, rows, cols, blocked_nodes, visited),
                    reverse=True,
                )
                for nb in next_nodes:
                    visited.add(nb)
                    path.append(nb)
                    if dfs(nb):
                        return True
                    path.pop()
                    visited.remove(nb)
                return False

            if dfs(start):
                return path.copy()
    return None


def _bfs_connector(
    rows: int,
    cols: int,
    start: CoarseNode,
    goal: CoarseNode,
    blocked_nodes: set[CoarseNode],
) -> list[CoarseNode] | None:
    queue = deque([start])
    prev: dict[CoarseNode, CoarseNode | None] = {start: None}
    while queue:
        node = queue.popleft()
        if node == goal:
            break
        neighbors = _coarse_neighbors(node, rows, cols)
        random.shuffle(neighbors)
        for nb in neighbors:
            if nb in prev:
                continue
            if nb != goal and nb in blocked_nodes:
                continue
            prev[nb] = node
            queue.append(nb)
    if goal not in prev:
        return None

    path = [goal]
    while prev[path[-1]] is not None:
        path.append(prev[path[-1]])
    path.reverse()
    return path


def _connect_components(
    rows: int,
    cols: int,
    graph: dict[CoarseNode, set[CoarseNode]],
    terminals: set[CoarseNode],
) -> bool:
    for _ in range(CONNECT_RETRIES):
        components = _graph_components(graph)
        if len(components) == 1:
            return True
        random.shuffle(components)
        made_progress = False
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                left = [node for node in components[i] if node not in terminals]
                right = [node for node in components[j] if node not in terminals]
                random.shuffle(left)
                random.shuffle(right)
                for a in left:
                    for b in right:
                        connector = _bfs_connector(rows, cols, a, b, set(graph) - {a, b})
                        if connector is None:
                            continue
                        _add_graph_path(graph, connector)
                        made_progress = True
                        break
                    if made_progress:
                        break
                if made_progress:
                    break
            if made_progress:
                break
        if not made_progress:
            return False
    return len(_graph_components(graph)) == 1


def _add_cycle(
    rows: int,
    cols: int,
    graph: dict[CoarseNode, set[CoarseNode]],
    terminals: set[CoarseNode],
) -> bool:
    anchors = [node for node in graph if node not in terminals]
    if len(anchors) < 2:
        return False

    existing_edges = _edge_set(graph)
    for _ in range(CONNECT_RETRIES):
        a, b = random.sample(anchors, 2)
        if _edge_key(a, b) in existing_edges:
            continue
        connector = _bfs_connector(rows, cols, a, b, set(graph) - {a, b})
        if connector is None:
            continue
        if len(connector) == 2 and _edge_key(a, b) in existing_edges:
            continue
        _add_graph_path(graph, connector)
        return True
    return False


def _fine_node(node: CoarseNode) -> Coord:
    return (1 + 2 * node[0], 1 + 2 * node[1])


def _fine_edge_cell(a: CoarseNode, b: CoarseNode) -> Coord:
    ra, ca = _fine_node(a)
    rb, cb = _fine_node(b)
    return ((ra + rb) // 2, (ca + cb) // 2)


def _render_instance(
    rows: int,
    cols: int,
    graph: dict[CoarseNode, set[CoarseNode]],
    routes: list[tuple[int, list[CoarseNode]]],
) -> tuple[np.ndarray, np.ndarray]:
    height = 2 * rows + 1
    width = 2 * cols + 1
    output = np.full((height, width), BG, dtype=int)

    graph_cells: set[Coord] = set()
    for node in graph:
        graph_cells.add(_fine_node(node))
    for a, b in _edge_set(graph):
        graph_cells.add(_fine_edge_cell(a, b))
    for cell in graph_cells:
        output[cell] = GREY

    route_cells: dict[Coord, int] = {}
    terminal_cells: set[Coord] = set()
    for color, path in routes:
        for idx, node in enumerate(path):
            cell = _fine_node(node)
            route_cells[cell] = color
            if idx == 0 or idx == len(path) - 1:
                terminal_cells.add(cell)
        for a, b in zip(path, path[1:]):
            route_cells[_fine_edge_cell(a, b)] = color

    for cell, color in route_cells.items():
        output[cell] = color

    input_grid = output.copy()
    for cell in route_cells:
        if cell not in terminal_cells:
            input_grid[cell] = GREY

    return input_grid, output


def _choose_pair_count(rows: int, cols: int) -> int:
    choices = PAIR_COUNT_LARGE if rows * cols >= 42 else PAIR_COUNT_SMALL
    return random.choice(choices)


def _build_candidate() -> tuple[np.ndarray, np.ndarray] | None:
    rows = random.randint(*COARSE_ROWS_RANGE)
    cols = random.randint(*COARSE_COLS_RANGE)
    pair_count = _choose_pair_count(rows, cols)
    colors = random.sample(COLOR_POOL, pair_count)

    graph: dict[CoarseNode, set[CoarseNode]] = {}
    routes: list[tuple[int, list[CoarseNode]]] = []
    terminals: set[CoarseNode] = set()

    for color in colors:
        route = _sample_route(rows, cols, set(graph))
        if route is None:
            return None
        _add_graph_path(graph, route)
        routes.append((color, route))
        terminals.add(route[0])
        terminals.add(route[-1])

    if not _connect_components(rows, cols, graph, terminals):
        return None

    if random.random() < LOOP_PROBABILITY:
        _add_cycle(rows, cols, graph, terminals)

    for _color, route in routes:
        if len(graph[route[0]]) != 1 or len(graph[route[-1]]) != 1:
            return None

    inp, expected = _render_instance(rows, cols, graph, routes)
    return inp, expected


def _is_unique_expected(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        solutions = _solve_assignments(inp, max_solutions=2)
    except AssertionError:
        return False
    if len(solutions) != 1:
        return False
    actual = main(inp)
    return np.array_equal(actual, expected)


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        for _ in range(GRAPH_BUILD_RETRIES):
            candidate = _build_candidate()
            if candidate is None:
                continue
            inp, expected = candidate
            if _is_unique_expected(inp, expected):
                return inp
    raise RuntimeError(f"failed to generate a valid unique input after {GENERATION_RETRIES} attempts")
