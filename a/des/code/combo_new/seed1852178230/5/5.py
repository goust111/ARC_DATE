from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
MAGENTA = 6
ORANGE = 7
MAROON = 9

SOURCE_COLORS = (RED, BLUE, YELLOW)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

GRID_MIN = 15
GRID_MAX = 29
MIN_AXIS_GAP = 3
MIN_AXIS_COUNT = 4
MAX_AXIS_COUNT = 6
MAX_GENERATION_ATTEMPTS = 400

Coord = tuple[int, int]
Lattice = tuple[int, int]


def _neighbors(shape: tuple[int, int], cell: Coord) -> list[Coord]:
    h, w = shape
    r, c = cell
    out: list[Coord] = []
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _mask_neighbors(mask: np.ndarray, cell: Coord) -> list[Coord]:
    return [nbr for nbr in _neighbors(mask.shape, cell) if bool(mask[nbr])]


def _degree(mask: np.ndarray, cell: Coord) -> int:
    return len(_mask_neighbors(mask, cell))


def _mix_colors(colors: list[int]) -> int:
    unique = frozenset(int(color) for color in colors)
    assert unique, "expected at least one incoming color"
    if len(unique) == 1:
        return next(iter(unique))
    if unique == frozenset((RED, BLUE)):
        return MAGENTA
    if unique == frozenset((RED, YELLOW)):
        return ORANGE
    if unique == frozenset((BLUE, YELLOW)):
        return GREEN
    assert unique == frozenset((RED, BLUE, YELLOW))
    return MAROON


def _component_cells(mask: np.ndarray) -> list[set[Coord]]:
    unseen = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    components: list[set[Coord]] = []
    while unseen:
        start = unseen.pop()
        comp = {start}
        queue = deque([start])
        while queue:
            cell = queue.popleft()
            for nbr in _mask_neighbors(mask, cell):
                if nbr in unseen:
                    unseen.remove(nbr)
                    comp.add(nbr)
                    queue.append(nbr)
        components.append(comp)
    return components


def _reachable_without(mask: np.ndarray, start: Coord, blocked: Coord) -> set[Coord]:
    seen = {start}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        for nbr in _mask_neighbors(mask, cell):
            if nbr == blocked or nbr in seen:
                continue
            seen.add(nbr)
            queue.append(nbr)
    return seen


def _bfs_path(mask: np.ndarray, start: Coord, goal: Coord) -> list[Coord]:
    queue = deque([start])
    parent: dict[Coord, Coord | None] = {start: None}
    while queue:
        cell = queue.popleft()
        if cell == goal:
            break
        for nbr in _mask_neighbors(mask, cell):
            if nbr not in parent:
                parent[nbr] = cell
                queue.append(nbr)
    assert goal in parent, "expected a path inside the pipe network"
    path: list[Coord] = []
    cur: Coord | None = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _source_endpoints(grid: np.ndarray) -> list[Coord]:
    mask = grid != BLACK
    out: list[Coord] = []
    for rr, cc in np.argwhere(mask):
        cell = (int(rr), int(cc))
        if int(grid[cell]) in SOURCE_COLORS and _degree(mask, cell) == 1:
            out.append(cell)
    return out


def _junctions(mask: np.ndarray, component: set[Coord]) -> list[Coord]:
    return [cell for cell in component if _degree(mask, cell) >= 3]


def _choose_mix_junction(mask: np.ndarray, sources: set[Coord], component: set[Coord]) -> tuple[Coord, list[dict[str, object]]]:
    candidates: list[tuple[tuple[int, int, int], Coord, list[dict[str, object]]]] = []
    fallbacks: list[tuple[tuple[int, int, int], Coord, list[dict[str, object]]]] = []
    for junction in _junctions(mask, component):
        branch_info: list[dict[str, object]] = []
        source_branch_count = 0
        output_branch_count = 0
        total_sources = 0
        for nbr in _mask_neighbors(mask, junction):
            cells = _reachable_without(mask, nbr, junction)
            count = sum(1 for source in sources if source in cells)
            branch_info.append({"neighbor": nbr, "cells": cells, "source_count": count})
            total_sources += count
            if count > 0:
                source_branch_count += 1
            else:
                output_branch_count += 1
        if source_branch_count >= 2:
            score = (source_branch_count, output_branch_count, total_sources)
            record = (score, junction, branch_info)
            if output_branch_count >= 1:
                candidates.append(record)
            else:
                fallbacks.append(record)

    if candidates:
        _score, junction, info = max(candidates, key=lambda item: item[0])
        return junction, info

    assert fallbacks, "expected at least one junction fed by multiple source branches"
    _score, junction, info = max(fallbacks, key=lambda item: item[0])
    return junction, info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate source colors through a gray pipe tree and mix them at source-gathering junctions.

    Input structure:
    - The background is black.
    - A one-pixel-wide orthogonal pipe network is drawn in gray.
    - Some degree-1 endpoints are recolored red, blue, or yellow to mark the source ends.
    - The network may also contain T or cross junctions, and some gray endpoints act as downstream sinks.

    Rule:
    - Copy the grid and keep the black background unchanged.
    - Each colored source endpoint sends its color along its unique branch until it reaches the first junction that is fed by multiple source-bearing branches.
    - At such a junction, mix the arrived colors with the visible ARC color rules:
      red+blue->magenta, red+yellow->orange, blue+yellow->green, and all three -> maroon; repeated copies of one color still stay that color.
    - Paint the junction with the mixed color, then propagate that mixed color through every branch of that junction that contains no source endpoint, including any downstream fanout subtree.

    Key insight:
    - The pipe network behaves like a source tree: source-bearing branches color inward first, the first multi-source junction determines the mixed color, and every source-free branch downstream of that junction becomes a uniform continuation of that mix.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    mask = g != BLACK
    sources = _source_endpoints(g)
    assert len(sources) >= 2, "expected at least two colored source endpoints"

    out = g.copy()
    source_set = set(sources)

    for component in _component_cells(mask):
        comp_sources = {cell for cell in source_set if cell in component}
        if not comp_sources:
            continue

        mix_cell, branch_info = _choose_mix_junction(mask, comp_sources, component)
        mix_color = _mix_colors([int(g[cell]) for cell in comp_sources])

        for source in comp_sources:
            path = _bfs_path(mask, source, mix_cell)
            for cell in path[:-1]:
                out[cell] = int(g[source])

        out[mix_cell] = mix_color

        for branch in branch_info:
            if int(branch["source_count"]) > 0:
                continue
            for cell in branch["cells"]:
                out[cell] = mix_color

    return out


def _sample_axis_positions(limit: int) -> list[int]:
    feasible = [
        count
        for count in range(MIN_AXIS_COUNT, MAX_AXIS_COUNT + 1)
        if 1 + (count - 1) * MIN_AXIS_GAP <= limit - 2
    ]
    assert feasible, "grid too small for the requested lattice spacing"
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


def _lat_neighbors(node: Lattice, rows_n: int, cols_n: int) -> list[Lattice]:
    r, c = node
    out: list[Lattice] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < rows_n:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < cols_n:
        out.append((r, c + 1))
    return out


def _add_edge(adj: dict[Lattice, set[Lattice]], a: Lattice, b: Lattice) -> None:
    adj.setdefault(a, set()).add(b)
    adj.setdefault(b, set()).add(a)


def _grow_simple_path(
    adj: dict[Lattice, set[Lattice]],
    used: set[Lattice],
    parent: Lattice,
    first_step: Lattice,
    rows_n: int,
    cols_n: int,
    min_edges: int,
    max_edges: int,
) -> Lattice | None:
    if first_step in used:
        return None
    _add_edge(adj, parent, first_step)
    used.add(first_step)

    prev = parent
    current = first_step
    direction = (current[0] - prev[0], current[1] - prev[1])
    target_edges = random.randint(min_edges, max_edges)
    built_edges = 1

    while built_edges < target_edges:
        choices = [nbr for nbr in _lat_neighbors(current, rows_n, cols_n) if nbr != prev and nbr not in used]
        if not choices:
            break
        straight = (current[0] + direction[0], current[1] + direction[1])
        if straight in choices and random.random() < 0.65:
            nxt = straight
        else:
            nxt = random.choice(choices)
        _add_edge(adj, current, nxt)
        used.add(nxt)
        prev, current = current, nxt
        direction = (current[0] - prev[0], current[1] - prev[1])
        built_edges += 1

    if built_edges < min_edges:
        return None
    return current


def _extend_chain(
    adj: dict[Lattice, set[Lattice]],
    used: set[Lattice],
    parent: Lattice,
    current: Lattice,
    rows_n: int,
    cols_n: int,
    max_extra_edges: int,
) -> tuple[Lattice, Lattice]:
    prev = parent
    node = current
    direction = (node[0] - prev[0], node[1] - prev[1])
    for _ in range(random.randint(0, max_extra_edges)):
        choices = [nbr for nbr in _lat_neighbors(node, rows_n, cols_n) if nbr != prev and nbr not in used]
        if not choices:
            break
        straight = (node[0] + direction[0], node[1] + direction[1])
        if straight in choices and random.random() < 0.7:
            nxt = straight
        else:
            nxt = random.choice(choices)
        _add_edge(adj, node, nxt)
        used.add(nxt)
        prev, node = node, nxt
        direction = (node[0] - prev[0], node[1] - prev[1])
    return prev, node


def _grow_output_subtree(
    adj: dict[Lattice, set[Lattice]],
    used: set[Lattice],
    parent: Lattice,
    current: Lattice,
    rows_n: int,
    cols_n: int,
    depth: int,
    max_depth: int,
    stats: dict[str, int],
) -> None:
    prev, node = _extend_chain(adj, used, parent, current, rows_n, cols_n, max_extra_edges=2)
    available = [nbr for nbr in _lat_neighbors(node, rows_n, cols_n) if nbr != prev and nbr not in used]
    can_branch = depth < max_depth and len(available) >= 2
    if not can_branch or random.random() >= 0.68:
        return

    child_count = 3 if len(available) >= 3 and random.random() < 0.3 else 2
    chosen = random.sample(available, child_count)
    stats["fanouts"] += 1
    for child in chosen:
        _add_edge(adj, node, child)
        used.add(child)
        _grow_output_subtree(adj, used, node, child, rows_n, cols_n, depth + 1, max_depth, stats)


def _graph_degrees(adj: dict[Lattice, set[Lattice]]) -> dict[Lattice, int]:
    return {node: len(neighbors) for node, neighbors in adj.items()}


def _path_nodes(adj: dict[Lattice, set[Lattice]], start: Lattice, goal: Lattice) -> list[Lattice]:
    queue = deque([start])
    parent: dict[Lattice, Lattice | None] = {start: None}
    while queue:
        node = queue.popleft()
        if node == goal:
            break
        for nbr in adj[node]:
            if nbr not in parent:
                parent[nbr] = node
                queue.append(nbr)
    assert goal in parent
    out: list[Lattice] = []
    cur: Lattice | None = goal
    while cur is not None:
        out.append(cur)
        cur = parent[cur]
    out.reverse()
    return out


def _draw_edge(grid: np.ndarray, a: Coord, b: Coord, color: int) -> None:
    ra, ca = a
    rb, cb = b
    if ra == rb:
        c0, c1 = sorted((ca, cb))
        grid[ra, c0 : c1 + 1] = color
    else:
        assert ca == cb
        r0, r1 = sorted((ra, rb))
        grid[r0 : r1 + 1, ca] = color


def _render_graph(
    height: int,
    width: int,
    rows: list[int],
    cols: list[int],
    adj: dict[Lattice, set[Lattice]],
    source_nodes: list[Lattice],
    source_colors: list[int],
) -> np.ndarray:
    grid = np.full((height, width), BLACK, dtype=int)
    coords = {node: (rows[node[0]], cols[node[1]]) for node in adj}
    drawn_edges: set[tuple[Lattice, Lattice]] = set()
    for a, neighbors in adj.items():
        for b in neighbors:
            edge = tuple(sorted((a, b)))
            if edge in drawn_edges:
                continue
            drawn_edges.add(edge)
            _draw_edge(grid, coords[a], coords[b], GRAY)
    for node, color in zip(source_nodes, source_colors):
        grid[coords[node]] = int(color)
    return grid


def _choose_source_colors() -> list[int]:
    kind = random.choice(["same", "rb", "ry", "by", "tri"])
    if kind == "same":
        color = random.choice(SOURCE_COLORS)
        count = 3 if random.random() < 0.45 else 2
        return [color] * count
    if kind == "rb":
        colors = [RED, BLUE]
    elif kind == "ry":
        colors = [RED, YELLOW]
    elif kind == "by":
        colors = [BLUE, YELLOW]
    else:
        return [RED, BLUE, YELLOW]
    if random.random() < 0.25:
        colors = colors + [random.choice(colors)]
    return colors


def _generate_candidate() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    rows = _sample_axis_positions(height)
    cols = _sample_axis_positions(width)
    rows_n, cols_n = len(rows), len(cols)
    if rows_n < 4 or cols_n < 4:
        return None

    interior = [(r, c) for r in range(1, rows_n - 1) for c in range(1, cols_n - 1)]
    if not interior:
        return None

    source_colors = _choose_source_colors()
    source_count = len(source_colors)
    output_count = 1 if source_count >= 3 else (2 if random.random() < 0.5 else 1)
    total_degree = source_count + output_count
    if total_degree > 4:
        return None

    main_node = random.choice(interior)
    directions = DIRS.copy()
    random.shuffle(directions)
    chosen_dirs = directions[:total_degree]
    source_dirs = chosen_dirs[:source_count]
    output_dirs = chosen_dirs[source_count:]

    adj: dict[Lattice, set[Lattice]] = {main_node: set()}
    used = {main_node}
    source_nodes: list[Lattice] = []

    for dr, dc in source_dirs:
        first = (main_node[0] + dr, main_node[1] + dc)
        if not (0 <= first[0] < rows_n and 0 <= first[1] < cols_n):
            return None
        leaf = _grow_simple_path(adj, used, main_node, first, rows_n, cols_n, min_edges=1, max_edges=3)
        if leaf is None:
            return None
        source_nodes.append(leaf)

    stats = {"fanouts": 0}
    max_depth = 2 if random.random() < 0.55 else 1
    for dr, dc in output_dirs:
        first = (main_node[0] + dr, main_node[1] + dc)
        if not (0 <= first[0] < rows_n and 0 <= first[1] < cols_n):
            return None
        if first in used:
            return None
        _add_edge(adj, main_node, first)
        used.add(first)
        _grow_output_subtree(adj, used, main_node, first, rows_n, cols_n, depth=0, max_depth=max_depth, stats=stats)

    degrees = _graph_degrees(adj)
    if degrees.get(main_node) != total_degree:
        return None
    if any(degrees[node] != 1 for node in source_nodes):
        return None

    leaf_nodes = [node for node, deg in degrees.items() if deg == 1]
    if len(leaf_nodes) <= len(source_nodes):
        return None

    if random.random() < 0.7 and stats["fanouts"] == 0:
        return None

    grid = _render_graph(height, width, rows, cols, adj, source_nodes, source_colors)
    source_cells = _source_endpoints(grid)
    if len(source_cells) != len(source_nodes):
        return None

    mask = grid != BLACK
    junction_count = sum(_degree(mask, (int(r), int(c))) >= 3 for r, c in np.argwhere(mask))
    if junction_count < 1:
        return None

    expected_mix = _mix_colors(source_colors)
    actual = main(grid)
    if np.array_equal(actual, grid):
        return None
    if expected_mix not in set(map(int, actual.ravel())):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pipe-mixing input")
