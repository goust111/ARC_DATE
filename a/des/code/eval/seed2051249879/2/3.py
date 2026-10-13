from __future__ import annotations

from collections import deque
from itertools import combinations
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
MAROON = 9

SOURCE_COLORS = (RED, BLUE, YELLOW)
INPUT_COLORS = {BLACK, BLUE, RED, YELLOW, GREY, MAGENTA}
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GRID_MIN = 11
GRID_MAX = 21
GENERATION_ATTEMPTS = 400
TREE_ATTEMPTS = 200

RECIPE_SINGLE = "single"
RECIPE_DOUBLE = "double"
RECIPE_TRIPLE = "triple"
RECIPE_UNREACHABLE = "unreachable"
RECIPES = (RECIPE_SINGLE, RECIPE_DOUBLE, RECIPE_TRIPLE, RECIPE_UNREACHABLE)


Coord = tuple[int, int]
LatticeCoord = tuple[int, int]


def _neighbors(cell: Coord, shape: tuple[int, int]) -> list[Coord]:
    h, w = shape
    r, c = cell
    result = []
    for dr, dc in DIR4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            result.append((rr, cc))
    return result


def _component_cells(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[Coord] = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                cell = q.popleft()
                comp.append(cell)
                for nb in _neighbors(cell, mask.shape):
                    if mask[nb] and not seen[nb]:
                        seen[nb] = True
                        q.append(nb)
            components.append(comp)
    return components


def _bfs_parents(start: Coord, allowed: set[Coord], shape: tuple[int, int]) -> dict[Coord, Coord | None]:
    parent: dict[Coord, Coord | None] = {start: None}
    q = deque([start])
    while q:
        cell = q.popleft()
        for nb in _neighbors(cell, shape):
            if nb not in allowed or nb in parent:
                continue
            parent[nb] = cell
            q.append(nb)
    return parent


def _mix_colors(colors: set[int]) -> int:
    key = frozenset(int(color) for color in colors)
    if key == frozenset({RED}):
        return RED
    if key == frozenset({BLUE}):
        return BLUE
    if key == frozenset({YELLOW}):
        return YELLOW
    if key == frozenset({RED, BLUE}):
        return MAROON
    if key == frozenset({RED, YELLOW}):
        return ORANGE
    if key == frozenset({BLUE, YELLOW}):
        return GREEN
    if key == frozenset({RED, BLUE, YELLOW}):
        return GREY
    raise AssertionError(f"unsupported source mix: {sorted(key)}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate source colors through a tree-like pipe network and mix colors where source-to-terminal paths overlap.

    Input structure:
    - Black is the background.
    - Grey cells form single-pixel-wide horizontal/vertical pipe trees.
    - Red, blue, and yellow leaf cells are signal sources.
    - Magenta leaf cells are terminals.

    Rule:
    - For every source/terminal pair in the same pipe component, the source color
      occupies the unique path between them.
    - Grey and magenta cells take the mixed color of all source colors whose
      source-to-terminal paths pass through that cell.
    - Two-color mixes are red+blue=maroon, red+yellow=orange,
      blue+yellow=green; all three colors mix back to grey.
    - A terminal with no connected source remains magenta.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    mask = g != BLACK
    contributions: dict[Coord, set[int]] = {}

    for comp in _component_cells(mask):
        comp_set = set(comp)
        sources = [cell for cell in comp if int(g[cell]) in SOURCE_COLORS]
        terminals = [cell for cell in comp if int(g[cell]) == MAGENTA]
        if not sources or not terminals:
            continue

        for source in sources:
            source_color = int(g[source])
            parent = _bfs_parents(source, comp_set, g.shape)
            for terminal in terminals:
                if terminal not in parent:
                    continue
                cur = terminal
                while cur is not None and cur != source:
                    contributions.setdefault(cur, set()).add(source_color)
                    cur = parent[cur]

    out = g.copy()
    for cell, colors in contributions.items():
        if int(g[cell]) in SOURCE_COLORS:
            continue
        out[cell] = _mix_colors(colors)
    return out


def _rand_odd(lo: int, hi: int) -> int:
    options = [n for n in range(lo, hi + 1) if n % 2 == 1]
    return random.choice(options)


def _node_to_cell(node: LatticeCoord) -> Coord:
    return (1 + 2 * node[0], 1 + 2 * node[1])


def _edge_midpoint(a: LatticeCoord, b: LatticeCoord) -> Coord:
    ar, ac = _node_to_cell(a)
    br, bc = _node_to_cell(b)
    return ((ar + br) // 2, (ac + bc) // 2)


def _lattice_neighbors(node: LatticeCoord, shape: tuple[int, int]) -> list[LatticeCoord]:
    h, w = shape
    r, c = node
    result = []
    for dr, dc in DIR4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            result.append((rr, cc))
    return result


def _canon_edge(a: LatticeCoord, b: LatticeCoord) -> tuple[LatticeCoord, LatticeCoord]:
    return (a, b) if a <= b else (b, a)


def _tree_adjacency(nodes: set[LatticeCoord], edges: set[tuple[LatticeCoord, LatticeCoord]]) -> dict[LatticeCoord, set[LatticeCoord]]:
    adj = {node: set() for node in nodes}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def _component_actual_cells(
    nodes: set[LatticeCoord], edges: set[tuple[LatticeCoord, LatticeCoord]]
) -> set[Coord]:
    cells = {_node_to_cell(node) for node in nodes}
    for a, b in edges:
        cells.add(_edge_midpoint(a, b))
    return cells


def _grow_tree(
    lattice_shape: tuple[int, int],
    min_nodes: int,
    max_nodes: int,
    min_leaves: int,
    forbidden_cells: set[Coord],
) -> tuple[set[LatticeCoord], set[tuple[LatticeCoord, LatticeCoord]], dict[LatticeCoord, set[LatticeCoord]], list[LatticeCoord], set[Coord]] | None:
    lattice_h, lattice_w = lattice_shape
    max_nodes = min(max_nodes, lattice_h * lattice_w)
    if min_nodes > max_nodes:
        return None

    for _ in range(TREE_ATTEMPTS):
        target_nodes = random.randint(min_nodes, max_nodes)
        starts = [
            (r, c)
            for r in range(lattice_h)
            for c in range(lattice_w)
            if _node_to_cell((r, c)) not in forbidden_cells
        ]
        if not starts:
            return None

        start = random.choice(starts)
        nodes: set[LatticeCoord] = {start}
        edges: set[tuple[LatticeCoord, LatticeCoord]] = set()
        degrees: dict[LatticeCoord, int] = {start: 0}

        while len(nodes) < target_nodes:
            frontier: list[tuple[LatticeCoord, LatticeCoord]] = []
            for node in nodes:
                for nb in _lattice_neighbors(node, lattice_shape):
                    if nb in nodes:
                        continue
                    new_node_cell = _node_to_cell(nb)
                    new_edge_cell = _edge_midpoint(node, nb)
                    if new_node_cell in forbidden_cells or new_edge_cell in forbidden_cells:
                        continue
                    frontier.append((node, nb))
            if not frontier:
                break

            parent, child = random.choice(frontier)
            nodes.add(child)
            edges.add(_canon_edge(parent, child))
            degrees[parent] = degrees.get(parent, 0) + 1
            degrees[child] = 1

        adj = _tree_adjacency(nodes, edges)
        leaves = [node for node, nbs in adj.items() if len(nbs) == 1]
        actual_cells = _component_actual_cells(nodes, edges)
        if len(nodes) >= min_nodes and len(leaves) >= min_leaves and not (actual_cells & forbidden_cells):
            return nodes, edges, adj, leaves, actual_cells
    return None


def _tree_distances(adj: dict[LatticeCoord, set[LatticeCoord]], origin: LatticeCoord) -> dict[LatticeCoord, int]:
    dist = {origin: 0}
    q = deque([origin])
    while q:
        node = q.popleft()
        for nb in adj[node]:
            if nb not in dist:
                dist[nb] = dist[node] + 1
                q.append(nb)
    return dist


def _choose_source_leaves(leaves: list[LatticeCoord], adj: dict[LatticeCoord, set[LatticeCoord]], count: int) -> list[LatticeCoord]:
    assert 1 <= count < len(leaves)
    if count == 1:
        return [random.choice(leaves)]

    dists = {leaf: _tree_distances(adj, leaf) for leaf in leaves}
    best: tuple[int, tuple[int, ...], tuple[LatticeCoord, ...]] | None = None
    ordered = leaves[:]
    random.shuffle(ordered)
    for combo in combinations(ordered, count):
        score = sum(dists[a][b] for a, b in combinations(combo, 2))
        tie = tuple(sorted(dists[leaf][other] for leaf in combo for other in leaves if other != leaf))
        candidate = (score, tie, combo)
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    return list(best[2])


def _draw_component(
    grid: np.ndarray,
    nodes: set[LatticeCoord],
    edges: set[tuple[LatticeCoord, LatticeCoord]],
    leaf_colors: dict[LatticeCoord, int],
) -> dict[LatticeCoord, Coord]:
    for node in nodes:
        grid[_node_to_cell(node)] = GREY
    for a, b in edges:
        grid[_edge_midpoint(a, b)] = GREY
    actual_leaf_cells: dict[LatticeCoord, Coord] = {}
    for leaf, color in leaf_colors.items():
        cell = _node_to_cell(leaf)
        actual_leaf_cells[leaf] = cell
        grid[cell] = int(color)
    return actual_leaf_cells


def _component_degrees(grid: np.ndarray, component: list[Coord]) -> dict[Coord, int]:
    mask = grid != BLACK
    return {cell: sum(1 for nb in _neighbors(cell, grid.shape) if mask[nb]) for cell in component}


def _is_valid_input(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        return False
    if not set(np.unique(g)).issubset(INPUT_COLORS):
        return False

    mask = g != BLACK
    if not mask.any():
        return False

    for comp in _component_cells(mask):
        degrees = _component_degrees(g, comp)
        edge_count = sum(degrees.values()) // 2
        if edge_count != len(comp) - 1:
            return False

        leaves = {cell for cell, degree in degrees.items() if degree == 1}
        marked_leaves = {cell for cell in comp if int(g[cell]) in SOURCE_COLORS or int(g[cell]) == MAGENTA}
        if leaves != marked_leaves:
            return False

        for cell, degree in degrees.items():
            color = int(g[cell])
            if color == GREY and degree <= 1:
                return False
            if color in SOURCE_COLORS or color == MAGENTA:
                if degree != 1:
                    return False
    return True


def _expected_mix_for_sources(source_colors: list[int]) -> int:
    return _mix_colors(set(source_colors))


def _main_component_recipe(recipe: str) -> list[int]:
    if recipe == RECIPE_SINGLE:
        return [random.choice(SOURCE_COLORS)]
    if recipe == RECIPE_DOUBLE:
        return random.sample(list(SOURCE_COLORS), 2)
    if recipe == RECIPE_TRIPLE:
        return list(SOURCE_COLORS)
    if recipe == RECIPE_UNREACHABLE:
        count = random.choice((2, 3))
        return random.sample(list(SOURCE_COLORS), count)
    raise AssertionError(recipe)


def _validate_recipe_output(
    recipe: str,
    inp: np.ndarray,
    out: np.ndarray,
    main_terminals: list[Coord],
    main_source_colors: list[int],
    unreachable_terminals: list[Coord],
) -> bool:
    if not np.any(out != inp):
        return False

    if recipe == RECIPE_SINGLE:
        target = main_source_colors[0]
        return bool(main_terminals) and all(int(out[cell]) == target for cell in main_terminals)

    if recipe == RECIPE_DOUBLE:
        target = _expected_mix_for_sources(main_source_colors)
        return bool(main_terminals) and all(int(out[cell]) == target for cell in main_terminals)

    if recipe == RECIPE_TRIPLE:
        return bool(main_terminals) and all(int(out[cell]) == GREY for cell in main_terminals)

    if recipe == RECIPE_UNREACHABLE:
        if not unreachable_terminals or not all(int(out[cell]) == MAGENTA for cell in unreachable_terminals):
            return False
        target = _expected_mix_for_sources(main_source_colors)
        return bool(main_terminals) and all(int(out[cell]) == target for cell in main_terminals)

    raise AssertionError(recipe)


def _gen_attempt() -> np.ndarray | None:
    recipe = random.choice(RECIPES)
    min_size = 13 if recipe in {RECIPE_TRIPLE, RECIPE_UNREACHABLE} else GRID_MIN
    h = _rand_odd(min_size, GRID_MAX)
    w = _rand_odd(min_size, GRID_MAX)
    grid = np.full((h, w), BLACK, dtype=int)

    lattice_shape = ((h - 1) // 2, (w - 1) // 2)
    occupied: set[Coord] = set()

    main_source_colors = _main_component_recipe(recipe)
    main_source_count = len(main_source_colors)
    main_tree = _grow_tree(
        lattice_shape=lattice_shape,
        min_nodes=max(3, main_source_count + 2),
        max_nodes=min(14, lattice_shape[0] * lattice_shape[1]),
        min_leaves=main_source_count + 1,
        forbidden_cells=occupied,
    )
    if main_tree is None:
        return None

    nodes, edges, adj, leaves, actual_cells = main_tree
    source_leaves = _choose_source_leaves(leaves, adj, main_source_count)
    source_leaf_set = set(source_leaves)
    terminal_leaves = [leaf for leaf in leaves if leaf not in source_leaf_set]
    if not terminal_leaves:
        return None

    random.shuffle(main_source_colors)
    leaf_colors = {leaf: MAGENTA for leaf in terminal_leaves}
    for leaf, color in zip(source_leaves, main_source_colors):
        leaf_colors[leaf] = int(color)

    actual_leaf_cells = _draw_component(grid, nodes, edges, leaf_colors)
    main_terminals = [actual_leaf_cells[leaf] for leaf in terminal_leaves]
    occupied |= actual_cells

    unreachable_terminals: list[Coord] = []
    if recipe == RECIPE_UNREACHABLE:
        silent_tree = _grow_tree(
            lattice_shape=lattice_shape,
            min_nodes=3,
            max_nodes=min(10, lattice_shape[0] * lattice_shape[1]),
            min_leaves=2,
            forbidden_cells=occupied,
        )
        if silent_tree is None:
            return None
        snodes, sedges, _sadj, sleaves, sactual = silent_tree
        silent_colors = {leaf: MAGENTA for leaf in sleaves}
        silent_actual_leaf_cells = _draw_component(grid, snodes, sedges, silent_colors)
        unreachable_terminals = [silent_actual_leaf_cells[leaf] for leaf in sleaves]
        occupied |= sactual

    if not _is_valid_input(grid):
        return None

    out = main(grid)
    if not _validate_recipe_output(recipe, grid, out, main_terminals, main_source_colors, unreachable_terminals):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-pipe input")
