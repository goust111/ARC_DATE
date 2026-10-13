from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DRIVER = 5
RING_CW = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
)
TANGENT_DELTAS = ((3, 0), (-3, 0), (0, 3), (0, -3))
ALL_COLORS = tuple(range(10))


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _ring_cells(center: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = center
    return [(r + dr, c + dc) for dr, dc in RING_CW]


def _find_ring_centers(g: np.ndarray, bg: int) -> list[tuple[int, int]]:
    assert bg != DRIVER, "the grey driver center must differ from the background"
    h, w = g.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            center_color = int(g[r, c])
            if center_color not in (bg, DRIVER):
                continue
            if all(int(g[r + dr, c + dc]) != bg for dr, dc in RING_CW):
                centers.append((r, c))

    assert centers, "expected at least one 3x3 ring"
    assert sum(int(g[r, c]) == DRIVER for r, c in centers) == 1, "expected one grey driver center"

    occupied: set[tuple[int, int]] = set()
    for center in centers:
        cells = set(_ring_cells(center))
        assert not (cells & occupied), "ring perimeters must not overlap"
        occupied |= cells
        assert center not in occupied, "ring centers must stay inside their own empty middle cell"
    return sorted(centers)


def _build_graph(centers: list[tuple[int, int]]) -> list[list[int]]:
    graph = [[] for _ in centers]
    for i, (r0, c0) in enumerate(centers):
        for j in range(i + 1, len(centers)):
            r1, c1 = centers[j]
            if (r1 - r0, c1 - c0) in TANGENT_DELTAS:
                graph[i].append(j)
                graph[j].append(i)

    if len(centers) == 1:
        return graph

    edge_count = sum(len(nbs) for nbs in graph) // 2
    assert edge_count == len(centers) - 1, "tangent relation must be a tree"

    seen = {0}
    q = deque([0])
    while q:
        node = q.popleft()
        for nb in graph[node]:
            if nb not in seen:
                seen.add(nb)
                q.append(nb)
    assert len(seen) == len(centers), "all rings must be connected through tangencies"
    return graph


def _distances_from(graph: list[list[int]], start: int) -> list[int]:
    dist = [-1] * len(graph)
    dist[start] = 0
    q = deque([start])
    while q:
        node = q.popleft()
        for nb in graph[node]:
            if dist[nb] == -1:
                dist[nb] = dist[node] + 1
                q.append(nb)
    assert all(d >= 0 for d in dist)
    return dist


def _rotate_ring(out: np.ndarray, src: np.ndarray, center: tuple[int, int], clockwise: bool) -> None:
    cells = _ring_cells(center)
    values = [int(src[r, c]) for r, c in cells]
    shift = 1 if clockwise else -1
    for i, value in enumerate(values):
        rr, cc = cells[(i + shift) % len(cells)]
        out[rr, cc] = value


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate a tangent tree of 3x3 color rings like meshed gears from the grey driver.

    Input structure:
    - The grid has one dominant background color.
    - Each object is a non-overlapping 3x3 ring: eight non-background perimeter cells around one center cell.
    - Exactly one center cell is grey, marking the driver ring; every other ring center is background.
    - Ring boxes touch only by full horizontal or vertical sides, and those tangencies form one connected tree.

    Rule:
    - The driver ring's eight perimeter colors move one step clockwise around its center.
    - Across every side tangency, the neighboring ring turns in the opposite direction.
    - Therefore rings at even tangent distance from the driver rotate clockwise, and rings at odd distance rotate counterclockwise.
    - All center cells stay unchanged, including the grey driver center.

    Key insight:
    - The side-tangent chain is a gear train, so a ring's direction is determined by the parity of its tree distance from the grey driver, not by its own colors or absolute position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    centers = _find_ring_centers(g, bg)
    driver = next(i for i, (r, c) in enumerate(centers) if int(g[r, c]) == DRIVER)
    graph = _build_graph(centers)
    distances = _distances_from(graph, driver)

    out = g.copy()
    for i, center in enumerate(centers):
        _rotate_ring(out, g, center, clockwise=(distances[i] % 2 == 0))
    return out


# --------------- generator ---------------

GEN_RETRIES = 300
NODE_COUNT_RANGE = (3, 8)
BACKGROUND_COLORS = tuple(c for c in ALL_COLORS if c != DRIVER)
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _neighbors(node: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = node
    return [(r + dr, c + dc) for dr, dc in DIRS]


def _normalize_tree(
    nodes: set[tuple[int, int]],
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
) -> tuple[list[tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int]]]]:
    min_r = min(r for r, _ in nodes)
    min_c = min(c for _, c in nodes)
    mapping = {node: (node[0] - min_r, node[1] - min_c) for node in nodes}
    norm_nodes = sorted(mapping.values())
    norm_edges = [(mapping[a], mapping[b]) for a, b in edges]
    return norm_nodes, norm_edges


def _add_random_leaf(
    nodes: set[tuple[int, int]],
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
) -> bool:
    options: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for parent in nodes:
        for nb in _neighbors(parent):
            if nb in nodes:
                continue
            if sum(adj in nodes for adj in _neighbors(nb)) == 1:
                options.append((parent, nb))
    if not options:
        return False
    parent, child = random.choice(options)
    nodes.add(child)
    edges.append((parent, child))
    return True


def _make_chain(n: int) -> tuple[list[tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int]]]] | None:
    for _ in range(80):
        nodes = {(0, 0)}
        edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
        end = (0, 0)
        while len(nodes) < n:
            candidates = [
                nb
                for nb in _neighbors(end)
                if nb not in nodes and sum(adj in nodes for adj in _neighbors(nb)) == 1
            ]
            if not candidates:
                break
            child = random.choice(candidates)
            nodes.add(child)
            edges.append((end, child))
            end = child
        if len(nodes) == n:
            return _normalize_tree(nodes, edges)
    return None


def _make_branching_tree(n: int) -> tuple[list[tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int]]]] | None:
    if n < 4:
        return None
    for _ in range(80):
        nodes = {(0, 0)}
        edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for dr, dc in random.sample(DIRS, 3):
            child = (dr, dc)
            nodes.add(child)
            edges.append(((0, 0), child))
        while len(nodes) < n and _add_random_leaf(nodes, edges):
            pass
        if len(nodes) == n:
            return _normalize_tree(nodes, edges)
    return None


def _abstract_graph(
    nodes: list[tuple[int, int]],
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    graph = {node: [] for node in nodes}
    for a, b in edges:
        graph[a].append(b)
        graph[b].append(a)
    return graph


def _abstract_distances(
    graph: dict[tuple[int, int], list[tuple[int, int]]],
    start: tuple[int, int],
) -> dict[tuple[int, int], int]:
    dist = {start: 0}
    q = deque([start])
    while q:
        node = q.popleft()
        for nb in graph[node]:
            if nb not in dist:
                dist[nb] = dist[node] + 1
                q.append(nb)
    return dist


def _choose_driver(
    nodes: list[tuple[int, int]],
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
) -> tuple[int, int] | None:
    graph = _abstract_graph(nodes, edges)
    candidates = []
    for node in nodes:
        dist = _abstract_distances(graph, node)
        values = list(dist.values())
        if max(values) >= 2 and any(d % 2 == 1 for d in values) and any(d > 0 and d % 2 == 0 for d in values):
            candidates.append(node)
    return random.choice(candidates) if candidates else None


def _choose_canvas(lat_h: int, lat_w: int, ring_count: int) -> tuple[int, int, int, int] | None:
    base_h = 3 * lat_h
    base_w = 3 * lat_w
    min_h = base_h + 2
    min_w = base_w + 2
    if min_h > 30 or min_w > 30:
        return None

    for _ in range(80):
        h = random.randint(min_h, min(30, base_h + 12))
        w = random.randint(min_w, min(30, base_w + 12))
        top = random.randint(1, h - base_h - 1)
        left = random.randint(1, w - base_w - 1)
        background_cells = h * w - (8 * ring_count + 1)
        if background_cells > ring_count:
            return h, w, top, left
    return None


def _render_grid(
    nodes: list[tuple[int, int]],
    driver_node: tuple[int, int],
    bg: int,
) -> tuple[np.ndarray, dict[tuple[int, int], tuple[int, int]]] | None:
    lat_h = max(r for r, _ in nodes) + 1
    lat_w = max(c for _, c in nodes) + 1
    canvas = _choose_canvas(lat_h, lat_w, len(nodes))
    if canvas is None:
        return None
    h, w, top, left = canvas
    grid = np.full((h, w), bg, dtype=int)
    palette = [c for c in ALL_COLORS if c not in (bg, DRIVER)]
    assert len(palette) == 8

    centers: dict[tuple[int, int], tuple[int, int]] = {}
    for node in nodes:
        rr = top + 3 * node[0] + 1
        cc = left + 3 * node[1] + 1
        centers[node] = (rr, cc)
        colors = random.sample(palette, 8)
        for (dr, dc), color in zip(RING_CW, colors):
            grid[rr + dr, cc + dc] = color
        grid[rr, cc] = DRIVER if node == driver_node else bg
    return grid, centers


def _expected_from_plan(
    grid: np.ndarray,
    centers_by_node: dict[tuple[int, int], tuple[int, int]],
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
    driver_node: tuple[int, int],
) -> np.ndarray:
    graph = _abstract_graph(list(centers_by_node), edges)
    distances = _abstract_distances(graph, driver_node)
    out = grid.copy()
    for node, center in centers_by_node.items():
        _rotate_ring(out, grid, center, clockwise=(distances[node] % 2 == 0))
    return out


def _semantically_valid(
    grid: np.ndarray,
    centers_by_node: dict[tuple[int, int], tuple[int, int]],
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
    driver_node: tuple[int, int],
    bg: int,
    require_branch: bool,
) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30 or _get_bg(grid) != bg:
        return False
    planned_centers = set(centers_by_node.values())
    try:
        detected = set(_find_ring_centers(grid, bg))
        graph = _abstract_graph(list(centers_by_node), edges)
        distances = _abstract_distances(graph, driver_node)
    except AssertionError:
        return False
    if detected != planned_centers:
        return False
    if len(distances) != len(centers_by_node):
        return False
    values = list(distances.values())
    if not any(d % 2 == 1 for d in values):
        return False
    if not any(d > 0 and d % 2 == 0 for d in values):
        return False
    if require_branch and max(len(nbs) for nbs in graph.values()) < 3:
        return False

    expected = _expected_from_plan(grid, centers_by_node, edges, driver_node)
    if np.array_equal(expected, grid):
        return False
    for center in planned_centers:
        before = [int(grid[r, c]) for r, c in _ring_cells(center)]
        after = [int(expected[r, c]) for r, c in _ring_cells(center)]
        if before == after:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*NODE_COUNT_RANGE)
    require_branch = random.random() < 0.7
    if require_branch:
        n = max(n, 4)
        tree = _make_branching_tree(n)
    else:
        tree = _make_chain(n)
    if tree is None:
        return None
    nodes, edges = tree

    driver_node = _choose_driver(nodes, edges)
    if driver_node is None:
        return None
    bg = random.choice(BACKGROUND_COLORS)
    rendered = _render_grid(nodes, driver_node, bg)
    if rendered is None:
        return None
    grid, centers_by_node = rendered
    if not _semantically_valid(grid, centers_by_node, edges, driver_node, bg, require_branch):
        return None

    expected = _expected_from_plan(grid, centers_by_node, edges, driver_node)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    return grid if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gear-ring-chain input")
