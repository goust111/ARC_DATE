from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 8
RED = 2
BLUE = 1
GREEN = 3
YELLOW = 4
MAROON = 9

DEPTH_COLORS = {
    0: RED,
    1: BLUE,
    2: GREEN,
    3: YELLOW,
}

INPUT_COLORS = [0, 1, 2, 3, 4, 5, 6, 7, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_BY_DIR = {
    (-1, 0): "up",
    (1, 0): "down",
    (0, -1): "left",
    (0, 1): "right",
}
DIR_BY_SIDE = {name: delta for delta, name in SIDE_BY_DIR.items()}

OBJECT_COUNT_RANGE = (4, 8)
COARSE_SIZE_RANGE = (5, 7)
TILE_SIZE_CHOICES = (4, 5, 6)
GEN_RETRIES = 500


def _touches_edge(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _label_components(grid: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    labels = np.full((h, w), -1, dtype=int)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if int(g[r, c]) == BG or int(labels[r, c]) != -1:
                continue
            color = int(g[r, c])
            q = deque([(r, c)])
            labels[r, c] = len(components)
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIRS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and int(labels[nr, nc]) == -1 and int(g[nr, nc]) == color:
                        labels[nr, nc] = len(components)
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))

    return components, labels


def _adjacency_graph(labels: np.ndarray) -> list[set[int]]:
    h, w = labels.shape
    n = int(labels.max()) + 1
    graph = [set() for _ in range(n)]
    for r in range(h):
        for c in range(w):
            src = int(labels[r, c])
            if src < 0:
                continue
            for dr, dc in ((1, 0), (0, 1)):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    dst = int(labels[nr, nc])
                    if dst >= 0 and dst != src:
                        graph[src].add(dst)
                        graph[dst].add(src)
    return graph


def _analyze_tree(grid: np.ndarray) -> tuple[list[np.ndarray], dict[int, int], list[set[int]], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    components, labels = _label_components(g)
    assert components, "expected at least one foreground object"

    roots = [i for i, coords in enumerate(components) if _touches_edge(coords, g.shape)]
    assert len(roots) == 1, "expected exactly one edge-touching root object"
    root = roots[0]

    graph = _adjacency_graph(labels)
    edge_count = sum(len(neighbors) for neighbors in graph) // 2
    assert edge_count == len(components) - 1, "object adjacency must form a tree"

    depths = {root: 0}
    q = deque([root])
    while q:
        node = q.popleft()
        for nbr in graph[node]:
            if nbr not in depths:
                depths[nbr] = depths[node] + 1
                q.append(nbr)

    assert len(depths) == len(components), "object adjacency tree must be connected"
    return components, depths, graph, root


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each foreground object by its depth in the object-adjacency tree.

    The unique object touching the grid edge is the root (depth 0). Objects that
    are 4-adjacent to the root are depth 1, their unvisited neighbors are depth
    2, and so on. Depths 0..3 map to red, blue, green, yellow; deeper levels are
    recolored maroon.
    """
    g = np.asarray(grid, dtype=int)
    components, depths, _graph, _root = _analyze_tree(g)

    out = np.full_like(g, BG)
    for idx, coords in enumerate(components):
        color = int(DEPTH_COLORS.get(depths[idx], MAROON))
        out[coords[:, 0], coords[:, 1]] = color
    return out


def _is_boundary_cell(pos: tuple[int, int], coarse_shape: tuple[int, int]) -> bool:
    r, c = pos
    h, w = coarse_shape
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _coarse_neighbors(pos: tuple[int, int], coarse_shape: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = pos
    h, w = coarse_shape
    out = []
    for dr, dc in DIRS4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _candidate_cells(
    occupied: set[tuple[int, int]],
    coarse_shape: tuple[int, int],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    h, w = coarse_shape
    for r in range(h):
        for c in range(w):
            cell = (r, c)
            if cell in occupied or _is_boundary_cell(cell, coarse_shape):
                continue
            parents = [nbr for nbr in _coarse_neighbors(cell, coarse_shape) if nbr in occupied]
            if len(parents) == 1:
                candidates.append((cell, parents[0]))
    random.shuffle(candidates)
    return candidates


def _root_position(coarse_shape: tuple[int, int]) -> tuple[tuple[int, int], str]:
    h, w = coarse_shape
    side = random.choice(["up", "down", "left", "right"])
    if side == "up":
        return (0, random.randint(1, w - 2)), side
    if side == "down":
        return (h - 1, random.randint(1, w - 2)), side
    if side == "left":
        return (random.randint(1, h - 2), 0), side
    return (random.randint(1, h - 2), w - 1), side


def _grow_deep_tree(
    n_objects: int,
    coarse_shape: tuple[int, int],
    root: tuple[int, int],
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], tuple[int, int]], dict[tuple[int, int], int]] | None:
    occupied = {root}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    depths = {root: 0}
    tip = root

    for _ in range(n_objects - 1):
        options = [cell for cell, par in _candidate_cells(occupied, coarse_shape) if par == tip]
        if not options:
            return None
        child = random.choice(options)
        parent[child] = tip
        depths[child] = depths[tip] + 1
        occupied.add(child)
        tip = child

    order = [root]
    while len(order) < n_objects:
        order.append(next(pos for pos, par in parent.items() if par == order[-1]))
    return order, parent, depths


def _grow_branched_tree(
    n_objects: int,
    coarse_shape: tuple[int, int],
    root: tuple[int, int],
    trunk_depth: int,
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], tuple[int, int]], dict[tuple[int, int], int]] | None:
    occupied = {root}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    depths = {root: 0}
    order = [root]
    tip = root

    for _ in range(trunk_depth):
        options = [cell for cell, par in _candidate_cells(occupied, coarse_shape) if par == tip]
        if not options:
            return None
        child = random.choice(options)
        parent[child] = tip
        depths[child] = depths[tip] + 1
        occupied.add(child)
        order.append(child)
        tip = child

    while len(order) < n_objects:
        candidates = _candidate_cells(occupied, coarse_shape)
        if not candidates:
            return None
        depth_levels = sorted({depths[par] for _cell, par in candidates})
        cutoff_index = 0 if random.random() < 0.75 else min(1, len(depth_levels) - 1)
        cutoff = depth_levels[cutoff_index]
        pool = [(cell, par) for cell, par in candidates if depths[par] <= cutoff]
        child, par = random.choice(pool)
        parent[child] = par
        depths[child] = depths[par] + 1
        occupied.add(child)
        order.append(child)

    return order, parent, depths


def _tree_neighbors(
    order: list[tuple[int, int]],
    parent: dict[tuple[int, int], tuple[int, int]],
) -> dict[tuple[int, int], set[str]]:
    neighbors = {pos: set() for pos in order}
    for child, par in parent.items():
        dr = child[0] - par[0]
        dc = child[1] - par[1]
        side = SIDE_BY_DIR[(dr, dc)]
        opposite = SIDE_BY_DIR[(-dr, -dc)]
        neighbors[par].add(side)
        neighbors[child].add(opposite)
    return neighbors


def _port(side: str, tile_size: int) -> tuple[int, int]:
    mid = tile_size // 2
    if side == "up":
        return (0, mid)
    if side == "down":
        return (tile_size - 1, mid)
    if side == "left":
        return (mid, 0)
    return (mid, tile_size - 1)


def _path_to_port(center: tuple[int, int], port: tuple[int, int]) -> set[tuple[int, int]]:
    cr, cc = center
    pr, pc = port
    cells = set()
    if cr == pr:
        lo, hi = sorted((cc, pc))
        for c in range(lo, hi + 1):
            cells.add((cr, c))
    elif cc == pc:
        lo, hi = sorted((cr, pr))
        for r in range(lo, hi + 1):
            cells.add((r, cc))
    else:
        raise AssertionError("center and port must share a row or column")
    return cells


def _grow_shape(tile_size: int, required_sides: set[str], outward_side: str | None) -> np.ndarray:
    mid = tile_size // 2
    center = (mid, mid)
    cells = {center}

    if tile_size >= 5 and random.random() < 0.5:
        for dr, dc in ((0, -1), (-1, 0), (-1, -1)):
            rr = center[0] + dr
            cc = center[1] + dc
            if 0 <= rr < tile_size and 0 <= cc < tile_size:
                cells.add((rr, cc))

    for side in sorted(required_sides):
        cells |= _path_to_port(center, _port(side, tile_size))
    if outward_side is not None:
        cells |= _path_to_port(center, _port(outward_side, tile_size))

    target_area = random.randint(len(cells), min(tile_size * tile_size, len(cells) + tile_size + 4))
    while len(cells) < target_area:
        frontier = set()
        for r, c in cells:
            for dr, dc in DIRS4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < tile_size and 0 <= nc < tile_size and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(list(frontier)))

    mask = np.zeros((tile_size, tile_size), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _render_grid(
    order: list[tuple[int, int]],
    parent: dict[tuple[int, int], tuple[int, int]],
    coarse_shape: tuple[int, int],
    tile_size: int,
    root_side: str,
) -> np.ndarray:
    h, w = coarse_shape
    grid = np.full((h * tile_size, w * tile_size), BG, dtype=int)

    colors = random.sample(INPUT_COLORS, len(order))
    neighbors = _tree_neighbors(order, parent)
    root = order[0]

    for color, pos in zip(colors, order, strict=True):
        outward = root_side if pos == root else None
        mask = _grow_shape(tile_size, neighbors[pos], outward)
        r0 = pos[0] * tile_size
        c0 = pos[1] * tile_size
        grid[r0 : r0 + tile_size, c0 : c0 + tile_size][mask] = int(color)

    return grid


def _generate_candidate() -> tuple[np.ndarray, str]:
    mode = random.choices(
        ["deep", "branched", "hybrid"],
        weights=[3, 2, 2],
        k=1,
    )[0]
    if mode == "deep":
        n_objects = random.randint(5, OBJECT_COUNT_RANGE[1])
    else:
        n_objects = random.randint(*OBJECT_COUNT_RANGE)

    while True:
        coarse_h = random.randint(*COARSE_SIZE_RANGE)
        coarse_w = random.randint(*COARSE_SIZE_RANGE)
        if (coarse_h - 2) * (coarse_w - 2) >= n_objects - 1:
            break

    root, root_side = _root_position((coarse_h, coarse_w))

    if mode == "deep":
        grown = _grow_deep_tree(n_objects, (coarse_h, coarse_w), root)
    elif mode == "branched":
        trunk_depth = min(random.randint(3, 4), n_objects - 1)
        grown = _grow_branched_tree(n_objects, (coarse_h, coarse_w), root, trunk_depth)
    else:
        trunk_depth = min(random.randint(4, 5), n_objects - 1)
        grown = _grow_branched_tree(n_objects, (coarse_h, coarse_w), root, trunk_depth)

    if grown is None:
        raise RuntimeError("tree growth failed")

    order, parent, _depths = grown
    tile_size = random.choice(TILE_SIZE_CHOICES)
    grid = _render_grid(order, parent, (coarse_h, coarse_w), tile_size, root_side)
    return grid, mode


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            grid, mode = _generate_candidate()
            components, depths, _graph, _root = _analyze_tree(grid)
        except Exception:
            continue

        max_depth = max(depths.values())
        if mode == "deep" and max_depth < 4:
            continue
        if mode != "deep" and max_depth < 3:
            continue
        if len(components) < OBJECT_COUNT_RANGE[0]:
            continue
        main(grid)
        return grid

    raise RuntimeError("failed to generate a valid ARC instance")
