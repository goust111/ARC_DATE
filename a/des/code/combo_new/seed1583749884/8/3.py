import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GRAY = 5
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

LATTICE_MIN = 5
LATTICE_MAX = 14
MAX_TREE_NODES = 26
MAX_GENERATION_ATTEMPTS = 400


def _degree_map(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(int), 1)
    return (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    )


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _infer_path_color(grid: np.ndarray, bg: int) -> int:
    colors = [int(color) for color in np.unique(grid) if int(color) != bg]
    assert colors, "expected at least one path color"
    if GRAY in colors:
        return GRAY
    if len(colors) == 1:
        return int(colors[0])
    counts = {color: int(np.count_nonzero(grid == color)) for color in colors}
    return int(max(colors, key=lambda color: counts[color]))


def _component_size(mask: np.ndarray) -> int:
    start = np.argwhere(mask)
    if start.size == 0:
        return 0
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    sr, sc = map(int, start[0])
    queue = deque([(sr, sc)])
    seen[sr, sc] = True
    size = 0
    while queue:
        r, c = queue.popleft()
        size += 1
        for dr, dc in DIR4:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                queue.append((rr, cc))
    return size


def main(grid: np.ndarray) -> np.ndarray:
    """Mark dead ends red and branch points blue in a thin orthogonal maze path.

    Input structure:
    - The background is black.
    - Grey pixels form a one-pixel-wide orthogonal path network.
    - Path pixels are 4-connected and may include dead ends, turns, corridors, and branch points.

    Rule:
    - For each grey path pixel, count how many of its 4-neighbors are also path pixels.
    - Recolor degree-1 path pixels red.
    - Recolor degree-3 and degree-4 path pixels blue.
    - Keep degree-2 path pixels grey and leave the background unchanged.

    Key insight:
    - The output class of every path pixel is determined only by its local 4-neighbor degree.
    """
    g = np.asarray(grid, dtype=int)
    bg = BLACK if np.any(g == BLACK) else _most_common_color(g)
    path_color = _infer_path_color(g, bg)

    path_mask = g == path_color
    neighbor_counts = _degree_map(path_mask)

    out = g.copy()
    out[path_mask & (neighbor_counts == 1)] = RED
    out[path_mask & (neighbor_counts >= 3)] = BLUE
    return out


def _free_neighbors(node: tuple[int, int], nodes: set[tuple[int, int]], lh: int, lw: int) -> list[tuple[int, int]]:
    r, c = node
    free = []
    for dr, dc in DIR4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < lh and 0 <= cc < lw and (rr, cc) not in nodes:
            free.append((rr, cc))
    return free


def _weighted_choice(items: list[tuple[int, int]], weights: list[int]) -> tuple[int, int]:
    total = sum(weights)
    pick = random.randint(1, total)
    running = 0
    for item, weight in zip(items, weights, strict=True):
        running += weight
        if pick <= running:
            return item
    return items[-1]


def _grow_tree(lh: int, lw: int) -> tuple[set[tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int]]]]:
    root = (random.randint(1, lh - 2), random.randint(1, lw - 2))
    root_degree = 4 if random.random() < 0.5 else 3

    all_neighbors = _free_neighbors(root, set(), lh, lw)
    chosen_neighbors = random.sample(all_neighbors, root_degree)

    nodes = {root, *chosen_neighbors}
    edges = [(root, nbr) for nbr in chosen_neighbors]
    degrees = {root: root_degree}
    for nbr in chosen_neighbors:
        degrees[nbr] = 1

    area = lh * lw
    min_nodes = root_degree + 4
    max_nodes = min(MAX_TREE_NODES, max(min_nodes, area // 3 + 4))
    target_nodes = random.randint(min_nodes, max_nodes)

    steps_left = target_nodes * 10
    while len(nodes) < target_nodes and steps_left > 0:
        candidates = []
        weights = []
        for node in nodes:
            free = _free_neighbors(node, nodes, lh, lw)
            if not free:
                continue
            deg = degrees[node]
            weight = 1 + len(free)
            if deg <= 1:
                weight += 3
            elif deg == 2:
                weight += 1
            candidates.append(node)
            weights.append(weight)
        if not candidates:
            break

        parent = _weighted_choice(candidates, weights)
        child = random.choice(_free_neighbors(parent, nodes, lh, lw))
        nodes.add(child)
        edges.append((parent, child))
        degrees[parent] += 1
        degrees[child] = 1
        steps_left -= 1

    return nodes, edges


def _rasterize_tree(
    lh: int,
    lw: int,
    nodes: set[tuple[int, int]],
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
) -> np.ndarray:
    grid = np.full((2 * lh + 1, 2 * lw + 1), BLACK, dtype=int)
    for r, c in nodes:
        grid[2 * r + 1, 2 * c + 1] = GRAY
    for (r0, c0), (r1, c1) in edges:
        grid[r0 + r1 + 1, c0 + c1 + 1] = GRAY
    return grid


def _is_valid_input(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if max(grid.shape) > 30:
        return False
    if set(map(int, np.unique(grid))) - {BLACK, GRAY}:
        return False

    mask = grid == GRAY
    area = int(np.count_nonzero(mask))
    if area < 9:
        return False
    if _component_size(mask) != area:
        return False

    degree = _degree_map(mask)
    deg1 = int(np.count_nonzero(mask & (degree == 1)))
    deg2 = int(np.count_nonzero(mask & (degree == 2)))
    deg34 = int(np.count_nonzero(mask & (degree >= 3)))
    if deg1 < 2 or deg2 < 1 or deg34 < 1:
        return False

    if not np.any(mask[1:-1, 1:-1]):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    lh = random.randint(LATTICE_MIN, LATTICE_MAX)
    lw = random.randint(LATTICE_MIN, LATTICE_MAX)
    nodes, edges = _grow_tree(lh, lw)
    grid = _rasterize_tree(lh, lw, nodes, edges)
    return grid if _is_valid_input(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid maze-degree input")
