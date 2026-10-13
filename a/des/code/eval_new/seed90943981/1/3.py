from __future__ import annotations

from collections import deque
from functools import lru_cache
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAROON = 9

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(pos: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = pos
    h, w = shape
    out = []
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _graph_from_mask(mask: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    h, w = mask.shape
    graph: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in np.argwhere(mask):
        node = (int(r), int(c))
        graph[node] = [nbr for nbr in _neighbors4(node, (h, w)) if bool(mask[nbr])]
    return graph


def _rooted_component(
    graph: dict[tuple[int, int], list[tuple[int, int]]], root: tuple[int, int]
) -> tuple[dict[tuple[int, int], tuple[int, int] | None], dict[tuple[int, int], list[tuple[int, int]]]]:
    parent: dict[tuple[int, int], tuple[int, int] | None] = {root: None}
    order = [root]
    q = deque([root])
    while q:
        node = q.popleft()
        for nbr in graph.get(node, []):
            if nbr in parent:
                continue
            parent[nbr] = node
            order.append(nbr)
            q.append(nbr)

    children = {node: [] for node in order}
    for node, par in parent.items():
        if par is not None:
            children[par].append(node)
    return parent, children


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route a signal through a grey wire tree and light only the terminals reached under red/green splitter rules.

    Input structure:
    - Background is black.
    - A single connected wire network is drawn with one-pixel-wide horizontal and vertical segments.
    - Most wire pixels are grey; the root endpoint is a blue emitter and the leaf endpoints are maroon terminal lamps.
    - Some degree-3 T-junction pixels are colored red or green instead of grey.

    Rule:
    - Start the signal at the blue emitter and follow the wire tree away from the root.
    - At a green splitter, continue into both child branches.
    - At a red splitter, compare the two child branches by the number of grey pixels from the splitter to the nearest terminal in each branch; take only the shorter branch, and take neither if the counts tie.
    - Every reached maroon terminal becomes yellow; all wires, splitters, and untouched terminals stay unchanged.

    Key insight:
    - The task is topological: first orient the wire tree from the blue root, then treat each red or green degree-3 junction as a routing rule, with red decisions based only on visible branch length measured by grey wire pixels.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    roots = np.argwhere(g == BLUE)
    if roots.size == 0:
        return out

    root = tuple(int(v) for v in roots[0])
    mask = g != BLACK
    graph = _graph_from_mask(mask)
    parent, children = _rooted_component(graph, root)

    color_of = {node: int(g[node]) for node in parent}

    @lru_cache(maxsize=None)
    def grey_distance(node: tuple[int, int]) -> int:
        color = color_of[node]
        if color == MAROON and not children[node]:
            return 0
        if not children[node]:
            return 10**9
        own = 1 if color == GREY else 0
        return own + min(grey_distance(child) for child in children[node])

    @lru_cache(maxsize=None)
    def reached_terminals(node: tuple[int, int]) -> frozenset[tuple[int, int]]:
        color = color_of[node]
        kids = children[node]

        if color == MAROON:
            return frozenset({node}) if not kids else frozenset()
        if not kids:
            return frozenset()

        if color == RED and len(kids) == 2:
            d0 = grey_distance(kids[0])
            d1 = grey_distance(kids[1])
            if d0 == d1:
                return frozenset()
            return reached_terminals(kids[0] if d0 < d1 else kids[1])

        if color == GREEN:
            lit: set[tuple[int, int]] = set()
            for child in kids:
                lit.update(reached_terminals(child))
            return frozenset(lit)

        if len(kids) == 1:
            return reached_terminals(kids[0])

        lit = set()
        for child in kids:
            lit.update(reached_terminals(child))
        return frozenset(lit)

    for node in reached_terminals(root):
        if int(out[node]) == MAROON:
            out[node] = YELLOW

    return out


def _occupied_graph(cells: set[tuple[int, int]], shape: tuple[int, int]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    return {cell: [nbr for nbr in _neighbors4(cell, shape) if nbr in cells] for cell in cells}


def _tree_info(
    cells: set[tuple[int, int]], root: tuple[int, int], shape: tuple[int, int]
) -> tuple[
    dict[tuple[int, int], list[tuple[int, int]]],
    dict[tuple[int, int], tuple[int, int] | None],
    dict[tuple[int, int], list[tuple[int, int]]],
]:
    graph = _occupied_graph(cells, shape)
    parent, children = _rooted_component(graph, root)
    return graph, parent, children


def _straight_degree2(node: tuple[int, int], graph: dict[tuple[int, int], list[tuple[int, int]]]) -> bool:
    nbrs = graph[node]
    if len(nbrs) != 2:
        return False
    return nbrs[0][0] == nbrs[1][0] or nbrs[0][1] == nbrs[1][1]


def _branch_dirs(node: tuple[int, int], graph: dict[tuple[int, int], list[tuple[int, int]]]) -> list[tuple[int, int]]:
    nbrs = graph[node]
    if nbrs[0][0] == nbrs[1][0]:
        return [(-1, 0), (1, 0)]
    return [(0, -1), (0, 1)]


def _max_ray_length(
    cells: set[tuple[int, int]],
    start: tuple[int, int],
    direction: tuple[int, int],
    shape: tuple[int, int],
) -> int:
    h, w = shape
    prev = start
    r, c = start
    length = 0
    while True:
        r += direction[0]
        c += direction[1]
        node = (r, c)
        if not (0 <= r < h and 0 <= c < w):
            return length
        if node in cells:
            return length
        occupied_nbrs = {nbr for nbr in _neighbors4(node, shape) if nbr in cells}
        if occupied_nbrs - {prev}:
            return length
        prev = node
        length += 1


def _add_segment(
    cells: set[tuple[int, int]], start: tuple[int, int], direction: tuple[int, int], length: int
) -> tuple[int, int]:
    r, c = start
    for _ in range(length):
        r += direction[0]
        c += direction[1]
        cells.add((r, c))
    return (r, c)


def _gen_trunk(rng: random.Random, shape: tuple[int, int]) -> tuple[set[tuple[int, int]], tuple[int, int]] | None:
    h, w = shape
    if h < 12 or w < 12:
        return None

    for _ in range(120):
        root = (rng.randrange(3, h - 3), rng.randrange(3, w - 3))
        cells = {root}
        current = root
        last_dir: tuple[int, int] | None = None
        seg_count = rng.randint(3, 5)
        ok = True

        for _seg in range(seg_count):
            if last_dir is None:
                dirs = DIRS4[:]
                rng.shuffle(dirs)
            else:
                reverse = (-last_dir[0], -last_dir[1])
                perp = [d for d in DIRS4 if d not in (last_dir, reverse)]
                dirs = [last_dir] + perp if rng.random() < 0.45 else perp + [last_dir]
                if rng.random() < 0.5:
                    rng.shuffle(perp)
                    dirs = [last_dir] + perp
                else:
                    rng.shuffle(perp)
                    dirs = perp + [last_dir]

            success = False
            lengths = list(range(3, 7))
            rng.shuffle(lengths)
            for direction in dirs:
                for length in lengths:
                    if _max_ray_length(cells, current, direction, shape) < length:
                        continue
                    current = _add_segment(cells, current, direction, length)
                    last_dir = direction
                    success = True
                    break
                if success:
                    break
            if not success:
                ok = False
                break

        if not ok:
            continue

        graph = _occupied_graph(cells, shape)
        if len(graph[root]) != 1:
            continue
        if len(cells) < 12:
            continue
        if sum(len(nbrs) == 1 for nbrs in graph.values()) < 2:
            continue
        if sum(_straight_degree2(node, graph) for node in graph) < 4:
            continue
        return cells, root

    return None


def _add_random_green_branch(
    cells: set[tuple[int, int]], root: tuple[int, int], shape: tuple[int, int], rng: random.Random
) -> tuple[int, int] | None:
    graph, _parent, children = _tree_info(cells, root, shape)
    options: list[tuple[tuple[int, int], tuple[int, int], int]] = []

    for node in cells:
        if node == root or not _straight_degree2(node, graph):
            continue
        if len(children[node]) != 1:
            continue
        for direction in _branch_dirs(node, graph):
            room = _max_ray_length(cells, node, direction, shape)
            if room >= 2:
                options.append((node, direction, room))

    if not options:
        return None

    rng.shuffle(options)
    for node, direction, room in options:
        upper = min(room, 6)
        if upper < 2:
            continue
        length = rng.randint(2, upper)
        _add_segment(cells, node, direction, length)
        return node
    return None


def _simple_chain_distance(
    start: tuple[int, int],
    graph: dict[tuple[int, int], list[tuple[int, int]]],
    children: dict[tuple[int, int], list[tuple[int, int]]],
) -> int | None:
    distance = 0
    node = start
    while True:
        degree = len(graph[node])
        if degree == 1:
            return distance
        if degree != 2 or len(children[node]) != 1:
            return None
        distance += 1
        node = children[node][0]


def _add_special_red_branch(
    cells: set[tuple[int, int]],
    root: tuple[int, int],
    shape: tuple[int, int],
    rng: random.Random,
    mode: str,
) -> tuple[int, int] | None:
    graph, _parent, children = _tree_info(cells, root, shape)
    options: list[tuple[tuple[int, int], tuple[int, int], int]] = []

    for node in cells:
        if node == root or not _straight_degree2(node, graph):
            continue
        if len(children[node]) != 1:
            continue
        base_dist = _simple_chain_distance(children[node][0], graph, children)
        if base_dist is None or base_dist < 1:
            continue

        for direction in _branch_dirs(node, graph):
            room = _max_ray_length(cells, node, direction, shape)
            if room < 2:
                continue

            if mode == "tie":
                length = base_dist + 1
                if 2 <= length <= room:
                    options.append((node, direction, length))
            elif mode == "new_shorter":
                if base_dist < 2:
                    continue
                max_len = min(room, base_dist)
                if max_len >= 2:
                    length = rng.randint(2, max_len)
                    options.append((node, direction, length))
            else:
                min_len = base_dist + 2
                max_len = min(room, base_dist + 4)
                if min_len <= max_len:
                    length = rng.randint(min_len, max_len)
                    options.append((node, direction, length))

    if not options:
        return None

    node, direction, length = rng.choice(options)
    _add_segment(cells, node, direction, length)
    return node


def _paint_grid(
    cells: set[tuple[int, int]],
    root: tuple[int, int],
    shape: tuple[int, int],
    red_splitter: tuple[int, int],
) -> np.ndarray:
    graph = _occupied_graph(cells, shape)
    grid = np.full(shape, BLACK, dtype=int)

    for node in cells:
        degree = len(graph[node])
        if node == root:
            grid[node] = BLUE
        elif degree == 1:
            grid[node] = MAROON
        elif degree == 3:
            grid[node] = RED if node == red_splitter else GREEN
        else:
            grid[node] = GREY

    return grid


def _valid_signal_result(inp: np.ndarray) -> bool:
    out = main(inp)
    lit = int(np.count_nonzero(out == YELLOW))
    unlit = int(np.count_nonzero(out == MAROON))
    return lit >= 1 and unlit >= 1


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    shape = (rng.randint(16, 28), rng.randint(16, 28))
    trunk = _gen_trunk(rng, shape)
    if trunk is None:
        return None

    cells, root = trunk
    split_target = rng.randint(2, 4)
    for _ in range(split_target - 1):
        if _add_random_green_branch(cells, root, shape, rng) is None:
            return None

    mode = rng.choice(["tie", "new_shorter", "new_longer"])
    red_splitter = _add_special_red_branch(cells, root, shape, rng, mode)
    if red_splitter is None:
        return None

    graph = _occupied_graph(cells, shape)
    if sum(len(nbrs) == 3 for nbrs in graph.values()) < 2:
        return None
    if sum(len(nbrs) == 1 for node, nbrs in graph.items() if node != root) < 3:
        return None
    if sum(len(nbrs) for nbrs in graph.values()) != 2 * (len(cells) - 1):
        return None

    grid = _paint_grid(cells, root, shape, red_splitter)
    if not _valid_signal_result(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-routing input")
