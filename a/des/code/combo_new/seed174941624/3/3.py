from __future__ import annotations

import random

import numpy as np


BG = 0
TRACK = 5
ROOT = 9
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8)
DIRS = [(-1, 0), (0, -1), (1, 0), (0, 1)]
DIR_PRIORITY = {direction: idx for idx, direction in enumerate(DIRS)}

COARSE_SIZE_RANGE = (9, 12)
TARGET_LEAF_RANGE = (3, 6)
TRUNK_LEN_RANGE = (2, 4)
SEGMENT_LEN_RANGE = (1, 3)
EXTRA_CHAIN_RANGE = (0, 3)
MAX_GENERATION_ATTEMPTS = 500
MAX_GROWTH_STEPS = 32

Cell = tuple[int, int]


def _neighbors(cell: Cell, shape: tuple[int, int]) -> list[Cell]:
    h, w = shape
    r, c = cell
    result: list[Cell] = []
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            result.append((nr, nc))
    return result


def _track_graph(grid: np.ndarray) -> tuple[Cell, list[Cell], dict[Cell, list[Cell]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    track_cells = {tuple(map(int, coord)) for coord in np.argwhere(g != BG)}
    assert track_cells, "expected a nonempty track"

    root_pixels = np.argwhere(g == ROOT)
    assert len(root_pixels) == 1, "expected exactly one maroon root marker"
    root = tuple(map(int, root_pixels[0]))

    seed_mask = (g != BG) & (g != TRACK) & (g != ROOT)
    seeds = [tuple(map(int, coord)) for coord in np.argwhere(seed_mask)]
    seed_colors = [int(g[cell]) for cell in seeds]
    assert len(seed_colors) == len(set(seed_colors)), "seed colors must be unique"
    assert seeds, "expected at least one colored seed"

    adj: dict[Cell, list[Cell]] = {}
    for cell in track_cells:
        adj[cell] = [nbr for nbr in _neighbors(cell, g.shape) if nbr in track_cells]

    assert len(adj[root]) == 1, "root marker must be a terminal track cell"
    for seed in seeds:
        assert len(adj[seed]) == 1, "each seed must be a leaf"

    degree_one = {cell for cell, nbrs in adj.items() if len(nbrs) == 1}
    assert degree_one == {root, *seeds}, "all terminal cells must be the root or a seed"

    seen: set[Cell] = set()
    stack = [root]
    while stack:
        cell = stack.pop()
        if cell in seen:
            continue
        seen.add(cell)
        for nbr in adj[cell]:
            if nbr not in seen:
                stack.append(nbr)

    assert seen == track_cells, "track must be connected"
    edge_count = sum(len(nbrs) for nbrs in adj.values()) // 2
    assert edge_count == len(track_cells) - 1, "track must be a tree"

    for cell, nbrs in adj.items():
        if cell in degree_one:
            continue
        value = int(g[cell])
        assert value == TRACK, "all nonterminal track cells must remain grey"
        assert 2 <= len(nbrs) <= 4, "internal track cells must have degree 2-4"

    return root, seeds, adj


def _sorted_children(node: Cell, parent: Cell | None, adj: dict[Cell, list[Cell]]) -> list[Cell]:
    children = [nbr for nbr in adj[node] if nbr != parent]
    return sorted(children, key=lambda nbr: DIR_PRIORITY[(nbr[0] - node[0], nbr[1] - node[1])])


def _leaf_order(root: Cell, seeds: list[Cell], adj: dict[Cell, list[Cell]]) -> list[Cell]:
    seed_set = set(seeds)
    order: list[Cell] = []

    def dfs(node: Cell, parent: Cell | None) -> None:
        if node in seed_set:
            order.append(node)
            return
        for child in _sorted_children(node, parent, adj):
            dfs(child, node)

    dfs(root, None)
    assert set(order) == seed_set and len(order) == len(seeds), "DFS must visit every seed once"
    return order


def _leaf_branch_cells(seed: Cell, root: Cell, adj: dict[Cell, list[Cell]]) -> list[Cell]:
    prev = seed
    cur = adj[seed][0]
    cells: list[Cell] = []
    while cur != root and len(adj[cur]) == 2:
        cells.append(cur)
        left, right = adj[cur]
        nxt = right if left == prev else left
        prev, cur = cur, nxt
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each seed color backward along its leaf track by the DFS rank of that leaf.

    Input structure:
    - The background is black.
    - Grey single-pixel-wide horizontal and vertical track segments form one connected tree.
    - A unique maroon terminal pixel marks the root end of the tree.
    - Every other terminal track end is a uniquely colored seed pixel sitting at a leaf.

    Rule:
    - Traverse the tree from the maroon root with depth-first search, always ordering outgoing directions as up, then left, then down, then right.
    - Number the colored leaf seeds by their DFS visitation order: 1, 2, 3, ...
    - For each seed, walk from the seed toward the tree until the first non-degree-2 node; recolor up to its rank many grey cells on that final leaf branch with the seed color.
    - Do not recolor the branch junction itself, and stop early if the branch is shorter than the rank.

    Key insight:
    - The seed colors do not encode their growth lengths directly; the lengths come from the hidden DFS order induced by the rooted track topology and the global up-then-left tie-break.
    """
    g = np.asarray(grid, dtype=int)
    root, seeds, adj = _track_graph(g)
    order = _leaf_order(root, seeds, adj)

    out = g.copy()
    for rank, seed in enumerate(order, start=1):
        color = int(g[seed])
        for cell in _leaf_branch_cells(seed, root, adj)[:rank]:
            out[cell] = color
    out[root] = ROOT
    return out


def _sample_root(h: int, w: int) -> tuple[Cell, Cell]:
    side = random.choice(("top", "bottom", "left", "right"))
    if side == "top":
        return (0, random.randint(1, w - 2)), (1, 0)
    if side == "bottom":
        return (h - 1, random.randint(1, w - 2)), (-1, 0)
    if side == "left":
        return (random.randint(1, h - 2), 0), (0, 1)
    return (random.randint(1, h - 2), w - 1), (0, -1)


def _max_straight_len(start: Cell, direction: Cell, h: int, w: int, occupied: set[Cell]) -> int:
    steps = 0
    r, c = start
    dr, dc = direction
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            break
        if (r, c) in occupied:
            break
        steps += 1
    return steps


def _add_straight_segment(
    adj: dict[Cell, set[Cell]],
    occupied: set[Cell],
    start: Cell,
    direction: Cell,
    length: int,
) -> Cell | None:
    cells: list[Cell] = []
    r, c = start
    dr, dc = direction
    for _ in range(length):
        r += dr
        c += dc
        cell = (r, c)
        if cell in occupied:
            return None
        cells.append(cell)

    prev = start
    for cell in cells:
        adj.setdefault(prev, set()).add(cell)
        adj.setdefault(cell, set()).add(prev)
        occupied.add(cell)
        prev = cell
    return prev


def _leaf_parent(adj: dict[Cell, set[Cell]], leaf: Cell) -> Cell | None:
    nbrs = adj[leaf]
    if not nbrs:
        return None
    assert len(nbrs) == 1
    return next(iter(nbrs))


def _extension_options(adj: dict[Cell, set[Cell]], occupied: set[Cell], leaf: Cell, h: int, w: int) -> list[tuple[Cell, int]]:
    parent = _leaf_parent(adj, leaf)
    blocked = None
    if parent is not None:
        blocked = (parent[0] - leaf[0], parent[1] - leaf[1])

    options: list[tuple[Cell, int]] = []
    for direction in DIRS:
        if direction == blocked:
            continue
        max_len = min(SEGMENT_LEN_RANGE[1], _max_straight_len(leaf, direction, h, w, occupied))
        if max_len >= SEGMENT_LEN_RANGE[0]:
            options.append((direction, max_len))
    return options


def _weighted_choice(options: list[int], scores: list[float]) -> int:
    total = sum(scores)
    pick = random.random() * total
    running = 0.0
    for option, score in zip(options, scores):
        running += score
        if pick <= running:
            return option
    return options[-1]


def _choose_branch_count(leaf_count: int, target_leaves: int, branch_nodes: int, max_children: int) -> int | None:
    need = target_leaves - leaf_count
    choices = [k for k in (1, 2, 3) if k <= max_children and leaf_count - 1 + k <= target_leaves]
    if not choices:
        return None
    if need <= 0 and branch_nodes >= 1:
        return 1 if 1 in choices else None
    if branch_nodes == 0:
        branching = [k for k in choices if k >= 2]
        if not branching:
            return None
        choices = branching

    scores: list[float] = []
    desired_inc = max(0, min(2, need))
    for k in choices:
        inc = k - 1
        score = 1.0
        if inc > 0:
            score += 2.0
        if inc == desired_inc:
            score += 2.0
        if k == 3:
            score += 0.5
        if need > 0 and inc == 0:
            score *= 0.25
        scores.append(score)
    return _weighted_choice(choices, scores)


def _grow_tree() -> dict[Cell, set[Cell]] | None:
    h = random.randint(*COARSE_SIZE_RANGE)
    w = random.randint(*COARSE_SIZE_RANGE)
    root, inward = _sample_root(h, w)
    occupied = {root}
    adj: dict[Cell, set[Cell]] = {root: set()}

    trunk_max = min(TRUNK_LEN_RANGE[1], _max_straight_len(root, inward, h, w, occupied))
    if trunk_max < TRUNK_LEN_RANGE[0]:
        return None
    trunk_len = random.randint(TRUNK_LEN_RANGE[0], trunk_max)
    tip = _add_straight_segment(adj, occupied, root, inward, trunk_len)
    if tip is None:
        return None

    target_leaves = random.randint(*TARGET_LEAF_RANGE)
    for _ in range(MAX_GROWTH_STEPS):
        leaves = [cell for cell, nbrs in adj.items() if len(nbrs) == 1 and cell != root]
        branch_nodes = sum(1 for nbrs in adj.values() if len(nbrs) >= 3)
        if len(leaves) == target_leaves and branch_nodes >= 1:
            break

        candidates = leaves[:]
        random.shuffle(candidates)
        expanded = False
        for leaf in candidates:
            options = _extension_options(adj, occupied, leaf, h, w)
            if not options:
                continue
            child_count = _choose_branch_count(len(leaves), target_leaves, branch_nodes, len(options))
            if child_count is None:
                continue

            directions = random.sample(options, child_count)
            endpoints: list[Cell | None] = []
            for direction, max_len in directions:
                seg_len = random.randint(SEGMENT_LEN_RANGE[0], max_len)
                endpoints.append(_add_straight_segment(adj, occupied, leaf, direction, seg_len))
            if all(endpoint is not None for endpoint in endpoints):
                expanded = True
                break
            return None

        if not expanded:
            return None

    leaves = [cell for cell, nbrs in adj.items() if len(nbrs) == 1 and cell != root]
    branch_nodes = sum(1 for nbrs in adj.values() if len(nbrs) >= 3)
    if len(leaves) != target_leaves or branch_nodes == 0:
        return None

    for _ in range(random.randint(*EXTRA_CHAIN_RANGE)):
        leaves = [cell for cell, nbrs in adj.items() if len(nbrs) == 1 and cell != root]
        random.shuffle(leaves)
        extended = False
        for leaf in leaves:
            options = _extension_options(adj, occupied, leaf, h, w)
            if not options:
                continue
            direction, max_len = random.choice(options)
            seg_len = random.randint(SEGMENT_LEN_RANGE[0], max_len)
            if _add_straight_segment(adj, occupied, leaf, direction, seg_len) is not None:
                extended = True
                break
        if not extended:
            break

    return adj


def _pixel(cell: Cell) -> Cell:
    return 2 * cell[0] + 1, 2 * cell[1] + 1


def _render_grid(adj: dict[Cell, set[Cell]]) -> np.ndarray:
    cells = list(adj)
    h = max(r for r, _c in cells) + 1
    w = max(c for _r, c in cells) + 1
    grid = np.full((2 * h + 1, 2 * w + 1), BG, dtype=int)

    for cell in cells:
        pr, pc = _pixel(cell)
        grid[pr, pc] = TRACK
        for nbr in adj[cell]:
            if cell > nbr:
                continue
            nr, nc = _pixel(nbr)
            grid[(pr + nr) // 2, (pc + nc) // 2] = TRACK

    roots = [cell for cell, nbrs in adj.items() if len(nbrs) == 1]
    root = random.choice(roots)
    leaves = [cell for cell, nbrs in adj.items() if len(nbrs) == 1 and cell != root]
    if not (TARGET_LEAF_RANGE[0] <= len(leaves) <= min(TARGET_LEAF_RANGE[1], len(SEED_COLORS))):
        raise AssertionError("leaf count must fit the seed palette")

    grid[_pixel(root)] = ROOT
    seed_colors = random.sample(SEED_COLORS, len(leaves))
    random.shuffle(leaves)
    for leaf, color in zip(leaves, seed_colors):
        grid[_pixel(leaf)] = color

    return grid


def _valid_growth_profile(grid: np.ndarray) -> bool:
    root, seeds, adj = _track_graph(grid)
    order = _leaf_order(root, seeds, adj)
    lengths = [len(_leaf_branch_cells(seed, root, adj)) for seed in order]
    if not lengths:
        return False
    has_partial = any(length < rank for rank, length in enumerate(lengths, start=1))
    has_full = any(length >= rank for rank, length in enumerate(lengths, start=1))
    return has_partial and has_full


def _gen_attempt() -> np.ndarray | None:
    coarse = _grow_tree()
    if coarse is None:
        return None

    try:
        grid = _render_grid(coarse)
        if not _valid_growth_profile(grid):
            return None
        out = main(grid)
    except Exception:
        return None

    if grid.shape[0] > 30 or grid.shape[1] > 30:
        return None
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid track-growth ARC input")
