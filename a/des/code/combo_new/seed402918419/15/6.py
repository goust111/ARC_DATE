import random

import numpy as np


BLACK = 0
GREY = 5
SEED_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)


SPACING = 4
COARSE_ROWS_RANGE = (3, 7)
COARSE_COLS_RANGE = (3, 7)
COARSE_NODE_COUNT_RANGE = (6, 10)
BRANCHY_PROB = 0.65
MAX_LEAF_COUNT = 6
GEN_RETRIES = 400
TREE_RETRIES = 120


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _pipe_degrees(pipe_mask: np.ndarray) -> np.ndarray:
    pipe = pipe_mask.astype(int)
    deg = np.zeros(pipe_mask.shape, dtype=int)
    deg[1:, :] += pipe[:-1, :]
    deg[:-1, :] += pipe[1:, :]
    deg[:, 1:] += pipe[:, :-1]
    deg[:, :-1] += pipe[:, 1:]
    return deg


def _find_sources(grid: np.ndarray, pipe_mask: np.ndarray, degrees: np.ndarray) -> tuple[dict[tuple[int, int], int], list[tuple[int, int]]]:
    h, w = grid.shape
    sources: dict[tuple[int, int], int] = {}
    seed_cells: list[tuple[int, int]] = []

    for r, c in np.argwhere((grid != BLACK) & (grid != GREY)):
        rr = int(r)
        cc = int(c)
        endpoints = [
            (nr, nc)
            for nr, nc in _neighbors4(rr, cc, h, w)
            if pipe_mask[nr, nc] and int(degrees[nr, nc]) == 1
        ]
        if len(endpoints) == 1:
            sources[endpoints[0]] = int(grid[rr, cc])
            seed_cells.append((rr, cc))

    return sources, seed_cells


def _simulate_spread(pipe_mask: np.ndarray, sources: dict[tuple[int, int], int]) -> tuple[np.ndarray, np.ndarray]:
    owners = np.full(pipe_mask.shape, -1, dtype=int)
    boundary = np.zeros(pipe_mask.shape, dtype=bool)
    frontier: list[tuple[int, int]] = []
    h, w = pipe_mask.shape

    for (r, c), color in sources.items():
        owners[r, c] = int(color)
        frontier.append((r, c))

    while frontier:
        proposals: dict[tuple[int, int], set[int]] = {}
        for r, c in frontier:
            color = int(owners[r, c])
            for nr, nc in _neighbors4(r, c, h, w):
                if not pipe_mask[nr, nc]:
                    continue
                if boundary[nr, nc] or owners[nr, nc] != -1:
                    continue
                proposals.setdefault((nr, nc), set()).add(color)

        if not proposals:
            break

        next_frontier: list[tuple[int, int]] = []
        for cell, colors in proposals.items():
            if len(colors) == 1:
                color = next(iter(colors))
                owners[cell] = int(color)
                next_frontier.append(cell)
            else:
                boundary[cell] = True
        frontier = next_frontier

    return owners, boundary


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simultaneously flood-color a gray pipe network from colored endpoint seeds.

    Input structure:
    - The background is black.
    - A single connected gray pipe network is drawn with 1-pixel-wide orthogonal paths.
    - Some degree-1 pipe endpoints have a distinct colored seed pixel in a 4-neighbor cell beside them.
    - Not every pipe endpoint is necessarily seeded.

    Rule:
    - Remove the seed pixels from the output.
    - Treat each seeded pipe endpoint as an initial colored frontier and expand all frontiers synchronously along gray pipe cells.
    - If exactly one color reaches a gray pipe cell in a step, that cell becomes that color; if multiple colors reach it simultaneously, it stays gray as a permanent boundary.
    - Continue until no frontier can expand further, leaving all untouched pipe cells gray.

    Key insight:
    - The final partition is produced by explicit step-by-step simultaneous propagation on the pipe graph, so gray boundaries appear exactly where wavefronts tie.
    """
    g = np.asarray(grid, dtype=int)
    pipe_mask = g == GREY
    degrees = _pipe_degrees(pipe_mask)
    sources, _seed_cells = _find_sources(g, pipe_mask, degrees)
    owners, boundary = _simulate_spread(pipe_mask, sources)

    out = np.full(g.shape, BLACK, dtype=int)
    out[pipe_mask] = GREY

    claimed = pipe_mask & (owners != -1)
    out[claimed] = owners[claimed]
    out[boundary] = GREY
    return out


def _coarse_neighbors(node: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = node
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < rows:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < cols:
        out.append((r, c + 1))
    return out


def _coarse_degrees(nodes: set[tuple[int, int]], edges: list[tuple[tuple[int, int], tuple[int, int]]]) -> dict[tuple[int, int], int]:
    deg = {node: 0 for node in nodes}
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    return deg


def _build_coarse_tree(rows: int, cols: int, need_branch: bool) -> tuple[set[tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int]]]] | None:
    max_nodes = min(COARSE_NODE_COUNT_RANGE[1], rows * cols)
    if max_nodes < COARSE_NODE_COUNT_RANGE[0]:
        return None

    for _ in range(TREE_RETRIES):
        target_nodes = random.randint(COARSE_NODE_COUNT_RANGE[0], max_nodes)
        start = (random.randrange(rows), random.randrange(cols))
        nodes = {start}
        edges: list[tuple[tuple[int, int], tuple[int, int]]] = []

        while len(nodes) < target_nodes:
            degrees = _coarse_degrees(nodes, edges)
            options: list[tuple[tuple[int, int], tuple[int, int]]] = []
            for node in nodes:
                weight = 3 if degrees[node] <= 1 else 1
                for nb in _coarse_neighbors(node, rows, cols):
                    if nb not in nodes:
                        options.extend([(node, nb)] * weight)
            if not options:
                break
            parent, child = random.choice(options)
            nodes.add(child)
            edges.append((parent, child))

        if len(nodes) != target_nodes:
            continue

        degrees = _coarse_degrees(nodes, edges)
        leaves = [node for node, degree in degrees.items() if degree == 1]
        if not (2 <= len(leaves) <= MAX_LEAF_COUNT):
            continue

        row_span = max(r for r, _ in nodes) - min(r for r, _ in nodes)
        col_span = max(c for _, c in nodes) - min(c for _, c in nodes)
        if row_span < 1 or col_span < 1:
            continue

        if need_branch and (max(degrees.values()) < 3 or len(leaves) < 3):
            continue

        return nodes, edges

    return None


def _node_to_cell(node: tuple[int, int]) -> tuple[int, int]:
    return 1 + SPACING * node[0], 1 + SPACING * node[1]


def _rasterize_tree(rows: int, cols: int, nodes: set[tuple[int, int]], edges: list[tuple[tuple[int, int], tuple[int, int]]]) -> np.ndarray:
    h = SPACING * rows + 1
    w = SPACING * cols + 1
    grid = np.full((h, w), BLACK, dtype=int)

    for node in nodes:
        r, c = _node_to_cell(node)
        grid[r, c] = GREY

    for a, b in edges:
        r0, c0 = _node_to_cell(a)
        r1, c1 = _node_to_cell(b)
        if r0 == r1:
            lo, hi = sorted((c0, c1))
            grid[r0, lo : hi + 1] = GREY
        else:
            lo, hi = sorted((r0, r1))
            grid[lo : hi + 1, c0] = GREY

    return grid


def _seed_position(endpoint: tuple[int, int], pipe_mask: np.ndarray) -> tuple[int, int] | None:
    h, w = pipe_mask.shape
    r, c = endpoint
    pipe_neighbors = [(nr, nc) for nr, nc in _neighbors4(r, c, h, w) if pipe_mask[nr, nc]]
    if len(pipe_neighbors) != 1:
        return None

    nr, nc = pipe_neighbors[0]
    dr = nr - r
    dc = nc - c
    primary = (r - dr, c - dc)
    fallbacks = [primary] + [
        pos
        for pos in _neighbors4(r, c, h, w)
        if pos != (nr, nc)
    ]

    for rr, cc in fallbacks:
        if 0 <= rr < h and 0 <= cc < w and not pipe_mask[rr, cc]:
            return rr, cc
    return None


def _place_seeds(grid: np.ndarray, seed_all: bool) -> np.ndarray | None:
    pipe_mask = grid == GREY
    degrees = _pipe_degrees(pipe_mask)
    endpoints = [tuple(map(int, rc)) for rc in np.argwhere(pipe_mask & (degrees == 1))]
    if not (2 <= len(endpoints) <= len(SEED_PALETTE)):
        return None

    if len(endpoints) == 2 or seed_all:
        seed_count = len(endpoints)
    else:
        seed_count = random.randint(2, len(endpoints) - 1)

    chosen_endpoints = random.sample(endpoints, seed_count)
    seed_colors = random.sample(SEED_PALETTE, seed_count)
    used_positions: set[tuple[int, int]] = set()

    for endpoint, color in zip(chosen_endpoints, seed_colors):
        pos = _seed_position(endpoint, pipe_mask)
        if pos is None or pos in used_positions or int(grid[pos]) != BLACK:
            return None
        grid[pos] = int(color)
        used_positions.add(pos)

    return grid


def _oracle_valid(grid: np.ndarray) -> bool:
    pipe_mask = grid == GREY
    degrees = _pipe_degrees(pipe_mask)
    seed_mask = (grid != BLACK) & (grid != GREY)
    seed_positions = [tuple(map(int, rc)) for rc in np.argwhere(seed_mask)]
    if len(seed_positions) < 2:
        return False

    for r, c in seed_positions:
        attached = [
            (nr, nc)
            for nr, nc in _neighbors4(r, c, *grid.shape)
            if pipe_mask[nr, nc] and int(degrees[nr, nc]) == 1
        ]
        if len(attached) != 1:
            return False

    input_seed_colors = {int(v) for v in grid[seed_mask]}
    if len(input_seed_colors) != len(seed_positions):
        return False

    try:
        out = main(grid)
    except Exception:
        return False

    if out.shape != grid.shape or np.array_equal(out, grid):
        return False
    if np.any(out[seed_mask] != BLACK):
        return False

    colored_pipe = (out != BLACK) & (out != GREY)
    if np.any(colored_pipe & (~pipe_mask)):
        return False

    output_colors = {int(v) for v in out[colored_pipe]}
    if output_colors != input_seed_colors:
        return False

    pipe_count = int(np.count_nonzero(pipe_mask))
    boundary_count = int(np.count_nonzero(pipe_mask & (out == GREY)))
    colored_count = int(np.count_nonzero(colored_pipe))
    if pipe_count < 21 or boundary_count < 1:
        return False
    if colored_count < 8 or colored_count <= boundary_count:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*COARSE_ROWS_RANGE)
    cols = random.randint(*COARSE_COLS_RANGE)
    need_branch = random.random() < BRANCHY_PROB
    tree = _build_coarse_tree(rows, cols, need_branch)
    if tree is None:
        return None

    nodes, edges = tree
    grid = _rasterize_tree(rows, cols, nodes, edges)
    seed_all = random.random() < 0.5
    grid = _place_seeds(grid, seed_all)
    if grid is None:
        return None

    if not _oracle_valid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC instance")
