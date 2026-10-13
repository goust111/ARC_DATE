import heapq
import random
from collections import defaultdict

import numpy as np


BLACK = 0
GREY = 5
INF = 10**9

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
PERPENDICULAR = {
    (-1, 0): [(0, -1), (0, 1)],
    (1, 0): [(0, -1), (0, 1)],
    (0, -1): [(-1, 0), (1, 0)],
    (0, 1): [(-1, 0), (1, 0)],
}
COLOR_NAMES = {
    0: "black",
    1: "blue",
    2: "red",
    3: "green",
    4: "yellow",
    5: "gray",
    6: "magenta",
    7: "orange",
    8: "cyan",
    9: "brown",
}

SEED_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
VARIANTS = ("path", "t", "cross", "mixed")
SCALE_OPTIONS = (2, 3)
COARSE_ROWS_RANGE = (6, 8)
COARSE_COLS_RANGE = (6, 8)
GEN_RETRIES = 300


def _color_name(color: int) -> str:
    return COLOR_NAMES.get(int(color), f"color{int(color)}")


def _pipe_neighbors(mask: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    h, w = mask.shape
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]):
            out.append((nr, nc))
    return out


def _next_cells(
    adjacency: dict[tuple[int, int], tuple[tuple[int, int], ...]],
    prev_cell: tuple[int, int],
    cur_cell: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    neighbors = adjacency.get(cur_cell, ())
    degree = len(neighbors)
    if degree <= 1:
        return ()
    if degree == 2:
        return tuple(nb for nb in neighbors if nb != prev_cell)

    dr = cur_cell[0] - prev_cell[0]
    dc = cur_cell[1] - prev_cell[1]
    forward = (cur_cell[0] + dr, cur_cell[1] + dc)

    if degree == 3:
        return (forward,) if forward in neighbors else ()
    return tuple(nb for nb in neighbors if nb != prev_cell)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    pipe_mask = g != BLACK
    if not bool(pipe_mask.any()):
        return g.copy()

    adjacency: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {}
    for r, c in np.argwhere(pipe_mask):
        rr = int(r)
        cc = int(c)
        adjacency[(rr, cc)] = tuple(_pipe_neighbors(pipe_mask, rr, cc))

    seed_positions: list[tuple[int, int]] = []
    seed_mask = np.zeros((h, w), dtype=bool)
    colored = np.argwhere((g != BLACK) & (g != GREY))
    for r, c in colored:
        rr = int(r)
        cc = int(c)
        if len(adjacency.get((rr, cc), ())) == 1:
            seed_positions.append((rr, cc))
            seed_mask[rr, cc] = True

    if not seed_positions:
        return g.copy()

    out = g.copy()
    owner_color = np.full((h, w), -1, dtype=int)
    owner_time = np.full((h, w), INF, dtype=int)
    owner_dist = np.full((h, w), INF, dtype=int)

    for r, c in seed_positions:
        owner_color[r, c] = int(g[r, c])
        owner_time[r, c] = 0
        owner_dist[r, c] = 0

    pq: list[tuple[int, int, int, int, int, int, int]] = []
    best_state_time: dict[tuple[int, tuple[int, int], tuple[int, int]], int] = {}

    def enqueue(
        color: int,
        prev_cell: tuple[int, int],
        cur_cell: tuple[int, int],
        dist: int,
    ) -> None:
        key = (int(color), prev_cell, cur_cell)
        old = best_state_time.get(key, INF)
        if dist >= old:
            return
        best_state_time[key] = dist
        heapq.heappush(
            pq,
            (dist, dist, int(color), prev_cell[0], prev_cell[1], cur_cell[0], cur_cell[1]),
        )

    for r, c in seed_positions:
        neighbors = adjacency.get((r, c), ())
        if len(neighbors) == 1:
            enqueue(int(g[r, c]), (r, c), neighbors[0], 1)

    while pq:
        cur_time = int(pq[0][0])
        batch: list[tuple[int, int, int, int, int, int, int]] = []
        while pq and int(pq[0][0]) == cur_time:
            batch.append(heapq.heappop(pq))

        by_cell: dict[tuple[int, int], list[tuple[int, int, tuple[int, int]]]] = defaultdict(list)
        for dist, _time, color, pr, pc, r, c in batch:
            by_cell[(int(r), int(c))].append((int(dist), int(color), (int(pr), int(pc))))

        for (r, c), arrivals in by_cell.items():
            claimed_color = int(owner_color[r, c])
            if claimed_color == -1:
                winner_dist, winner_color, _ = min(
                    arrivals, key=lambda item: (item[0], _color_name(item[1]))
                )
                owner_color[r, c] = int(winner_color)
                owner_time[r, c] = int(cur_time)
                owner_dist[r, c] = int(winner_dist)
                if not bool(seed_mask[r, c]):
                    out[r, c] = int(winner_color)
                allowed = [item for item in arrivals if int(item[1]) == int(winner_color)]
            elif int(owner_time[r, c]) == int(cur_time):
                winner_color = int(claimed_color)
                allowed = [item for item in arrivals if int(item[1]) == int(winner_color)]
            elif int(owner_color[r, c]) != -1 and int(owner_color[r, c]) != int(claimed_color):
                allowed = []
            else:
                winner_color = int(claimed_color)
                allowed = [item for item in arrivals if int(item[1]) == int(winner_color)]

            for _dist, color, prev_cell in allowed:
                for next_cell in _next_cells(adjacency, prev_cell, (r, c)):
                    nr, nc = next_cell
                    if (
                        int(owner_color[nr, nc]) != -1
                        and int(owner_color[nr, nc]) != int(color)
                        and int(owner_time[nr, nc]) < int(cur_time) + 1
                    ):
                        continue
                    enqueue(int(color), (r, c), next_cell, int(cur_time) + 1)

    return out


def _inside(pos: tuple[int, int], rows: int, cols: int) -> bool:
    return 0 <= pos[0] < rows and 0 <= pos[1] < cols


def _add_edge(
    graph: dict[tuple[int, int], set[tuple[int, int]]],
    a: tuple[int, int],
    b: tuple[int, int],
) -> None:
    graph.setdefault(a, set()).add(b)
    graph.setdefault(b, set()).add(a)


def _can_add_child(
    graph: dict[tuple[int, int], set[tuple[int, int]]],
    parent: tuple[int, int],
    child: tuple[int, int],
    rows: int,
    cols: int,
) -> bool:
    if not _inside(child, rows, cols) or child in graph:
        return False
    occupied = []
    for dr, dc in DIRS4:
        nb = (child[0] + dr, child[1] + dc)
        if nb in graph:
            occupied.append(nb)
    return len(occupied) == 1 and occupied[0] == parent


def _leaves(graph: dict[tuple[int, int], set[tuple[int, int]]]) -> list[tuple[int, int]]:
    return [node for node, neigh in graph.items() if len(neigh) == 1]


def _has_degree(graph: dict[tuple[int, int], set[tuple[int, int]]], degree: int) -> bool:
    return any(len(neigh) == degree for neigh in graph.values())


def _has_corner(graph: dict[tuple[int, int], set[tuple[int, int]]]) -> bool:
    for node, neigh in graph.items():
        if len(neigh) != 2:
            continue
        dirs = [(nb[0] - node[0], nb[1] - node[1]) for nb in neigh]
        if dirs[0] != (-dirs[1][0], -dirs[1][1]):
            return True
    return False


def _is_connected(graph: dict[tuple[int, int], set[tuple[int, int]]]) -> bool:
    if not graph:
        return False
    start = next(iter(graph))
    seen = {start}
    stack = [start]
    while stack:
        cur = stack.pop()
        for nb in graph[cur]:
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(graph)


def _is_tree(graph: dict[tuple[int, int], set[tuple[int, int]]]) -> bool:
    edge_count = sum(len(neigh) for neigh in graph.values()) // 2
    return _is_connected(graph) and edge_count == len(graph) - 1


def _extend_from_leaf(
    graph: dict[tuple[int, int], set[tuple[int, int]]],
    leaf: tuple[int, int],
    steps: int,
    rows: int,
    cols: int,
    *,
    prefer_turn: bool = False,
    force_turn: bool = False,
) -> tuple[tuple[int, int], bool, list[tuple[int, int]]]:
    if steps <= 0 or leaf not in graph or len(graph[leaf]) != 1:
        return leaf, False, [leaf]

    parent = next(iter(graph[leaf]))
    current = leaf
    last_dir = (current[0] - parent[0], current[1] - parent[1])
    turned = False
    path = [leaf]

    for _ in range(steps):
        candidates = []
        back = (-last_dir[0], -last_dir[1])
        for d in DIRS4:
            if d == back:
                continue
            nxt = (current[0] + d[0], current[1] + d[1])
            if not _can_add_child(graph, current, nxt, rows, cols):
                continue
            is_turn = d != last_dir
            score = 0
            if force_turn and not turned:
                score = 2 if is_turn else 0
            elif prefer_turn and not turned:
                score = 2 if is_turn else 1
            else:
                score = 1 if d == last_dir else 0
            candidates.append((score, random.random(), d, nxt))
        if not candidates:
            break
        best = max(item[0] for item in candidates)
        _, _, chosen_dir, nxt = random.choice([item for item in candidates if item[0] == best])
        _add_edge(graph, current, nxt)
        path.append(nxt)
        if chosen_dir != last_dir:
            turned = True
        current = nxt
        last_dir = chosen_dir

    return current, turned, path


def _extend_in_direction(
    graph: dict[tuple[int, int], set[tuple[int, int]]],
    leaf: tuple[int, int],
    direction: tuple[int, int],
    steps: int,
    rows: int,
    cols: int,
) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    current = leaf
    path: list[tuple[int, int]] = []
    for _ in range(steps):
        nxt = (current[0] + direction[0], current[1] + direction[1])
        if not _can_add_child(graph, current, nxt, rows, cols):
            break
        _add_edge(graph, current, nxt)
        current = nxt
        path.append(current)
    return current, path


def _pick_center(rows: int, cols: int, margin: int = 2) -> tuple[int, int]:
    return (
        random.randint(margin, rows - margin - 1),
        random.randint(margin, cols - margin - 1),
    )


def _build_path_graph(rows: int, cols: int) -> dict[tuple[int, int], set[tuple[int, int]]] | None:
    for _ in range(80):
        start = (random.randint(1, rows - 2), random.randint(1, cols - 2))
        graph: dict[tuple[int, int], set[tuple[int, int]]] = {start: set()}
        current = start
        last_dir: tuple[int, int] | None = None
        turned = False
        target_edges = random.randint(4, 7)

        for step in range(target_edges):
            candidates = []
            for d in DIRS4:
                if last_dir is not None and d == (-last_dir[0], -last_dir[1]):
                    continue
                nxt = (current[0] + d[0], current[1] + d[1])
                if not _can_add_child(graph, current, nxt, rows, cols):
                    continue
                is_turn = last_dir is not None and d != last_dir
                score = 2 if (not turned and is_turn and step >= 1) else int(last_dir is None or d == last_dir)
                candidates.append((score, random.random(), d, nxt))
            if not candidates:
                break
            best = max(item[0] for item in candidates)
            _, _, chosen_dir, nxt = random.choice([item for item in candidates if item[0] == best])
            _add_edge(graph, current, nxt)
            if last_dir is not None and chosen_dir != last_dir:
                turned = True
            current = nxt
            last_dir = chosen_dir

        if _is_tree(graph) and len(_leaves(graph)) == 2 and _has_corner(graph):
            return graph
    return None


def _build_t_graph(rows: int, cols: int) -> dict[tuple[int, int], set[tuple[int, int]]] | None:
    for _ in range(80):
        center = _pick_center(rows, cols)
        graph: dict[tuple[int, int], set[tuple[int, int]]] = {center: set()}
        missing = random.choice(DIRS4)
        arm_dirs = [d for d in DIRS4 if d != missing]
        leaves: list[tuple[tuple[int, int], tuple[int, int]]] = []
        ok = True
        for d in arm_dirs:
            leaf = (center[0] + d[0], center[1] + d[1])
            if not _can_add_child(graph, center, leaf, rows, cols):
                ok = False
                break
            _add_edge(graph, center, leaf)
            leaves.append((leaf, d))
        if not ok:
            continue

        random.shuffle(leaves)
        for idx, (leaf, _d) in enumerate(leaves):
            min_steps = 1 if idx == 0 else 0
            steps = random.randint(min_steps, 3)
            _extend_from_leaf(
                graph,
                leaf,
                steps,
                rows,
                cols,
                prefer_turn=idx == 0,
                force_turn=idx == 0 and steps >= 2,
            )

        if _is_tree(graph) and len(_leaves(graph)) == 3 and _has_degree(graph, 3):
            return graph
    return None


def _build_cross_graph(rows: int, cols: int) -> dict[tuple[int, int], set[tuple[int, int]]] | None:
    for _ in range(80):
        center = _pick_center(rows, cols)
        graph: dict[tuple[int, int], set[tuple[int, int]]] = {center: set()}
        leaves: list[tuple[int, int]] = []
        ok = True
        for d in DIRS4:
            leaf = (center[0] + d[0], center[1] + d[1])
            if not _can_add_child(graph, center, leaf, rows, cols):
                ok = False
                break
            _add_edge(graph, center, leaf)
            leaves.append(leaf)
        if not ok:
            continue

        random.shuffle(leaves)
        for idx, leaf in enumerate(leaves):
            min_steps = 1 if idx == 0 else 0
            steps = random.randint(min_steps, 3)
            _extend_from_leaf(
                graph,
                leaf,
                steps,
                rows,
                cols,
                prefer_turn=idx == 0,
                force_turn=idx == 0 and steps >= 2,
            )

        if _is_tree(graph) and len(_leaves(graph)) == 4 and _has_degree(graph, 4):
            return graph
    return None


def _build_mixed_graph(rows: int, cols: int) -> dict[tuple[int, int], set[tuple[int, int]]] | None:
    for _ in range(120):
        center = _pick_center(rows, cols)
        graph: dict[tuple[int, int], set[tuple[int, int]]] = {center: set()}
        arms: dict[tuple[int, int], tuple[int, int]] = {}
        ok = True
        for d in DIRS4:
            leaf = (center[0] + d[0], center[1] + d[1])
            if not _can_add_child(graph, center, leaf, rows, cols):
                ok = False
                break
            _add_edge(graph, center, leaf)
            arms[d] = leaf
        if not ok:
            continue

        trunk_dir = random.choice(DIRS4)
        trunk_leaf = arms[trunk_dir]
        trunk_leaf, trunk_path = _extend_in_direction(
            graph, trunk_leaf, trunk_dir, random.randint(1, 2), rows, cols
        )
        arm_nodes = [arms[trunk_dir]] + trunk_path
        if len(arm_nodes) < 2:
            continue

        branch_node = random.choice(arm_nodes[:-1])
        branch_dirs = PERPENDICULAR[trunk_dir][:]
        random.shuffle(branch_dirs)
        branched = False
        for branch_dir in branch_dirs:
            branch_leaf = (branch_node[0] + branch_dir[0], branch_node[1] + branch_dir[1])
            if not _can_add_child(graph, branch_node, branch_leaf, rows, cols):
                continue
            _add_edge(graph, branch_node, branch_leaf)
            _extend_from_leaf(
                graph,
                branch_leaf,
                random.randint(0, 2),
                rows,
                cols,
                prefer_turn=True,
            )
            branched = True
            break
        if not branched:
            continue

        for d, leaf in arms.items():
            if d == trunk_dir:
                continue
            _extend_from_leaf(
                graph,
                leaf,
                random.randint(0, 2),
                rows,
                cols,
                prefer_turn=(random.random() < 0.4),
            )

        if (
            _is_tree(graph)
            and _has_degree(graph, 4)
            and _has_degree(graph, 3)
            and _has_corner(graph)
            and 4 <= len(_leaves(graph)) <= len(SEED_COLOR_POOL)
        ):
            return graph
    return None


def _build_graph(variant: str, rows: int, cols: int):
    if variant == "path":
        return _build_path_graph(rows, cols)
    if variant == "t":
        return _build_t_graph(rows, cols)
    if variant == "cross":
        return _build_cross_graph(rows, cols)
    return _build_mixed_graph(rows, cols)


def _rasterize_graph(graph: dict[tuple[int, int], set[tuple[int, int]]]) -> np.ndarray:
    scale = random.choice(SCALE_OPTIONS)
    node_pixels = {node: (node[0] * scale, node[1] * scale) for node in graph}
    pipe_pixels: set[tuple[int, int]] = set()

    for node, pixel in node_pixels.items():
        pipe_pixels.add(pixel)
        for nb in graph[node]:
            if node >= nb:
                continue
            r0, c0 = pixel
            r1, c1 = node_pixels[nb]
            if r0 == r1:
                step = 1 if c1 > c0 else -1
                for c in range(c0, c1 + step, step):
                    pipe_pixels.add((r0, c))
            else:
                step = 1 if r1 > r0 else -1
                for r in range(r0, r1 + step, step):
                    pipe_pixels.add((r, c0))

    rows = [p[0] for p in pipe_pixels]
    cols = [p[1] for p in pipe_pixels]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    top = random.randint(1, 4)
    bottom = random.randint(1, 4)
    left = random.randint(1, 4)
    right = random.randint(1, 4)
    h = (max_r - min_r + 1) + top + bottom
    w = (max_c - min_c + 1) + left + right
    grid = np.zeros((h, w), dtype=int)

    for r, c in pipe_pixels:
        grid[r - min_r + top, c - min_c + left] = GREY

    leaves = _leaves(graph)
    colors = random.sample(SEED_COLOR_POOL, len(leaves))
    random.shuffle(leaves)
    for leaf, color in zip(leaves, colors):
        pr, pc = node_pixels[leaf]
        grid[pr - min_r + top, pc - min_c + left] = int(color)

    return grid


def _valid_generated_grid(grid: np.ndarray, variant: str) -> bool:
    out = main(grid)
    if out.shape != grid.shape:
        return False
    if not np.any(grid != out):
        return False
    if np.any((grid == GREY) & (out == GREY)):
        return False

    pipe_mask = grid != BLACK
    degree_counts = []
    seed_count = 0
    for r, c in np.argwhere(pipe_mask):
        rr = int(r)
        cc = int(c)
        deg = len(_pipe_neighbors(pipe_mask, rr, cc))
        degree_counts.append(deg)
        if grid[rr, cc] != GREY:
            if deg != 1:
                return False
            seed_count += 1

    if seed_count < 2 or seed_count > len(SEED_COLOR_POOL):
        return False
    if any(deg > 4 or deg < 1 for deg in degree_counts):
        return False

    if variant == "path":
        return seed_count == 2
    if variant == "t":
        return 3 in degree_counts
    if variant == "cross":
        return 4 in degree_counts
    return 3 in degree_counts and 4 in degree_counts


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        variant = random.choice(VARIANTS)
        coarse_rows = random.randint(*COARSE_ROWS_RANGE)
        coarse_cols = random.randint(*COARSE_COLS_RANGE)
        graph = _build_graph(variant, coarse_rows, coarse_cols)
        if graph is None:
            continue
        if len(_leaves(graph)) > len(SEED_COLOR_POOL):
            continue
        grid = _rasterize_graph(graph)
        if _valid_generated_grid(grid, variant):
            return grid
    raise RuntimeError("failed to generate a valid pipe-network ARC input")
