from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

COLOR_CYCLE = [RED, BLUE, GREEN, YELLOW]
COLOR_INDEX = {color: idx for idx, color in enumerate(COLOR_CYCLE)}
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_ATTEMPTS = 300
COARSE_ROWS = (3, 5)
COARSE_COLS = (3, 6)
NODE_COUNT = (4, 9)
SPACING_CHOICES = [5, 6, 7]
MARGIN_BASE = 7
MARGIN_JITTER = 3
FRAME_SIZE = (3, 6)
FRAME_NEAR_DIST = [3, 4]
SEED_COLORS = [RED, BLUE, GREEN]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if _in_bounds(h, w, nr, nc) and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _shift_color(color: int, steps: int) -> int:
    return COLOR_CYCLE[(COLOR_INDEX[color] + steps) % len(COLOR_CYCLE)]


def _direction(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    ar, ac = a
    br, bc = b
    dr = 0 if ar == br else (1 if br > ar else -1)
    dc = 0 if ac == bc else (1 if bc > ac else -1)
    assert (dr == 0) != (dc == 0), "tree edges must be axis-aligned"
    return dr, dc


def _find_node_centers(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) == BLACK:
                continue
            if all(int(grid[r + dr, c + dc]) == GREY for dr, dc in DIRS4):
                centers.append((r, c))
    assert centers, "expected at least one network node"
    return centers


def _build_graph(grid: np.ndarray, centers: list[tuple[int, int]]) -> dict[tuple[int, int], list[tuple[tuple[int, int], int]]]:
    h, w = grid.shape
    center_set = set(centers)
    adj = {center: [] for center in centers}
    seen_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    for center in centers:
        cr, cc = center
        for dr, dc in DIRS4:
            arm_r, arm_c = cr + dr, cc + dc
            if not _in_bounds(h, w, arm_r, arm_c) or int(grid[arm_r, arm_c]) != GREY:
                continue
            nr, nc = cr + 2 * dr, cc + 2 * dc
            if not _in_bounds(h, w, nr, nc) or int(grid[nr, nc]) == BLACK:
                continue

            path: list[tuple[int, int]] = []
            while _in_bounds(h, w, nr, nc) and int(grid[nr, nc]) != BLACK:
                path.append((nr, nc))
                if (nr, nc) in center_set:
                    other = (nr, nc)
                    break
                nr += dr
                nc += dc
            else:
                continue

            if other == center:
                continue

            edge_key = tuple(sorted((center, other)))
            if edge_key in seen_edges:
                continue

            mid_cells = path[:-1]
            yellow_count = sum(int(grid[r, c]) == YELLOW for r, c in mid_cells)
            assert yellow_count == 1, "each connection must contain exactly one transformer"

            seen_edges.add(edge_key)
            adj[center].append((other, yellow_count))
            adj[other].append((center, yellow_count))

    assert len(seen_edges) == len(centers) - 1, "network must be a tree"

    reached = set()
    q = deque([centers[0]])
    reached.add(centers[0])
    while q:
        node = q.popleft()
        for nxt, _ in adj[node]:
            if nxt not in reached:
                reached.add(nxt)
                q.append(nxt)
    assert len(reached) == len(centers), "network nodes must be connected"
    return adj


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for comp in _components(grid != BLACK):
        if {int(grid[r, c]) for r, c in comp} != {GREY}:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        if len(comp) != 2 * h + 2 * w - 4:
            continue

        comp_set = set(comp)
        ok = True
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                on_border = r in (r0, r1) or c in (c0, c1)
                if on_border != ((r, c) in comp_set):
                    ok = False
                    break
                if not on_border and int(grid[r, c]) != BLACK:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            frames.append((r0, c0, r1, c1))
    return frames


def _propagate_colors(
    adj: dict[tuple[int, int], list[tuple[tuple[int, int], int]]],
    seed_colors: dict[tuple[int, int], int],
) -> dict[tuple[int, int], int]:
    seed_order = {node: idx for idx, node in enumerate(sorted(seed_colors))}
    dist: dict[tuple[int, int], int] = {}
    transformers: dict[tuple[int, int], int] = {}
    origin: dict[tuple[int, int], tuple[int, int]] = {}
    q = deque()

    for node, color in seed_colors.items():
        assert color in COLOR_INDEX, "seed colors must come from the visible cycle"
        dist[node] = 0
        transformers[node] = 0
        origin[node] = node
        q.append(node)

    while q:
        node = q.popleft()
        for nxt, yellow_count in adj[node]:
            cand_dist = dist[node] + 1
            cand_origin = origin[node]
            cand_transformers = transformers[node] + yellow_count
            if nxt not in dist:
                better = True
            elif cand_dist < dist[nxt]:
                better = True
            elif cand_dist == dist[nxt] and seed_order[cand_origin] < seed_order[origin[nxt]]:
                better = True
            else:
                better = False
            if better:
                dist[nxt] = cand_dist
                transformers[nxt] = cand_transformers
                origin[nxt] = cand_origin
                q.append(nxt)

    assert len(dist) == len(adj), "all nodes must receive a propagated color"
    colors: dict[tuple[int, int], int] = {}
    for node in adj:
        base_color = seed_colors[origin[node]]
        colors[node] = _shift_color(base_color, transformers[node])
    return colors


def _frame_relation(
    leaf: tuple[int, int],
    frame: tuple[int, int, int, int],
) -> tuple[tuple[int, int], int] | None:
    r, c = leaf
    r0, c0, r1, c1 = frame
    options: list[tuple[tuple[int, int], int]] = []
    if c < c0 and r0 - 1 <= r <= r1 + 1:
        options.append(((0, 1), c0 - c))
    if c > c1 and r0 - 1 <= r <= r1 + 1:
        options.append(((0, -1), c - c1))
    if r < r0 and c0 - 1 <= c <= c1 + 1:
        options.append(((1, 0), r0 - r))
    if r > r1 and c0 - 1 <= c <= c1 + 1:
        options.append(((-1, 0), r - r1))
    if not options:
        return None
    return min(options, key=lambda item: item[1])


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    centers = _find_node_centers(g)
    adj = _build_graph(g, centers)
    seed_colors = {center: int(g[center]) for center in centers if int(g[center]) != GREY}
    assert seed_colors, "expected at least one seed node"

    node_colors = _propagate_colors(adj, seed_colors)
    frames = _find_frames(g)
    out = g.copy()

    leaves = [node for node, edges in adj.items() if len(edges) == 1]
    used_frames: set[int] = set()
    for leaf in sorted(leaves):
        parent, _ = adj[leaf][0]
        blocked_direction = _direction(leaf, parent)
        best_match: tuple[int, int] | None = None
        best_frame_index: int | None = None
        for idx, frame in enumerate(frames):
            if idx in used_frames:
                continue
            relation = _frame_relation(leaf, frame)
            if relation is None:
                continue
            direction, dist = relation
            if direction == blocked_direction or dist > 7:
                continue
            score = (dist, idx)
            if best_match is None or score < best_match:
                best_match = score
                best_frame_index = idx
        if best_frame_index is None:
            continue

        used_frames.add(best_frame_index)
        r0, c0, r1, c1 = frames[best_frame_index]
        out[r0 + 1 : r1, c0 + 1 : c1] = node_colors[leaf]

    return out


def _grid_neighbors(node: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = node
    out = []
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            out.append((nr, nc))
    return out


def _grow_tree(rows: int, cols: int, n_nodes: int) -> tuple[list[tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int]]]] | None:
    if n_nodes > rows * cols:
        return None
    start = (random.randrange(rows), random.randrange(cols))
    chosen = {start}
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    while len(chosen) < n_nodes:
        frontier: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for node in chosen:
            for nxt in _grid_neighbors(node, rows, cols):
                if nxt not in chosen:
                    frontier.append((node, nxt))
        if not frontier:
            return None
        edge = random.choice(frontier)
        chosen.add(edge[1])
        edges.append(edge)
    nodes = sorted(chosen)
    return nodes, edges


def _abstract_adjacency(
    nodes: list[tuple[int, int]],
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    adj = {node: [] for node in nodes}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    return adj


def _bfs_distances(
    adj: dict[tuple[int, int], list[tuple[int, int]]],
    start: tuple[int, int],
) -> dict[tuple[int, int], int]:
    dist = {start: 0}
    q = deque([start])
    while q:
        node = q.popleft()
        for nxt in adj[node]:
            if nxt not in dist:
                dist[nxt] = dist[node] + 1
                q.append(nxt)
    return dist


def _paint_cross(grid: np.ndarray, center: tuple[int, int], color: int) -> None:
    r, c = center
    grid[r, c] = color
    for dr, dc in DIRS4:
        grid[r + dr, c + dc] = GREY


def _paint_line(grid: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> None:
    ar, ac = a
    br, bc = b
    if ar == br:
        lo, hi = sorted((ac, bc))
        grid[ar, lo + 1 : hi] = GREY
        tcol = random.randint(lo + 2, hi - 2)
        grid[ar, tcol] = YELLOW
    else:
        lo, hi = sorted((ar, br))
        grid[lo + 1 : hi, ac] = GREY
        trow = random.randint(lo + 2, hi - 2)
        grid[trow, ac] = YELLOW


def _frame_bbox(
    center: tuple[int, int],
    direction: tuple[int, int],
    outer_h: int,
    outer_w: int,
    near_dist: int,
) -> tuple[int, int, int, int]:
    r, c = center
    if direction == (0, 1):
        c0 = c + near_dist
        c1 = c0 + outer_w - 1
        offset = random.randint(0, outer_h - 1)
        r0 = r - offset
        r1 = r0 + outer_h - 1
    elif direction == (0, -1):
        c1 = c - near_dist
        c0 = c1 - outer_w + 1
        offset = random.randint(0, outer_h - 1)
        r0 = r - offset
        r1 = r0 + outer_h - 1
    elif direction == (1, 0):
        r0 = r + near_dist
        r1 = r0 + outer_h - 1
        offset = random.randint(0, outer_w - 1)
        c0 = c - offset
        c1 = c0 + outer_w - 1
    else:
        r1 = r - near_dist
        r0 = r1 - outer_h + 1
        offset = random.randint(0, outer_w - 1)
        c0 = c - offset
        c1 = c0 + outer_w - 1
    return r0, c0, r1, c1


def _can_place_frame(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    h, w = grid.shape
    r0, c0, r1, c1 = bbox
    if r0 < 1 or c0 < 1 or r1 >= h - 1 or c1 >= w - 1:
        return False
    return bool(np.all(grid[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2] == BLACK))


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY


def _try_place_frames(
    grid: np.ndarray,
    pixel_nodes: dict[tuple[int, int], tuple[int, int]],
    abstract_adj: dict[tuple[int, int], list[tuple[int, int]]],
) -> dict[tuple[int, int], tuple[int, int, int, int]] | None:
    leaves = [node for node, nbrs in abstract_adj.items() if len(nbrs) == 1]
    random.shuffle(leaves)
    target_count = random.randint(1, min(len(leaves), 4))
    placed: dict[tuple[int, int], tuple[int, int, int, int]] = {}

    for leaf in leaves:
        if len(placed) >= target_count:
            break
        parent = abstract_adj[leaf][0]
        blocked_direction = _direction(leaf, parent)
        directions = [d for d in DIRS4 if d != blocked_direction]
        random.shuffle(directions)
        success = False
        for direction in directions:
            for _ in range(12):
                outer_h = random.randint(*FRAME_SIZE)
                outer_w = random.randint(*FRAME_SIZE)
                near_dist = random.choice(FRAME_NEAR_DIST)
                bbox = _frame_bbox(pixel_nodes[leaf], direction, outer_h, outer_w, near_dist)
                if not _can_place_frame(grid, bbox):
                    continue
                _paint_frame(grid, bbox)
                placed[leaf] = bbox
                success = True
                break
            if success:
                break

    if len(placed) != target_count:
        return None
    return placed


def _pick_seed_assignment(
    nodes: list[tuple[int, int]],
    abstract_adj: dict[tuple[int, int], list[tuple[int, int]]],
    framed_leaves: set[tuple[int, int]],
) -> dict[tuple[int, int], int] | None:
    dist_cache = {node: _bfs_distances(abstract_adj, node) for node in nodes}
    max_seeds = min(3, len(nodes), len(SEED_COLORS))
    counts = [count for count in (1, 2, 3) if count <= max_seeds]

    for _ in range(80):
        n_seeds = random.choice(counts)
        seed_nodes = random.sample(nodes, n_seeds)

        unique = True
        nearest: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
        for node in nodes:
            pairs = sorted((dist_cache[seed][node], seed) for seed in seed_nodes)
            if len(pairs) > 1 and pairs[0][0] == pairs[1][0]:
                unique = False
                break
            nearest[node] = pairs[0]
        if not unique:
            continue

        if not any(leaf not in seed_nodes for leaf in framed_leaves):
            continue

        colors = random.sample(SEED_COLORS, n_seeds)
        seed_assignment = {node: color for node, color in zip(seed_nodes, colors, strict=False)}

        transformed = []
        for leaf in framed_leaves:
            dist, seed = nearest[leaf]
            transformed.append(_shift_color(seed_assignment[seed], dist))
        if not transformed:
            continue
        if all(color == seed_assignment[nearest[leaf][1]] for leaf, color in zip(sorted(framed_leaves), transformed, strict=False)):
            continue

        return seed_assignment
    return None


def _generate_attempt() -> np.ndarray | None:
    coarse_rows = random.randint(*COARSE_ROWS)
    coarse_cols = random.randint(*COARSE_COLS)
    n_nodes = random.randint(NODE_COUNT[0], min(NODE_COUNT[1], coarse_rows * coarse_cols))
    tree = _grow_tree(coarse_rows, coarse_cols, n_nodes)
    if tree is None:
        return None
    nodes, edges = tree
    abstract_adj = _abstract_adjacency(nodes, edges)

    spacing = random.choice(SPACING_CHOICES)
    top = MARGIN_BASE + random.randint(0, MARGIN_JITTER)
    left = MARGIN_BASE + random.randint(0, MARGIN_JITTER)
    bottom = MARGIN_BASE + random.randint(0, MARGIN_JITTER)
    right = MARGIN_BASE + random.randint(0, MARGIN_JITTER)
    height = top + bottom + 1 + (coarse_rows - 1) * spacing
    width = left + right + 1 + (coarse_cols - 1) * spacing
    grid = np.zeros((height, width), dtype=int)

    pixel_nodes = {
        node: (top + node[0] * spacing, left + node[1] * spacing)
        for node in nodes
    }
    for center in pixel_nodes.values():
        _paint_cross(grid, center, GREY)
    for a, b in edges:
        _paint_line(grid, pixel_nodes[a], pixel_nodes[b])

    frames = _try_place_frames(grid, pixel_nodes, abstract_adj)
    if frames is None:
        return None

    seeds = _pick_seed_assignment(nodes, abstract_adj, set(frames))
    if seeds is None:
        return None

    for node, color in seeds.items():
        r, c = pixel_nodes[node]
        grid[r, c] = color

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
