import itertools
import random
from collections import Counter, deque

import numpy as np


BG_COLOR = 0
PIPE_COLOR = 5
SEED_COLOR = 2
MARKER_POOL = [1, 3, 4, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors4(shape: tuple[int, int], r: int, c: int) -> list[tuple[int, int]]:
    h, w = shape
    out = []
    for dr, dc in DIRS4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _edge_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (a, b) if a <= b else (b, a)


def _component_size(mask: np.ndarray) -> int:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return 0
    start = tuple(map(int, pts[0]))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(mask.shape, r, c):
            if bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen)


def _walk_and_recolor(
    out: np.ndarray,
    neighbors: dict[tuple[int, int], list[tuple[int, int]]],
    junction_info: dict[tuple[int, int], tuple[tuple[int, int], int]],
    prev: tuple[int, int] | None,
    node: tuple[int, int],
    color: int,
    visited_edges: set[tuple[tuple[int, int], tuple[int, int]]],
) -> None:
    cur_prev = prev
    cur = node

    while True:
        out[cur] = int(color)
        nbrs = neighbors[cur]

        if cur_prev is not None:
            visited_edges.add(_edge_key(cur_prev, cur))

        if len(nbrs) == 1:
            if cur_prev is not None:
                return
            nxt = nbrs[0]
            if _edge_key(cur, nxt) in visited_edges:
                return
            cur_prev, cur = cur, nxt
            continue

        forward = [n for n in nbrs if n != cur_prev]
        if len(nbrs) <= 2:
            if not forward:
                return
            nxt = forward[0]
            if _edge_key(cur, nxt) in visited_edges:
                return
            cur_prev, cur = cur, nxt
            continue

        selected, branch_color = junction_info[cur]
        for nxt in forward:
            if nxt == selected:
                continue
            if _edge_key(cur, nxt) not in visited_edges:
                _walk_and_recolor(out, neighbors, junction_info, cur, nxt, branch_color, visited_edges)

        if selected == cur_prev or _edge_key(cur, selected) in visited_edges:
            return
        cur_prev, cur = cur, selected


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color a pipe network by following the seeded flow and each junction's marker.

    The network is 4-connected and one cell wide. The red endpoint is the entry seed.
    At each junction, the uniquely colored adjacent marker cell indicates which branch
    keeps the current color; all other outgoing branches inherit the marker's color.
    """
    g = np.asarray(grid, dtype=int)
    bg = BG_COLOR if bool((g == BG_COLOR).any()) else int(Counter(map(int, g.flat)).most_common(1)[0][0])
    mask = g != bg
    if not bool(mask.any()):
        return g.copy()

    pipe_color = PIPE_COLOR if bool((g == PIPE_COLOR).any()) else int(Counter(map(int, g[mask].flat)).most_common(1)[0][0])

    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    neighbors = {cell: [nb for nb in _neighbors4(g.shape, cell[0], cell[1]) if bool(mask[nb])] for cell in coords}
    degrees = {cell: len(neighbors[cell]) for cell in coords}

    terminals = [cell for cell in coords if degrees[cell] == 1]
    seed_candidates = [cell for cell in terminals if int(g[cell]) == SEED_COLOR]
    if not seed_candidates:
        seed_candidates = [cell for cell in terminals if int(g[cell]) != int(pipe_color)]
    assert len(seed_candidates) == 1, "expected exactly one seeded terminal"
    seed = seed_candidates[0]
    seed_color = int(g[seed])

    junction_info: dict[tuple[int, int], tuple[tuple[int, int], int]] = {}
    for cell in coords:
        if degrees[cell] < 3:
            continue
        markers = []
        for nb in neighbors[cell]:
            col = int(g[nb])
            if col not in (bg, pipe_color, seed_color):
                markers.append((nb, col))
        assert len(markers) == 1, "each junction must have exactly one adjacent marker"
        junction_info[cell] = (markers[0][0], markers[0][1])

    out = g.copy()
    visited_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    _walk_and_recolor(out, neighbors, junction_info, None, seed, seed_color, visited_edges)
    return out


TOPOLOGIES = [
    "line",
    "single_t",
    "chain_t",
    "side_t",
    "single_cross",
    "cross_chain",
]
SEGMENT_LENGTHS = list(range(2, 7))
MARGIN_RANGE = (1, 5)
GRID_MIN = 10
GRID_MAX = 28
GEN_RETRIES = 400


def _opposite(direction: tuple[int, int]) -> tuple[int, int]:
    return (-direction[0], -direction[1])


def _plane_neighbors(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _segment_cells(start: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    r, c = start
    dr, dc = direction
    return [(r + dr * step, c + dc * step) for step in range(1, length + 1)]


def _valid_segment(
    start: tuple[int, int],
    direction: tuple[int, int],
    length: int,
    used: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    cells = []
    temp = set()
    prev = start
    for cell in _segment_cells(start, direction, length):
        if cell in used or cell in temp:
            return None
        for nb in _plane_neighbors(cell):
            if nb in used and nb != prev:
                return None
            if nb in temp and nb != prev:
                return None
        cells.append(cell)
        temp.add(cell)
        prev = cell
    return cells


def _subtree_sizes(tree: dict[int, dict[str, object]], node: int = 0) -> dict[int, int]:
    sizes = {}

    def dfs(cur: int) -> int:
        total = 1
        for child in tree[cur]["children"]:
            total += dfs(child)
        sizes[cur] = total
        return total

    dfs(node)
    return sizes


def _make_topology(kind: str, rng: random.Random) -> dict[int, dict[str, object]]:
    tree: dict[int, dict[str, object]] = {}

    def add_node(children: list[int] | None = None, selected: int | None = None) -> int:
        idx = len(tree)
        tree[idx] = {"children": [] if children is None else list(children), "selected": selected}
        return idx

    root = add_node()
    first = add_node()
    tree[root]["children"] = [first]
    tree[root]["selected"] = first

    if kind == "line":
        return tree

    if kind == "single_t":
        a = add_node()
        b = add_node()
        children = [a, b]
        tree[first]["children"] = children
        tree[first]["selected"] = rng.choice(children)
        return tree

    if kind == "chain_t":
        j2 = add_node()
        side = add_node()
        a = add_node()
        b = add_node()
        tree[first]["children"] = [j2, side]
        tree[first]["selected"] = j2
        tree[j2]["children"] = [a, b]
        tree[j2]["selected"] = rng.choice([a, b])
        return tree

    if kind == "side_t":
        leaf = add_node()
        j2 = add_node()
        a = add_node()
        b = add_node()
        tree[first]["children"] = [leaf, j2]
        tree[first]["selected"] = leaf
        tree[j2]["children"] = [a, b]
        tree[j2]["selected"] = rng.choice([a, b])
        return tree

    if kind == "single_cross":
        a = add_node()
        b = add_node()
        c = add_node()
        children = [a, b, c]
        tree[first]["children"] = children
        tree[first]["selected"] = rng.choice(children)
        return tree

    if kind == "cross_chain":
        j2 = add_node()
        side_a = add_node()
        side_b = add_node()
        a = add_node()
        b = add_node()
        tree[first]["children"] = [j2, side_a, side_b]
        tree[first]["selected"] = j2
        tree[j2]["children"] = [a, b]
        tree[j2]["selected"] = rng.choice([a, b])
        return tree

    raise ValueError(f"unknown topology: {kind}")


def _place_tree(
    tree: dict[int, dict[str, object]],
    rng: random.Random,
) -> tuple[dict[int, tuple[int, int]], dict[tuple[int, int], list[tuple[int, int]]], set[tuple[int, int]]] | None:
    sizes = _subtree_sizes(tree)

    def place(
        node: int,
        back_dir: tuple[int, int] | None,
        used: set[tuple[int, int]],
        node_pos: dict[int, tuple[int, int]],
        edge_cells: dict[tuple[int, int], list[tuple[int, int]]],
    ) -> tuple[set[tuple[int, int]], dict[int, tuple[int, int]], dict[tuple[int, int], list[tuple[int, int]]]] | None:
        children = list(tree[node]["children"])
        if not children:
            return used, node_pos, edge_cells

        avail = DIRS4[:] if back_dir is None else [d for d in DIRS4 if d != back_dir]
        if len(children) > len(avail):
            return None

        child_order = sorted(children, key=lambda ch: sizes[ch], reverse=True)
        assignments = list(itertools.permutations(avail, len(children)))
        rng.shuffle(assignments)

        def place_child(
            idx: int,
            cur_used: set[tuple[int, int]],
            cur_node_pos: dict[int, tuple[int, int]],
            cur_edge_cells: dict[tuple[int, int], list[tuple[int, int]]],
            mapping: dict[int, tuple[int, int]],
        ) -> tuple[set[tuple[int, int]], dict[int, tuple[int, int]], dict[tuple[int, int], list[tuple[int, int]]]] | None:
            if idx == len(child_order):
                return cur_used, cur_node_pos, cur_edge_cells

            child = child_order[idx]
            direction = mapping[child]
            lengths = SEGMENT_LENGTHS[:]
            if len(tree[node]["children"]) >= 2 and int(tree[node]["selected"]) == child and bool(tree[child]["children"]):
                lengths = [length for length in lengths if length >= 3]
            rng.shuffle(lengths)
            for length in lengths:
                cells = _valid_segment(cur_node_pos[node], direction, length, cur_used)
                if cells is None:
                    continue
                next_used = set(cur_used)
                next_used.update(cells)
                next_node_pos = dict(cur_node_pos)
                next_node_pos[child] = cells[-1]
                next_edge_cells = dict(cur_edge_cells)
                next_edge_cells[(node, child)] = cells

                placed = place(child, _opposite(direction), next_used, next_node_pos, next_edge_cells)
                if placed is None:
                    continue
                done = place_child(idx + 1, placed[0], placed[1], placed[2], mapping)
                if done is not None:
                    return done
            return None

        for choice in assignments:
            mapping = {child_order[i]: choice[i] for i in range(len(child_order))}
            done = place_child(0, used, node_pos, edge_cells, mapping)
            if done is not None:
                return done
        return None

    start_used = {(0, 0)}
    start_pos = {0: (0, 0)}
    start_edges: dict[tuple[int, int], list[tuple[int, int]]] = {}
    placed = place(0, None, start_used, start_pos, start_edges)
    if placed is None:
        return None

    used, node_pos, edge_cells = placed
    rows = [r for r, _ in used]
    cols = [c for _, c in used]
    if max(rows) - min(rows) + 1 > GRID_MAX - 2 or max(cols) - min(cols) + 1 > GRID_MAX - 2:
        return None
    return node_pos, edge_cells, used


def _render_input(
    tree: dict[int, dict[str, object]],
    node_pos: dict[int, tuple[int, int]],
    edge_cells: dict[tuple[int, int], list[tuple[int, int]]],
    used: set[tuple[int, int]],
    rng: random.Random,
) -> np.ndarray | None:
    rows = [r for r, _ in used]
    cols = [c for _, c in used]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    bbox_h = max_r - min_r + 1
    bbox_w = max_c - min_c + 1

    free_h = GRID_MAX - bbox_h
    free_w = GRID_MAX - bbox_w
    if free_h < 2 or free_w < 2:
        return None

    min_h_pad = min(free_h, max(2, GRID_MIN - bbox_h))
    min_w_pad = min(free_w, max(2, GRID_MIN - bbox_w))
    total_h_pad = rng.randint(min_h_pad, free_h)
    total_w_pad = rng.randint(min_w_pad, free_w)
    top = rng.randint(1, total_h_pad - 1)
    left = rng.randint(1, total_w_pad - 1)
    bottom = total_h_pad - top
    right = total_w_pad - left

    h = bbox_h + top + bottom
    w = bbox_w + left + right

    dr = top - min_r
    dc = left - min_c

    grid = np.full((h, w), BG_COLOR, dtype=int)
    for r, c in used:
        grid[r + dr, c + dc] = PIPE_COLOR

    junctions = [node for node in tree if len(tree[node]["children"]) >= 2]
    marker_colors = rng.sample(MARKER_POOL, len(junctions))
    for node, marker_color in zip(junctions, marker_colors):
        selected = int(tree[node]["selected"])
        marker_cell = edge_cells[(node, selected)][0]
        mr, mc = marker_cell
        grid[mr + dr, mc + dc] = int(marker_color)

    root_r, root_c = node_pos[0]
    grid[root_r + dr, root_c + dc] = SEED_COLOR
    return grid


def _is_connected(mask: np.ndarray) -> bool:
    return _component_size(mask) == int(mask.sum())


def _validate_generated(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        return False
    mask = g != BG_COLOR
    if not bool(mask.any()) or not _is_connected(mask):
        return False

    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    neighbors = {cell: [nb for nb in _neighbors4(g.shape, cell[0], cell[1]) if bool(mask[nb])] for cell in coords}
    degrees = {cell: len(neighbors[cell]) for cell in coords}
    if any(deg < 1 or deg > 4 for deg in degrees.values()):
        return False

    terminals = [cell for cell in coords if degrees[cell] == 1]
    seed_terms = [cell for cell in terminals if int(g[cell]) == SEED_COLOR]
    if len(seed_terms) != 1:
        return False
    if any(int(g[cell]) != PIPE_COLOR for cell in terminals if cell not in seed_terms):
        return False

    marker_positions = set()
    for cell in coords:
        if degrees[cell] < 3:
            continue
        special = []
        for nb in neighbors[cell]:
            col = int(g[nb])
            if col not in (PIPE_COLOR, SEED_COLOR):
                special.append(nb)
        if len(special) != 1:
            return False
        marker_positions.add(special[0])

    all_special = {cell for cell in coords if int(g[cell]) not in (PIPE_COLOR, SEED_COLOR)}
    if all_special != marker_positions:
        return False

    out = main(g)
    if out.shape != g.shape:
        return False
    if bool((out[mask] == PIPE_COLOR).any()):
        return False
    if bool((out[~mask] != BG_COLOR).any()):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    kind = rng.choice(TOPOLOGIES)
    tree = _make_topology(kind, rng)
    placed = _place_tree(tree, rng)
    if placed is None:
        return None

    node_pos, edge_cells, used = placed
    grid = _render_input(tree, node_pos, edge_cells, used, rng)
    if grid is None:
        return None
    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pipe-routing input")
