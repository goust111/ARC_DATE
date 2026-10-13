from __future__ import annotations

import itertools
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
MAGENTA = 6
ORANGE = 7

DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
PRIMARY_KEYS = ("R", "B", "Y")
PRIMARY_COLORS = {"R": RED, "B": BLUE, "Y": YELLOW}
PRIMARY_SUBSETS = {
    RED: frozenset({"R"}),
    BLUE: frozenset({"B"}),
    YELLOW: frozenset({"Y"}),
}
SUBSET_TO_COLOR = {
    frozenset({"R"}): RED,
    frozenset({"B"}): BLUE,
    frozenset({"Y"}): YELLOW,
    frozenset({"R", "B"}): MAGENTA,
    frozenset({"R", "Y"}): ORANGE,
    frozenset({"B", "Y"}): GREEN,
    frozenset({"R", "B", "Y"}): GRAY,
}

PLAN_CANVAS = 29
PLAN_CENTER = PLAN_CANVAS // 2
CHAIN_LEN_RANGE = (2, 4)
LEAF_LEN_RANGE = (2, 5)
GEN_ATTEMPTS = 400
LAYOUT_ATTEMPTS = 120


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS.values():
        yield r + dr, c + dc


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _opposite(name: str) -> str:
    return {"U": "D", "D": "U", "L": "R", "R": "L"}[name]


def _step(cell: tuple[int, int], name: str, count: int = 1) -> tuple[int, int]:
    dr, dc = DIRS[name]
    return cell[0] + dr * count, cell[1] + dc * count


def _line_cells(start: tuple[int, int], name: str, length: int) -> list[tuple[int, int]]:
    return [_step(start, name, i) for i in range(length + 1)]


def _mix_color(subset: frozenset[str]) -> int:
    return SUBSET_TO_COLOR[frozenset(subset)]


def _special_graph(pipe_mask: np.ndarray):
    H, W = pipe_mask.shape
    degrees: dict[tuple[int, int], int] = {}
    for r, c in np.argwhere(pipe_mask):
        cell = (int(r), int(c))
        degrees[cell] = sum(
            1 for rr, cc in _neighbors4(*cell) if _in_bounds(pipe_mask.shape, rr, cc) and pipe_mask[rr, cc]
        )

    special = {cell for cell, deg in degrees.items() if deg != 2}
    assert special, "expected at least one endpoint or junction"

    edge_paths: dict[frozenset[tuple[int, int]], list[tuple[int, int]]] = {}
    graph: dict[tuple[int, int], list[tuple[int, int]]] = {cell: [] for cell in special}
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    for node in special:
        for rr, cc in _neighbors4(*node):
            nxt = (rr, cc)
            if not (_in_bounds(pipe_mask.shape, rr, cc) and pipe_mask[nxt]):
                continue
            directed = (node, nxt)
            if directed in seen:
                continue
            seen.add(directed)
            path = [node, nxt]
            prev = node
            cur = nxt
            while cur not in special:
                next_cells = [
                    (nr, nc)
                    for nr, nc in _neighbors4(*cur)
                    if _in_bounds(pipe_mask.shape, nr, nc) and pipe_mask[nr, nc] and (nr, nc) != prev
                ]
                assert len(next_cells) == 1, "non-special pipe pixels must form simple segments"
                prev, cur = cur, next_cells[0]
                path.append(cur)
            other = cur
            seen.add((other, prev))
            key = frozenset({node, other})
            if key in edge_paths:
                continue
            edge_paths[key] = path
            graph[node].append(other)
            graph[other].append(node)

    return graph, edge_paths, degrees


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate primary seed colors through a gray channel tree and mix them at merge points.

    Input structure:
    - The background is black.
    - Gray pixels form a one-pixel-wide orthogonal channel network.
    - Several channel dead ends have one adjacent red, blue, or yellow seed pixel.

    Rule:
    - Each seed recolors the gray segment attached to its dead end until the next junction.
    - At a junction, combine the arriving primary-color sets: red+blue=magenta,
      red+yellow=orange, blue+yellow=green, and all three together stay gray.
    - The junction color continues through the unique downstream segment toward the
      remaining unseeded tail, while seed pixels themselves stay unchanged.

    Key insight:
    - The gray network can be reduced to a tree of straight pipe segments between
      endpoints and junctions; every segment takes the union of seed colors in the
      subtree behind it, and that same union determines each downstream mix.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= {BLACK, BLUE, RED, YELLOW, GRAY}, "unexpected input colors"

    pipe_mask = g == GRAY
    assert np.any(pipe_mask), "expected a gray channel network"
    graph, edge_paths, degrees = _special_graph(pipe_mask)

    seed_by_leaf: dict[tuple[int, int], frozenset[str]] = {}
    for r, c in np.argwhere(np.isin(g, [BLUE, RED, YELLOW])):
        seed = (int(r), int(c))
        neighbors = [
            (rr, cc)
            for rr, cc in _neighbors4(*seed)
            if _in_bounds(g.shape, rr, cc) and pipe_mask[rr, cc]
        ]
        assert len(neighbors) == 1, "every seed must touch exactly one pipe endpoint"
        leaf = neighbors[0]
        assert degrees[leaf] == 1, "seed must attach to a pipe dead end"
        assert leaf not in seed_by_leaf, "pipe dead ends must not hold multiple seeds"
        seed_by_leaf[leaf] = PRIMARY_SUBSETS[int(g[seed])]

    endpoints = [cell for cell, deg in degrees.items() if deg == 1]
    sink_candidates = [cell for cell in endpoints if cell not in seed_by_leaf]
    assert len(sink_candidates) == 1, "expected exactly one unseeded downstream tail"
    sink = sink_candidates[0]
    assert set(endpoints) == set(seed_by_leaf) | {sink}, "every other dead end must carry a seed"

    node_count = len(graph)
    edge_count = len(edge_paths)
    assert edge_count == node_count - 1, "pipe segment graph must be a tree"

    parent: dict[tuple[int, int], tuple[int, int] | None] = {sink: None}
    order = [sink]
    stack = [sink]
    while stack:
        node = stack.pop()
        for nb in graph[node]:
            if nb in parent:
                continue
            parent[nb] = node
            order.append(nb)
            stack.append(nb)
    assert len(parent) == node_count, "pipe tree must be connected"

    children: dict[tuple[int, int], list[tuple[int, int]]] = {node: [] for node in graph}
    for node, par in parent.items():
        if par is not None:
            children[par].append(node)

    aggregates: dict[tuple[int, int], frozenset[str]] = {}
    for node in reversed(order):
        node_children = children[node]
        if not node_children:
            assert node in seed_by_leaf, "every non-sink leaf must be seeded"
            aggregates[node] = seed_by_leaf[node]
            continue
        assert len(node_children) >= (1 if node == sink else 2), "internal nodes must merge multiple inputs"
        merged = frozenset().union(*(aggregates[ch] for ch in node_children))
        assert merged, "merged color set must be non-empty"
        aggregates[node] = frozenset(merged)

    out = g.copy()
    for node in order:
        par = parent[node]
        if par is None:
            continue
        key = frozenset({node, par})
        path = edge_paths[key]
        if path[0] != node:
            path = list(reversed(path))
        color = _mix_color(aggregates[node])
        for r, c in path[:-1]:
            out[r, c] = color

    out[sink] = _mix_color(aggregates[sink])
    return out


def _template_pair(colors: list[str], rng: random.Random) -> list[list[str]]:
    a, b = rng.sample(colors, 2)
    return [[a, b]]


def _template_triple(colors: list[str], rng: random.Random) -> list[list[str]]:
    picks = colors[:]
    rng.shuffle(picks)
    return [picks]


def _template_pair_to_triple(colors: list[str], rng: random.Random) -> list[list[str]]:
    a, b = rng.sample(colors, 2)
    c = next(col for col in colors if col not in {a, b})
    return [[a, b], [c]]


def _template_pair_to_pair(colors: list[str], rng: random.Random) -> list[list[str]]:
    a, b = rng.sample(colors, 2)
    return [[a, b], [rng.choice([a, b])]]


def _template_cross_to_triple(colors: list[str], rng: random.Random) -> list[list[str]]:
    a, b = rng.sample(colors, 2)
    c = next(col for col in colors if col not in {a, b})
    extra = [c, rng.choice([a, b])]
    rng.shuffle(extra)
    return [[a, b], extra]


def _template_long_cascade(colors: list[str], rng: random.Random) -> list[list[str]]:
    a, b = rng.sample(colors, 2)
    c = next(col for col in colors if col not in {a, b})
    return [[a, b], [rng.choice([a, b])], [c]]


def _choose_template(rng: random.Random) -> list[list[str]]:
    colors = list(PRIMARY_KEYS)
    templates = [
        _template_pair,
        _template_triple,
        _template_pair_to_triple,
        _template_pair_to_pair,
        _template_cross_to_triple,
        _template_long_cascade,
    ]
    weights = [2, 2, 2, 2, 2, 1]
    template_fn = rng.choices(templates, weights=weights, k=1)[0]
    return template_fn(colors, rng)


def _can_place_segment(
    blocked: set[tuple[int, int]],
    path: list[tuple[int, int]],
) -> bool:
    start = path[0]
    for idx, cell in enumerate(path[1:], start=1):
        r, c = cell
        if not (0 <= r < PLAN_CANVAS and 0 <= c < PLAN_CANVAS):
            return False
        if cell in blocked:
            return False
        allowed = {start} if idx == 1 else set()
        for nb in _neighbors4(r, c):
            if nb in blocked and nb not in allowed:
                return False
    return True


def _can_place_seed(
    blocked: set[tuple[int, int]],
    endpoint: tuple[int, int],
    seed: tuple[int, int],
) -> bool:
    r, c = seed
    if not (0 <= r < PLAN_CANVAS and 0 <= c < PLAN_CANVAS):
        return False
    if seed in blocked:
        return False
    for nb in _neighbors4(r, c):
        if nb in blocked and nb != endpoint:
            return False
    return True


def _draw_layout(node_leaf_colors: list[list[str]], rng: random.Random):
    n_nodes = len(node_leaf_colors)
    occupied_gray: set[tuple[int, int]] = set()
    blocked: set[tuple[int, int]] = set()
    seeds: list[tuple[tuple[int, int], str]] = []

    root = (
        PLAN_CENTER + rng.randint(-3, 3),
        PLAN_CENTER + rng.randint(-3, 3),
    )
    occupied_gray.add(root)
    blocked.add(root)
    node_positions = [root]
    outgoing_dirs: list[str] = []
    incoming_dirs: list[str | None] = [None]

    cur = root
    for _ in range(n_nodes - 1):
        incoming = incoming_dirs[-1]
        dirs = [name for name in DIRS if name != incoming]
        rng.shuffle(dirs)
        placed = False
        for name in dirs:
            length = rng.randint(*CHAIN_LEN_RANGE)
            path = _line_cells(cur, name, length)
            if not _can_place_segment(blocked, path):
                continue
            for cell in path[1:]:
                occupied_gray.add(cell)
                blocked.add(cell)
            cur = path[-1]
            node_positions.append(cur)
            outgoing_dirs.append(name)
            incoming_dirs.append(_opposite(name))
            placed = True
            break
        if not placed:
            return None

    last = node_positions[-1]
    incoming = incoming_dirs[-1]
    dirs = [name for name in DIRS if name != incoming]
    rng.shuffle(dirs)
    sink_pos = None
    for name in dirs:
        length = rng.randint(*CHAIN_LEN_RANGE)
        path = _line_cells(last, name, length)
        if not _can_place_segment(blocked, path):
            continue
        for cell in path[1:]:
            occupied_gray.add(cell)
            blocked.add(cell)
        outgoing_dirs.append(name)
        sink_pos = path[-1]
        break
    if sink_pos is None:
        return None

    for idx, node in enumerate(node_positions):
        incoming = incoming_dirs[idx]
        outgoing = outgoing_dirs[idx]
        leaf_count = len(node_leaf_colors[idx])
        available = [name for name in DIRS if name not in {incoming, outgoing}]
        combos = list(itertools.combinations(available, leaf_count))
        rng.shuffle(combos)
        placed = False

        for combo in combos:
            added_gray: list[tuple[int, int]] = []
            added_blocked: list[tuple[int, int]] = []
            added_seeds: list[tuple[tuple[int, int], str]] = []
            colors = node_leaf_colors[idx][:]
            rng.shuffle(colors)
            success = True

            for name, color_key in zip(combo, colors):
                lengths = list(range(LEAF_LEN_RANGE[0], LEAF_LEN_RANGE[1] + 1))
                rng.shuffle(lengths)
                branch_done = False
                for length in lengths:
                    path = _line_cells(node, name, length)
                    endpoint = path[-1]
                    seed = _step(endpoint, name)
                    if not _can_place_segment(blocked, path):
                        continue
                    if not _can_place_seed(blocked, endpoint, seed):
                        continue
                    for cell in path[1:]:
                        occupied_gray.add(cell)
                        blocked.add(cell)
                        added_gray.append(cell)
                        added_blocked.append(cell)
                    blocked.add(seed)
                    added_blocked.append(seed)
                    added_seeds.append((seed, color_key))
                    branch_done = True
                    break
                if not branch_done:
                    success = False
                    break

            if success:
                seeds.extend(added_seeds)
                placed = True
                break

            for cell in added_gray:
                occupied_gray.remove(cell)
            for cell in added_blocked:
                blocked.remove(cell)

        if not placed:
            return None

    canvas = np.full((PLAN_CANVAS, PLAN_CANVAS), BLACK, dtype=int)
    for r, c in occupied_gray:
        canvas[r, c] = GRAY
    for (r, c), color_key in seeds:
        canvas[r, c] = PRIMARY_COLORS[color_key]

    return canvas


def _transform(grid: np.ndarray, op: int) -> np.ndarray:
    g = np.asarray(grid)
    if op >= 4:
        g = np.fliplr(g)
    return np.rot90(g, op % 4)


def _crop_with_margin(grid: np.ndarray, rng: random.Random) -> np.ndarray:
    coords = np.argwhere(grid != BLACK)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    top = max(0, int(r0) - rng.randint(1, 3))
    left = max(0, int(c0) - rng.randint(1, 3))
    bottom = min(grid.shape[0], int(r1) + rng.randint(2, 4))
    right = min(grid.shape[1], int(c1) + rng.randint(2, 4))
    return np.array(grid[top:bottom, left:right], dtype=int)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    node_leaf_colors = _choose_template(rng)
    for _ in range(LAYOUT_ATTEMPTS):
        canvas = _draw_layout(node_leaf_colors, rng)
        if canvas is None:
            continue
        canvas = _transform(canvas, rng.randrange(8))
        grid = _crop_with_margin(canvas, rng)
        if max(grid.shape) > 30:
            continue
        try:
            out = main(grid)
        except AssertionError:
            continue
        if np.array_equal(grid, out):
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid channel-mixing ARC input")
