from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7
MAROON = 9

SOURCE_COLORS = (BLUE, RED, YELLOW)
PASSABLE_COLORS = (GREEN, ORANGE, BLUE, RED, YELLOW)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
INF = 10**9

GEN_RETRIES = 500
COARSE_ROWS_RANGE = (4, 7)
COARSE_COLS_RANGE = (4, 7)
NODE_COUNT_RANGE = (6, 12)
STEP_OPTIONS = (3, 4)
MARGIN_RANGE = (1, 2)
SOURCE_COUNT_OPTIONS = (2, 3)
MODE_OPTIONS = (
    (False, False),
    (True, False),
    (False, True),
    (True, True),
)


def _neighbors4(r: int, c: int, height: int, width: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < height and 0 <= cc < width:
            out.append((rr, cc))
    return out


def _coarse_neighbors(node: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = node
    return _neighbors4(r, c, rows, cols)


def _node_sides(node: tuple[int, int], rows: int, cols: int) -> tuple[str, ...]:
    r, c = node
    sides: list[str] = []
    if r == 0:
        sides.append("top")
    if r == rows - 1:
        sides.append("bottom")
    if c == 0:
        sides.append("left")
    if c == cols - 1:
        sides.append("right")
    return tuple(sides)


def _tree_degrees(
    nodes: set[tuple[int, int]], edges: set[tuple[tuple[int, int], tuple[int, int]]]
) -> dict[tuple[int, int], int]:
    deg = {node: 0 for node in nodes}
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    return deg


def _grow_tree(
    rows: int, cols: int, node_count: int
) -> tuple[set[tuple[int, int]], set[tuple[tuple[int, int], tuple[int, int]]]]:
    start = (random.randrange(rows), random.randrange(cols))
    nodes = {start}
    edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    deg: defaultdict[tuple[int, int], int] = defaultdict(int)

    while len(nodes) < node_count:
        options: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for node in nodes:
            fresh = [nbr for nbr in _coarse_neighbors(node, rows, cols) if nbr not in nodes]
            weight = 1 + min(2, deg[node])
            for nbr in fresh:
                options.extend([(node, nbr)] * weight)
        if not options:
            break
        parent, child = random.choice(options)
        edge = (parent, child) if parent <= child else (child, parent)
        nodes.add(child)
        edges.add(edge)
        deg[parent] += 1
        deg[child] += 1

    return nodes, edges


def _render_tree(
    rows: int,
    cols: int,
    step: int,
    margins: tuple[int, int, int, int],
    nodes: set[tuple[int, int]],
    edges: set[tuple[tuple[int, int], tuple[int, int]]],
    source_map: dict[tuple[int, int], int],
) -> np.ndarray:
    top, bottom, left, right = margins
    height = top + (rows - 1) * step + bottom + 1
    width = left + (cols - 1) * step + right + 1
    assert height <= 30 and width <= 30

    def pos(node: tuple[int, int]) -> tuple[int, int]:
        return (top + node[0] * step, left + node[1] * step)

    grid = np.full((height, width), BLACK, dtype=int)
    degrees = _tree_degrees(nodes, edges)

    for a, b in edges:
        ra, ca = pos(a)
        rb, cb = pos(b)
        if ra == rb:
            c0, c1 = sorted((ca, cb))
            grid[ra, c0 : c1 + 1] = GREEN
        else:
            r0, r1 = sorted((ra, rb))
            grid[r0 : r1 + 1, ca] = GREEN

    for node in nodes:
        r, c = pos(node)
        grid[r, c] = ORANGE if degrees[node] >= 3 else GREEN

    for node, color in source_map.items():
        r, c = pos(node)
        grid[r, c] = int(color)

    return grid


def _touches_edge_passable(grid: np.ndarray) -> bool:
    edge_mask = np.zeros_like(grid, dtype=bool)
    edge_mask[0, :] = True
    edge_mask[-1, :] = True
    edge_mask[:, 0] = True
    edge_mask[:, -1] = True
    return bool(np.isin(grid[edge_mask], PASSABLE_COLORS).any())


def _simulate_waves(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    passable = np.isin(g, PASSABLE_COLORS)

    sources = [
        (int(r), int(c), int(g[r, c]))
        for r, c in np.argwhere(np.isin(g, SOURCE_COLORS))
    ]
    assert sources, "expected at least one dye source"

    neighbor_map: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in np.argwhere(passable):
        rr = int(r)
        cc = int(c)
        nbs = [(nr, nc) for nr, nc in _neighbors4(rr, cc, height, width) if passable[nr, nc]]
        cell_color = int(g[rr, cc])
        if cell_color in SOURCE_COLORS:
            assert len(nbs) == 1, "each source must occupy a corridor dead end"
        elif cell_color != ORANGE:
            assert 1 <= len(nbs) <= 2, "only orange cells may branch"
        else:
            assert len(nbs) >= 1, "signpost must be attached to the corridor network"
        neighbor_map[(rr, cc)] = nbs

    best_time = np.full((height, width), INF, dtype=int)
    winners: dict[tuple[int, int], set[tuple[int, tuple[int, int] | None]]] = {}
    frontier: list[tuple[int, int, int, tuple[int, int] | None]] = []

    for sid, (r, c, _color) in enumerate(sources):
        best_time[r, c] = 0
        winners[(r, c)] = {(sid, None)}
        frontier.append((r, c, sid, None))

    time = 0
    while frontier:
        next_time = time + 1
        attempted: dict[tuple[int, int], set[tuple[int, tuple[int, int]]]] = defaultdict(set)

        for r, c, sid, prev in frontier:
            outgoing = neighbor_map[(r, c)]
            if prev is not None:
                outgoing = [nbr for nbr in outgoing if nbr != prev]
                if int(g[r, c]) != ORANGE:
                    assert len(outgoing) <= 1, "only orange cells may split a wave"
            for nr, nc in outgoing:
                if int(best_time[nr, nc]) < next_time:
                    continue
                attempted[(nr, nc)].add((sid, (r, c)))

        frontier = []
        if not attempted:
            break

        for (r, c), arrivals in attempted.items():
            if next_time < int(best_time[r, c]):
                best_time[r, c] = next_time
                winners[(r, c)] = set(arrivals)
                for sid, prev in arrivals:
                    frontier.append((r, c, sid, prev))
            elif next_time == int(best_time[r, c]):
                current = winners.setdefault((r, c), set())
                new_arrivals = arrivals - current
                if new_arrivals:
                    current.update(new_arrivals)
                    for sid, prev in new_arrivals:
                        frontier.append((r, c, sid, prev))

        time = next_time

    out = g.copy()
    for (r, c), arrivals in winners.items():
        colors = {int(sources[sid][2]) for sid, _prev in arrivals}
        out[r, c] = int(next(iter(colors))) if len(colors) == 1 else MAROON
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate dye waves through a corridor maze and recolor the earliest-reached cells.

    Input structure:
    - Background is black.
    - Green pixels form a one-cell-wide orthogonal corridor network; every non-orange corridor pixel has at most two corridor neighbors.
    - Orange pixels mark branch intersections, and red/blue/yellow source pixels sit on degree-1 dead ends in place of corridor pixels.

    Rule:
    - Each source emits a wave from its unique corridor exit and the waves advance synchronously by Manhattan corridor distance.
    - On green corridor cells a wave continues only forward, while on an orange signpost it branches to every connected direction except the one it came from.
    - A cell keeps the color of the wave(s) that reach it first; later arrivals stop there, and equal-time arrivals of different source colors recolor that cell to maroon.
    - Black background stays unchanged, untouched corridor cells keep their original green/orange color, and source cells remain the initial reached cells at distance zero.

    Key insight:
    - The puzzle is a simultaneous shortest-path process on a corridor graph where orange nodes are the only legal split points, so the final colors are determined by earliest arrival time with tie-based maroon mixing.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"
    return _simulate_waves(g)


def _example_ok(
    inp: np.ndarray,
    out: np.ndarray,
    want_tie: bool,
    want_boundary: bool,
) -> bool:
    if inp.ndim != 2 or out.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return False
    if np.array_equal(inp, out):
        return False
    if not np.any(inp == ORANGE):
        return False
    if not np.any(inp == GREEN):
        return False
    if not np.any(out[inp == ORANGE] != ORANGE):
        return False
    if np.count_nonzero(inp != out) < 6:
        return False
    has_tie = bool(np.any(out == MAROON))
    if has_tie != want_tie:
        return False
    if _touches_edge_passable(inp) != want_boundary:
        return False
    return True


def _gen_attempt(want_tie: bool, want_boundary: bool) -> np.ndarray | None:
    rows = random.randint(*COARSE_ROWS_RANGE)
    cols = random.randint(*COARSE_COLS_RANGE)
    max_nodes = min(NODE_COUNT_RANGE[1], rows * cols)
    node_count = random.randint(NODE_COUNT_RANGE[0], max_nodes)
    step = random.choice(STEP_OPTIONS)

    nodes, edges = _grow_tree(rows, cols, node_count)
    if len(nodes) != node_count:
        return None

    degrees = _tree_degrees(nodes, edges)
    leaves = [node for node, deg in degrees.items() if deg == 1]
    branch_nodes = [node for node, deg in degrees.items() if deg >= 3]
    if len(leaves) < 2 or not branch_nodes:
        return None

    boundary_leaves = [node for node in leaves if _node_sides(node, rows, cols)]
    possible_source_counts = [k for k in SOURCE_COUNT_OPTIONS if k <= len(leaves)]
    if want_boundary:
        possible_source_counts = [k for k in possible_source_counts if len(leaves) > k]
    if not possible_source_counts:
        return None

    for _ in range(40):
        source_count = random.choice(possible_source_counts)
        source_nodes = set(random.sample(leaves, source_count))
        non_source_boundary = [node for node in boundary_leaves if node not in source_nodes]
        if want_boundary and not non_source_boundary:
            continue

        top = random.randint(*MARGIN_RANGE)
        bottom = random.randint(*MARGIN_RANGE)
        left = random.randint(*MARGIN_RANGE)
        right = random.randint(*MARGIN_RANGE)
        if want_boundary:
            boundary_sides = [side for node in non_source_boundary for side in _node_sides(node, rows, cols)]
            side = random.choice(boundary_sides)
            if side == "top":
                top = 0
            elif side == "bottom":
                bottom = 0
            elif side == "left":
                left = 0
            else:
                right = 0

        colors = random.sample(list(SOURCE_COLORS), source_count)
        source_map = {node: color for node, color in zip(source_nodes, colors)}
        inp = _render_tree(rows, cols, step, (top, bottom, left, right), nodes, edges, source_map)

        try:
            out = main(inp)
        except AssertionError:
            continue

        if not _example_ok(inp, out, want_tie, want_boundary):
            continue
        return inp

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        want_tie, want_boundary = random.choice(MODE_OPTIONS)
        result = _gen_attempt(want_tie=want_tie, want_boundary=want_boundary)
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after repeated attempts")
