from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
MAROON = 9

SOURCE_COLORS = (1, 2, 3, 6, 7, 8)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
# --------------- solver ---------------


Cell = tuple[int, int]
ValveCandidate = tuple[Cell, str, tuple[Cell, ...]]


def _neighbors4(shape: tuple[int, int], r: int, c: int) -> list[Cell]:
    h, w = shape
    out: list[Cell] = []
    for dr, dc in DIRS4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _pipe_mask(grid: np.ndarray) -> np.ndarray:
    return (grid != BLACK) & (grid != YELLOW)


def _pipe_neighbors(mask: np.ndarray, r: int, c: int) -> list[Cell]:
    return [(rr, cc) for rr, cc in _neighbors4(mask.shape, r, c) if bool(mask[rr, cc])]


def _straight_axis(mask: np.ndarray, r: int, c: int) -> str | None:
    nbrs = _pipe_neighbors(mask, r, c)
    if len(nbrs) != 2:
        return None
    (r1, c1), (r2, c2) = nbrs
    if r1 == r == r2:
        return "h"
    if c1 == c == c2:
        return "v"
    return None


def _analyze_grid(
    grid: np.ndarray,
) -> tuple[np.ndarray, dict[int, list[Cell]], set[Cell], set[Cell]]:
    mask = _pipe_mask(grid)
    h, w = grid.shape
    sources: dict[int, list[Cell]] = {}
    open_valves: set[Cell] = set()
    closed_valves: set[Cell] = set()

    for r0, c0 in np.argwhere(mask):
        r = int(r0)
        c = int(c0)
        color = int(grid[r, c])
        if color == MAROON:
            axis = _straight_axis(mask, r, c)
            assert axis is not None, "valve must sit on a straight degree-2 pipe cell"
            if axis == "h":
                closed = (
                    (r > 0 and int(grid[r - 1, c]) == YELLOW)
                    or (r + 1 < h and int(grid[r + 1, c]) == YELLOW)
                )
            else:
                closed = (
                    (c > 0 and int(grid[r, c - 1]) == YELLOW)
                    or (c + 1 < w and int(grid[r, c + 1]) == YELLOW)
                )
            if closed:
                closed_valves.add((r, c))
            else:
                open_valves.add((r, c))
        elif color not in (GREY,):
            sources.setdefault(color, []).append((r, c))

    return mask, sources, open_valves, closed_valves


def _bfs_distances(traversable: np.ndarray, starts: list[Cell]) -> np.ndarray:
    dist = np.full(traversable.shape, -1, dtype=int)
    q: deque[Cell] = deque()

    for r, c in starts:
        if bool(traversable[r, c]) and int(dist[r, c]) == -1:
            dist[r, c] = 0
            q.append((r, c))

    while q:
        r, c = q.popleft()
        base = int(dist[r, c]) + 1
        for rr, cc in _neighbors4(traversable.shape, r, c):
            if not bool(traversable[rr, cc]) or int(dist[rr, cc]) != -1:
                continue
            dist[rr, cc] = base
            q.append((rr, cc))

    return dist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split the pipe graph by valve states, then recolor each traversable pipe cell by its unique nearest source.
    Closed valves stay maroon, open valves recolor only when a unique source reaches them, ties keep the original pipe pixel, and all yellow indicators are erased to black.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be two-dimensional"

    out = g.copy()
    out[g == YELLOW] = BLACK

    pipe_mask, sources, _open_valves, closed_valves = _analyze_grid(g)
    traversable = pipe_mask.copy()
    for r, c in closed_valves:
        traversable[r, c] = False

    dist_by_color = {
        color: _bfs_distances(traversable, starts)
        for color, starts in sorted(sources.items())
    }

    for r0, c0 in np.argwhere(pipe_mask):
        r = int(r0)
        c = int(c0)
        original = int(g[r, c])
        pos = (r, c)

        if original not in (GREY, MAROON):
            out[r, c] = original
            continue
        if pos in closed_valves:
            out[r, c] = MAROON
            continue

        best_dist: int | None = None
        winners: list[int] = []
        for color, dist in dist_by_color.items():
            d = int(dist[r, c])
            if d < 0:
                continue
            if best_dist is None or d < best_dist:
                best_dist = d
                winners = [color]
            elif d == best_dist:
                winners.append(color)

        if len(winners) == 1:
            out[r, c] = winners[0]
        else:
            out[r, c] = original

    return out


# --------------- generator ---------------

GEN_ATTEMPTS = 500
LATTICE_H_RANGE = (3, 5)
LATTICE_W_RANGE = (3, 5)
SPACING_CHOICES = (3, 4)
PAD_CHOICES = (1, 2)
PATH_NODE_RANGE = (4, 7)
TREE_NODE_RANGE = (5, 8)
MAX_VALVES = 4


def _lattice_neighbors(node: Cell, lh: int, lw: int) -> list[Cell]:
    r, c = node
    out: list[Cell] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < lh:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < lw:
        out.append((r, c + 1))
    return out


def _grow_path(lh: int, lw: int, count: int) -> tuple[set[Cell], list[tuple[Cell, Cell]]] | None:
    for _ in range(80):
        start = (random.randrange(lh), random.randrange(lw))
        path = [start]
        used = {start}
        while len(path) < count:
            choices = [node for node in _lattice_neighbors(path[-1], lh, lw) if node not in used]
            if not choices:
                break
            nxt = random.choice(choices)
            path.append(nxt)
            used.add(nxt)
        if len(path) == count:
            edges = list(zip(path[:-1], path[1:], strict=False))
            return used, edges
    return None


def _grow_connected_nodes(lh: int, lw: int, count: int) -> set[Cell] | None:
    start = (random.randrange(lh), random.randrange(lw))
    nodes = {start}
    while len(nodes) < count:
        frontier = {
            nbr
            for node in nodes
            for nbr in _lattice_neighbors(node, lh, lw)
            if nbr not in nodes
        }
        if not frontier:
            return None
        nodes.add(random.choice(tuple(frontier)))
    return nodes


def _random_spanning_tree(nodes: set[Cell], lh: int, lw: int) -> list[tuple[Cell, Cell]] | None:
    root = random.choice(tuple(nodes))
    tree_nodes = {root}
    remaining = set(nodes)
    remaining.remove(root)
    edges: list[tuple[Cell, Cell]] = []

    while remaining:
        frontier = [
            (u, v)
            for u in tree_nodes
            for v in _lattice_neighbors(u, lh, lw)
            if v in remaining
        ]
        if not frontier:
            return None
        u, v = random.choice(frontier)
        edges.append((u, v))
        tree_nodes.add(v)
        remaining.remove(v)

    return edges


def _node_degrees(edges: list[tuple[Cell, Cell]]) -> dict[Cell, int]:
    deg: dict[Cell, int] = {}
    for a, b in edges:
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    return deg


def _render_pipe(
    lh: int,
    lw: int,
    spacing: int,
    pads: tuple[int, int, int, int],
    edges: list[tuple[Cell, Cell]],
) -> tuple[np.ndarray, dict[Cell, Cell]]:
    pad_top, pad_bottom, pad_left, pad_right = pads
    h = pad_top + (lh - 1) * spacing + 1 + pad_bottom
    w = pad_left + (lw - 1) * spacing + 1 + pad_right
    grid = np.full((h, w), BLACK, dtype=int)
    node_cells = {
        node: (pad_top + node[0] * spacing, pad_left + node[1] * spacing)
        for edge in edges
        for node in edge
    }

    for a, b in edges:
        r1, c1 = node_cells[a]
        r2, c2 = node_cells[b]
        if r1 == r2:
            lo, hi = sorted((c1, c2))
            grid[r1, lo : hi + 1] = GREY
        else:
            lo, hi = sorted((r1, r2))
            grid[lo : hi + 1, c1] = GREY

    return grid, node_cells


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return True
    start = tuple(map(int, coords[0]))
    seen = {start}
    q: deque[Cell] = deque([start])
    while q:
        r, c = q.popleft()
        for rr, cc in _pipe_neighbors(mask, r, c):
            if (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return len(seen) == int(mask.sum())


def _eligible_valves(pipe_grid: np.ndarray) -> list[ValveCandidate]:
    mask = pipe_grid == GREY
    candidates: list[ValveCandidate] = []
    h, w = pipe_grid.shape
    for r0, c0 in np.argwhere(mask):
        r = int(r0)
        c = int(c0)
        axis = _straight_axis(mask, r, c)
        if axis is None:
            continue
        side_cells: list[Cell] = []
        if axis == "h":
            for rr in (r - 1, r + 1):
                if 0 <= rr < h and int(pipe_grid[rr, c]) == BLACK:
                    side_cells.append((rr, c))
        else:
            for cc in (c - 1, c + 1):
                if 0 <= cc < w and int(pipe_grid[r, cc]) == BLACK:
                    side_cells.append((r, cc))
        if side_cells:
            candidates.append(((r, c), axis, tuple(side_cells)))
    return candidates


def _sample_non_adjacent(candidates: list[ValveCandidate], count: int) -> list[ValveCandidate] | None:
    if count == 0:
        return []
    for _ in range(100):
        pool = candidates[:]
        random.shuffle(pool)
        chosen: list[ValveCandidate] = []
        for cand in pool:
            (r, c), _axis, _side = cand
            if all(abs(r - rr) + abs(c - cc) > 1 for (rr, cc), _a, _s in chosen):
                chosen.append(cand)
                if len(chosen) == count:
                    return chosen
    return None


def _choose_valves(candidates: list[ValveCandidate]) -> tuple[list[ValveCandidate], list[bool]] | None:
    if not candidates:
        return [], []

    mode = random.choice(("none", "open_only", "closed_only", "mixed"))
    max_count = min(MAX_VALVES, len(candidates))

    if mode == "none":
        return [], []
    if mode == "mixed" and max_count < 2:
        mode = random.choice(("open_only", "closed_only"))

    if mode == "open_only":
        count = random.randint(1, min(3, max_count))
        chosen = _sample_non_adjacent(candidates, count)
        return (chosen, [False] * count) if chosen is not None else None
    if mode == "closed_only":
        count = random.randint(1, min(3, max_count))
        chosen = _sample_non_adjacent(candidates, count)
        return (chosen, [True] * count) if chosen is not None else None

    count = random.randint(2, max_count)
    chosen = _sample_non_adjacent(candidates, count)
    if chosen is None:
        return None
    statuses = [True, False]
    statuses.extend(random.choice((True, False)) for _ in range(count - 2))
    random.shuffle(statuses)
    return chosen, statuses


def _paint_sources_and_valves(
    pipe_grid: np.ndarray,
    leaf_cells: list[Cell],
    valves: list[ValveCandidate],
    closed_flags: list[bool],
) -> np.ndarray:
    grid = pipe_grid.copy()
    source_colors = random.sample(SOURCE_COLORS, len(leaf_cells))
    random.shuffle(leaf_cells)

    for cell, color in zip(leaf_cells, source_colors, strict=False):
        grid[cell] = color

    for (r, c), _axis, _side_cells in valves:
        grid[r, c] = MAROON

    for valve, is_closed in zip(valves, closed_flags, strict=False):
        if not is_closed:
            continue
        (r, c), _axis, side_cells = valve
        grid[r, c] = MAROON
        indicator_count = 1 if len(side_cells) == 1 else random.choice((1, 1, 2))
        for rr, cc in random.sample(list(side_cells), indicator_count):
            grid[rr, cc] = YELLOW

    return grid


def _endpoints(mask: np.ndarray) -> list[Cell]:
    leaves: list[Cell] = []
    for r0, c0 in np.argwhere(mask):
        r = int(r0)
        c = int(c0)
        if len(_pipe_neighbors(mask, r, c)) == 1:
            leaves.append((r, c))
    return leaves


def _valid_generated(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if int((grid == GREY).sum()) == 0:
        return False

    pipe = _pipe_mask(grid)
    if not bool(pipe.any()) or not _connected(pipe):
        return False

    try:
        mask, sources, _open_valves, closed_valves = _analyze_grid(grid)
    except AssertionError:
        return False

    if len(sources) < 2:
        return False
    if any(len(cells) != 1 for cells in sources.values()):
        return False

    for r0, c0 in np.argwhere(mask):
        r = int(r0)
        c = int(c0)
        color = int(grid[r, c])
        deg = len(_pipe_neighbors(mask, r, c))
        if deg == 1 and color in (GREY, MAROON):
            return False
        if color not in (GREY, MAROON) and deg != 1:
            return False
        if color == MAROON and _straight_axis(mask, r, c) is None:
            return False

    endpoint_set = set(_endpoints(mask))
    source_cells = {cells[0] for cells in sources.values()}
    if endpoint_set != source_cells:
        return False

    for r0, c0 in np.argwhere(grid == YELLOW):
        r = int(r0)
        c = int(c0)
        matched = False
        for rr, cc in _neighbors4(grid.shape, r, c):
            if int(grid[rr, cc]) != MAROON:
                continue
            axis = _straight_axis(mask, rr, cc)
            if axis == "h" and cc == c and abs(rr - r) == 1:
                matched = True
            if axis == "v" and rr == r and abs(cc - c) == 1:
                matched = True
        if not matched:
            return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if bool(np.any(out == YELLOW)):
        return False
    if any(int(out[r, c]) != color for color, cells in sources.items() for r, c in cells):
        return False
    if any(int(out[r, c]) != MAROON for r, c in closed_valves):
        return False

    return True


def _build_topology() -> tuple[np.ndarray, list[Cell]] | None:
    lh = random.randint(*LATTICE_H_RANGE)
    lw = random.randint(*LATTICE_W_RANGE)
    mode = random.choice(("path", "tree"))

    if mode == "path":
        max_nodes = min(PATH_NODE_RANGE[1], lh * lw)
        count = random.randint(PATH_NODE_RANGE[0], max_nodes)
        grown = _grow_path(lh, lw, count)
        if grown is None:
            return None
        _nodes, edges = grown
    else:
        max_nodes = min(TREE_NODE_RANGE[1], lh * lw)
        count = random.randint(TREE_NODE_RANGE[0], max_nodes)
        nodes = _grow_connected_nodes(lh, lw, count)
        if nodes is None:
            return None
        edges = _random_spanning_tree(nodes, lh, lw)
        if edges is None:
            return None
        deg = _node_degrees(edges)
        if max(deg.values()) < 3:
            return None

    deg = _node_degrees(edges)
    leaf_nodes = [node for node, d in deg.items() if d == 1]
    if not (2 <= len(leaf_nodes) <= len(SOURCE_COLORS)):
        return None

    spacing = random.choice(SPACING_CHOICES)
    pads = (
        random.choice(PAD_CHOICES),
        random.choice(PAD_CHOICES),
        random.choice(PAD_CHOICES),
        random.choice(PAD_CHOICES),
    )
    pipe_grid, _node_cells = _render_pipe(lh, lw, spacing, pads, edges)
    pipe_mask = pipe_grid == GREY
    leaf_cells = _endpoints(pipe_mask)
    if len(leaf_cells) != len(leaf_nodes):
        return None
    return pipe_grid, leaf_cells


def _gen_attempt() -> np.ndarray | None:
    built = _build_topology()
    if built is None:
        return None
    pipe_grid, leaf_cells = built

    candidates = _eligible_valves(pipe_grid)
    chosen = _choose_valves(candidates)
    if chosen is None:
        return None
    valves, closed_flags = chosen
    grid = _paint_sources_and_valves(pipe_grid, leaf_cells, valves, closed_flags)

    if not _valid_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pipe-network instance")
