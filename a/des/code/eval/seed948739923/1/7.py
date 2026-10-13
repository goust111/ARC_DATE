from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
WALL = 5
AZURE = 8
COLOR_ORDER = [2, 1, 3, 4, 7]  # Red, Blue, Green, Yellow, Orange

CELL_ROWS_RANGE = (4, 7)
CELL_COLS_RANGE = (4, 8)
MAX_PAIRS = 5
GEN_RETRIES = 400


Coord = tuple[int, int]
Node = tuple[int, int]


def _step(a: Coord, b: Coord) -> Coord:
    return (b[0] - a[0], b[1] - a[1])


def _is_horizontal(delta: Coord) -> bool:
    return delta[0] == 0 and abs(delta[1]) == 1


def _is_vertical(delta: Coord) -> bool:
    return delta[1] == 0 and abs(delta[0]) == 1


def _neighbors4(cell: Coord, h: int, w: int) -> list[Coord]:
    r, c = cell
    out: list[Coord] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _marker_positions(grid: np.ndarray) -> dict[int, list[Coord]]:
    markers: dict[int, list[Coord]] = {}
    for color in sorted(set(map(int, grid.ravel()))):
        if color in (BG, WALL):
            continue
        coords = [tuple(map(int, pt)) for pt in np.argwhere(grid == color)]
        assert len(coords) == 2, f"color {color} must appear exactly twice"
        markers[color] = coords
    return markers


def _bfs_path(walkable: np.ndarray, start: Coord, end: Coord) -> list[Coord]:
    h, w = walkable.shape
    q: deque[Coord] = deque([start])
    parent: dict[Coord, Coord | None] = {start: None}
    while q:
        cur = q.popleft()
        if cur == end:
            break
        for nxt in _neighbors4(cur, h, w):
            if not walkable[nxt] or nxt in parent:
                continue
            parent[nxt] = cur
            q.append(nxt)
    assert end in parent, "paired markers must be connected through the corridor graph"

    path: list[Coord] = []
    cur: Coord | None = end
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _row_prefix(path: list[Coord]) -> list[Coord]:
    row = path[0][0]
    prefix: list[Coord] = []
    for cell in path:
        if cell[0] != row:
            break
        prefix.append(cell)
    return prefix


def _col_suffix(path: list[Coord]) -> list[Coord]:
    col = path[-1][1]
    suffix: list[Coord] = []
    for cell in reversed(path):
        if cell[1] != col:
            break
        suffix.append(cell)
    suffix.reverse()
    return suffix


def _turn_cells(path: list[Coord]) -> list[Coord]:
    turns: list[Coord] = []
    for idx in range(1, len(path) - 1):
        if _step(path[idx - 1], path[idx]) != _step(path[idx], path[idx + 1]):
            turns.append(path[idx])
    return turns


def _orientation_score(path: list[Coord]) -> tuple[int, int, int, int]:
    if len(path) < 2:
        return (0, 0, 0, 0)
    first = _step(path[0], path[1])
    last = _step(path[-2], path[-1])
    return (
        int(_is_horizontal(first)),
        int(_is_vertical(last)),
        len(_row_prefix(path)),
        len(_col_suffix(path)),
    )


def _orient_path(path: list[Coord]) -> list[Coord]:
    rev = list(reversed(path))
    if _orientation_score(rev) > _orientation_score(path):
        return rev
    return path


def _paint_footprint(path: list[Coord]) -> set[Coord]:
    cells = set(_row_prefix(path))
    cells.update(_col_suffix(path))
    cells.update(_turn_cells(path))
    cells.discard(path[0])
    cells.discard(path[-1])
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-color marker pairs through the maze corridor, but only paint
    the horizontal prefix at the chosen start and the vertical suffix at the
    chosen end; every turn cell becomes azure.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    markers = _marker_positions(g)
    walkable = g != WALL
    out = g.copy()

    ordered_colors = [color for color in COLOR_ORDER if color in markers]
    ordered_colors.extend(sorted(color for color in markers if color not in COLOR_ORDER))

    for color in ordered_colors:
        start, end = markers[color]
        path = _orient_path(_bfs_path(walkable, start, end))

        for r, c in _row_prefix(path):
            if g[r, c] == BG:
                out[r, c] = color
        for r, c in _col_suffix(path):
            if g[r, c] == BG:
                out[r, c] = color
        for r, c in _turn_cells(path):
            if g[r, c] == BG:
                out[r, c] = AZURE

    for color, coords in markers.items():
        for r, c in coords:
            out[r, c] = color

    return out


def _cell_neighbors(node: Node, rows: int, cols: int) -> list[Node]:
    r, c = node
    out: list[Node] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < rows:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < cols:
        out.append((r, c + 1))
    return out


def _build_maze_tree(rows: int, cols: int, rng: random.Random) -> dict[Node, list[Node]]:
    adj = {(r, c): [] for r in range(rows) for c in range(cols)}
    start = (rng.randrange(rows), rng.randrange(cols))
    stack = [start]
    seen = {start}
    while stack:
        cur = stack[-1]
        choices = [nxt for nxt in _cell_neighbors(cur, rows, cols) if nxt not in seen]
        if not choices:
            stack.pop()
            continue
        nxt = rng.choice(choices)
        seen.add(nxt)
        adj[cur].append(nxt)
        adj[nxt].append(cur)
        stack.append(nxt)
    return adj


def _render_node(node: Node) -> Coord:
    return (2 * node[0] + 1, 2 * node[1] + 1)


def _render_maze(adj: dict[Node, list[Node]], rows: int, cols: int) -> np.ndarray:
    grid = np.full((2 * rows + 1, 2 * cols + 1), WALL, dtype=int)
    for node, neighbors in adj.items():
        rr, cc = _render_node(node)
        grid[rr, cc] = BG
        for nxt in neighbors:
            if nxt < node:
                continue
            nr, nc = _render_node(nxt)
            grid[(rr + nr) // 2, (cc + nc) // 2] = BG
    return grid


def _leaf_sets(adj: dict[Node, list[Node]]) -> tuple[list[Node], list[Node]]:
    horizontal: list[Node] = []
    vertical: list[Node] = []
    for node, neighbors in adj.items():
        if len(neighbors) != 1:
            continue
        dr, dc = _step(node, neighbors[0])
        if dr == 0:
            horizontal.append(node)
        else:
            vertical.append(node)
    return horizontal, vertical


def _tree_path(adj: dict[Node, list[Node]], start: Node, end: Node) -> list[Node]:
    q: deque[Node] = deque([start])
    parent: dict[Node, Node | None] = {start: None}
    while q:
        cur = q.popleft()
        if cur == end:
            break
        for nxt in adj[cur]:
            if nxt in parent:
                continue
            parent[nxt] = cur
            q.append(nxt)
    assert end in parent

    path: list[Node] = []
    cur: Node | None = end
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _expand_path(nodes: list[Node]) -> list[Coord]:
    render_path = [_render_node(nodes[0])]
    for nxt in nodes[1:]:
        nr, nc = _render_node(nxt)
        cr, cc = render_path[-1]
        render_path.append(((cr + nr) // 2, (cc + nc) // 2))
        render_path.append((nr, nc))
    return render_path


def _candidate_pairs(adj: dict[Node, list[Node]]) -> list[dict[str, object]]:
    horizontal_leaves, vertical_leaves = _leaf_sets(adj)
    candidates: list[dict[str, object]] = []
    for start in horizontal_leaves:
        for end in vertical_leaves:
            node_path = _tree_path(adj, start, end)
            render_path = _expand_path(node_path)
            if len(_turn_cells(render_path)) == 0:
                continue
            footprint = _paint_footprint(render_path)
            if len(footprint) < 2:
                continue
            if _orientation_score(render_path)[:2] != (1, 1):
                continue
            candidates.append(
                {
                    "start": start,
                    "end": end,
                    "path": render_path,
                    "footprint": footprint,
                }
            )
    return candidates


def _disjoint_leaves(a: dict[str, object], b: dict[str, object]) -> bool:
    return a["start"] != b["start"] and a["end"] != b["end"]


def _greedy_fill(
    candidates: list[dict[str, object]],
    selected: list[dict[str, object]],
    used_h: set[Node],
    used_v: set[Node],
    target: int,
    require_disjoint: bool,
    rng: random.Random,
) -> list[dict[str, object]] | None:
    pool = candidates[:]
    rng.shuffle(pool)
    result = list(selected)
    for cand in pool:
        start = cand["start"]
        end = cand["end"]
        if start in used_h or end in used_v:
            continue
        if require_disjoint and any(cand["footprint"] & other["footprint"] for other in result):
            continue
        result.append(cand)
        used_h = used_h | {start}
        used_v = used_v | {end}
        if len(result) == target:
            return result
    return None


def _select_candidates(
    candidates: list[dict[str, object]], n_pairs: int, require_overlap: bool, rng: random.Random
) -> list[dict[str, object]] | None:
    if n_pairs == 1:
        if not candidates:
            return None
        return [rng.choice(candidates)]

    if require_overlap:
        pair_options: list[tuple[dict[str, object], dict[str, object]]] = []
        for idx, first in enumerate(candidates):
            for second in candidates[idx + 1 :]:
                if not _disjoint_leaves(first, second):
                    continue
                if first["footprint"] & second["footprint"]:
                    pair_options.append((first, second))
        rng.shuffle(pair_options)
        for first, second in pair_options[:200]:
            chosen = [first, second]
            used_h = {first["start"], second["start"]}
            used_v = {first["end"], second["end"]}
            filled = _greedy_fill(candidates, chosen, used_h, used_v, n_pairs, False, rng)
            if filled is not None:
                return filled
        return None

    for _ in range(80):
        filled = _greedy_fill(candidates, [], set(), set(), n_pairs, True, rng)
        if filled is not None:
            return filled
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    rows = rng.randint(*CELL_ROWS_RANGE)
    cols = rng.randint(*CELL_COLS_RANGE)
    adj = _build_maze_tree(rows, cols, rng)
    candidates = _candidate_pairs(adj)
    if not candidates:
        return None

    horizontal_leaves, vertical_leaves = _leaf_sets(adj)
    max_pairs = min(MAX_PAIRS, len(horizontal_leaves), len(vertical_leaves))
    if max_pairs == 0:
        return None

    low_pairs = 1 if max_pairs == 1 else 2
    n_pairs = rng.randint(low_pairs, max_pairs)
    require_overlap = n_pairs >= 2 and rng.random() < 0.5
    selected = _select_candidates(candidates, n_pairs, require_overlap, rng)
    if selected is None:
        return None

    colors = COLOR_ORDER[:]
    rng.shuffle(colors)
    chosen_colors = colors[:n_pairs]

    grid = _render_maze(adj, rows, cols)
    for color, cand in zip(chosen_colors, selected):
        sr, sc = _render_node(cand["start"])
        er, ec = _render_node(cand["end"])
        grid[sr, sc] = color
        grid[er, ec] = color

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None

    if require_overlap:
        footprints = [cand["footprint"] for cand in selected]
        if not any(footprints[i] & footprints[j] for i in range(len(footprints)) for j in range(i + 1, len(footprints))):
            return None
    else:
        footprints = [cand["footprint"] for cand in selected]
        if any(footprints[i] & footprints[j] for i in range(len(footprints)) for j in range(i + 1, len(footprints))):
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid maze-corridor puzzle input")
