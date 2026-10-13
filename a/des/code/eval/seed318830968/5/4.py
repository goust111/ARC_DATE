import random
from collections import deque
from itertools import combinations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
CYAN = 8
MAROON = 9

GATE_COLORS = (BLUE, RED, GREEN)
KEY_FRAME_COLOR = CYAN
GEN_MAX_ATTEMPTS = 500


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _mask_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    return cells


def _find_key_region(grid: np.ndarray) -> tuple[tuple[int, int, int, int], set[int]]:
    h, w = grid.shape
    candidates: list[tuple[tuple[int, int, int, int], set[int]]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BLACK):
        for comp in _mask_components(grid == color):
            if not any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp):
                continue
            r0, c0, r1, c1 = _bbox(comp)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            if set(comp) != _perimeter_cells(r0, c0, r1, c1):
                continue
            interior = grid[r0 + 1 : r1, c0 + 1 : c1]
            filled = [int(v) for v in interior.ravel() if int(v) != BLACK]
            if not (1 <= len(filled) <= 2):
                continue
            if any(v not in GATE_COLORS for v in filled):
                continue
            if len(set(filled)) != len(filled):
                continue
            candidates.append(((r0, c0, r1, c1), set(filled)))
    assert len(candidates) == 1, "expected one edge key region"
    return candidates[0]


def _adjacent_wire(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    h, w = grid.shape
    wires = [(nr, nc) for nr, nc in _neighbors4(r, c, h, w) if int(grid[nr, nc]) == GREY]
    assert len(wires) == 1, "marker must touch exactly one grey wire end"
    return wires[0]


def _reachable_wires(grid: np.ndarray, active_colors: set[int], start_wire: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = grid.shape
    passable = {GREY, *active_colors}
    seen = {start_wire}
    q = deque([start_wire])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if (nr, nc) in seen:
                continue
            if int(grid[nr, nc]) not in passable:
                continue
            seen.add((nr, nc))
            q.append((nr, nc))
    return seen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Doors embedded in a grey wire network open only when their colors appear in the edge key box.
    Reachable maroon terminals turn yellow; unreachable ones disappear.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    _key_box, active_colors = _find_key_region(g)

    yellow_cells = [tuple(map(int, rc)) for rc in np.argwhere(g == YELLOW)]
    assert len(yellow_cells) == 1, "expected one yellow start marker"
    start_wire = _adjacent_wire(g, *yellow_cells[0])
    reachable = _reachable_wires(g, active_colors, start_wire)

    out = g.copy()
    for r, c in np.argwhere(g == MAROON):
        rr, cc = int(r), int(c)
        wire = _adjacent_wire(g, rr, cc)
        out[rr, cc] = YELLOW if wire in reachable else BLACK
    return out


def _boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _expand_box(box: tuple[int, int, int, int], margin: int, h: int, w: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return (
        max(0, r0 - margin),
        max(0, c0 - margin),
        min(h - 1, r1 + margin),
        min(w - 1, c1 + margin),
    )


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _place_key_region(grid: np.ndarray) -> tuple[tuple[int, int, int, int], set[int]]:
    h, w = grid.shape
    frame_h = random.choice([5, 5, 6])
    frame_w = random.choice([5, 6, 7])
    side = random.choice(["top", "bottom", "left", "right"])
    if side == "top":
        r0 = 0
        c0 = random.randint(0, w - frame_w)
    elif side == "bottom":
        r0 = h - frame_h
        c0 = random.randint(0, w - frame_w)
    elif side == "left":
        r0 = random.randint(0, h - frame_h)
        c0 = 0
    else:
        r0 = random.randint(0, h - frame_h)
        c0 = w - frame_w
    box = (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)
    _draw_frame(grid, box, KEY_FRAME_COLOR)

    key_count = 1 if random.random() < 0.5 else 2
    active_colors = set(random.choice(list(combinations(GATE_COLORS, key_count))))
    interior = [(r, c) for r in range(r0 + 1, r0 + frame_h - 1) for c in range(c0 + 1, c0 + frame_w - 1)]
    for (r, c), color in zip(random.sample(interior, key_count), sorted(active_colors)):
        grid[r, c] = color
    return box, active_colors


def _coarse_neighbors(node: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = node
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < rows:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < cols:
        out.append((r, c + 1))
    return out


def _edge_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (a, b) if a <= b else (b, a)


def _build_tree(rows: int, cols: int, n_nodes: int):
    root = (random.randrange(rows), random.randrange(cols))
    nodes = {root}
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    frontier = [(root, nb) for nb in _coarse_neighbors(root, rows, cols)]
    while frontier and len(nodes) < n_nodes:
        idx = random.randrange(len(frontier))
        parent, child = frontier.pop(idx)
        if child in nodes:
            continue
        nodes.add(child)
        edges.append((parent, child))
        for nb in _coarse_neighbors(child, rows, cols):
            if nb not in nodes:
                frontier.append((child, nb))
    if len(nodes) != n_nodes:
        return None

    adj = {node: [] for node in nodes}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    return nodes, edges, adj


def _choose_network_anchor(
    h: int,
    w: int,
    rows: int,
    cols: int,
    forbidden: tuple[int, int, int, int],
) -> tuple[int, int] | None:
    max_r0 = h - 3 * rows
    max_c0 = w - 3 * cols
    if max_r0 < 2 or max_c0 < 2:
        return None
    candidates = []
    for r0 in range(2, max_r0 + 1):
        for c0 in range(2, max_c0 + 1):
            box = (r0 - 1, c0 - 1, r0 + 3 * (rows - 1) + 1, c0 + 3 * (cols - 1) + 1)
            if not _boxes_intersect(box, forbidden):
                candidates.append((r0, c0))
    return random.choice(candidates) if candidates else None


def _choose_gate_colors(gate_count: int) -> list[int]:
    distinct_count = 2 if gate_count == 2 else random.choice([2, 3])
    palette = random.sample(GATE_COLORS, distinct_count)
    colors = list(palette)
    while len(colors) < gate_count:
        colors.append(random.choice(palette))
    random.shuffle(colors)
    return colors


def _path_stats_from_start(
    start: tuple[int, int],
    leaves: list[tuple[int, int]],
    adj: dict[tuple[int, int], list[tuple[int, int]]],
    edge_gate: dict[tuple[tuple[int, int], tuple[int, int]], int | None],
    active_colors: set[int],
) -> tuple[int, int, bool]:
    parent = {start: (None, None)}
    q = deque([start])
    while q:
        node = q.popleft()
        for nb in adj[node]:
            if nb in parent:
                continue
            parent[nb] = (node, edge_gate[_edge_key(node, nb)])
            q.append(nb)

    reachable = 0
    unreachable = 0
    reachable_uses_open_gate = False
    for leaf in leaves:
        if leaf == start:
            continue
        cur = leaf
        passable = True
        used_open_gate = False
        while cur != start:
            prev, gate_color = parent[cur]
            if gate_color is not None:
                if gate_color in active_colors:
                    used_open_gate = True
                else:
                    passable = False
            cur = prev
        if passable:
            reachable += 1
            reachable_uses_open_gate |= used_open_gate
        else:
            unreachable += 1
    return reachable, unreachable, reachable_uses_open_gate


def _choose_start_leaf(
    leaves: list[tuple[int, int]],
    adj: dict[tuple[int, int], list[tuple[int, int]]],
    edge_gate: dict[tuple[tuple[int, int], tuple[int, int]], int | None],
    active_colors: set[int],
) -> tuple[int, int] | None:
    candidates = []
    for leaf in leaves:
        reachable, unreachable, uses_open_gate = _path_stats_from_start(
            leaf, leaves, adj, edge_gate, active_colors
        )
        if reachable >= 1 and unreachable >= 1 and uses_open_gate:
            candidates.append(leaf)
    return random.choice(candidates) if candidates else None


def _node_cell(anchor: tuple[int, int], node: tuple[int, int]) -> tuple[int, int]:
    r0, c0 = anchor
    nr, nc = node
    return r0 + 3 * nr, c0 + 3 * nc


def _draw_network(
    grid: np.ndarray,
    anchor: tuple[int, int],
    nodes: set[tuple[int, int]],
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
    edge_gate: dict[tuple[tuple[int, int], tuple[int, int]], int | None],
) -> None:
    for node in nodes:
        r, c = _node_cell(anchor, node)
        grid[r, c] = GREY
    for a, b in edges:
        ra, ca = _node_cell(anchor, a)
        rb, cb = _node_cell(anchor, b)
        dr = 0 if ra == rb else (1 if rb > ra else -1)
        dc = 0 if ca == cb else (1 if cb > ca else -1)
        fill = edge_gate[_edge_key(a, b)]
        color = GREY if fill is None else fill
        for step in (1, 2):
            grid[ra + dr * step, ca + dc * step] = color


def _place_terminal_markers(
    grid: np.ndarray,
    anchor: tuple[int, int],
    leaves: list[tuple[int, int]],
    start_leaf: tuple[int, int],
    adj: dict[tuple[int, int], list[tuple[int, int]]],
) -> None:
    for leaf in leaves:
        parent = adj[leaf][0]
        dr = leaf[0] - parent[0]
        dc = leaf[1] - parent[1]
        r, c = _node_cell(anchor, leaf)
        mr, mc = r + dr, c + dc
        assert int(grid[mr, mc]) == BLACK
        grid[mr, mc] = YELLOW if leaf == start_leaf else MAROON


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(19, 29)
    w = random.randint(19, 29)
    grid = np.full((h, w), BLACK, dtype=int)

    key_box, active_colors = _place_key_region(grid)
    forbidden = _expand_box(key_box, 1, h, w)

    max_rows = min(5, (h - 4) // 3)
    max_cols = min(6, (w - 4) // 3)
    if max_rows < 3 or max_cols < 3:
        return None
    rows = random.randint(3, max_rows)
    cols = random.randint(3, max_cols)

    anchor = _choose_network_anchor(h, w, rows, cols, forbidden)
    if anchor is None:
        return None

    max_nodes = min(rows * cols, 11)
    if max_nodes < 6:
        return None
    n_nodes = random.randint(6, max_nodes)
    built = _build_tree(rows, cols, n_nodes)
    if built is None:
        return None
    nodes, edges, adj = built

    degrees = {node: len(adj[node]) for node in nodes}
    leaves = [node for node, deg in degrees.items() if deg == 1]
    if not (3 <= len(leaves) <= 5):
        return None
    if max(degrees.values()) < 3:
        return None

    gate_count = random.randint(2, min(6, len(edges) - 1))
    gated_edges = set(random.sample([_edge_key(a, b) for a, b in edges], gate_count))
    gate_colors = _choose_gate_colors(gate_count)
    edge_gate: dict[tuple[tuple[int, int], tuple[int, int]], int | None] = {}
    for a, b in edges:
        edge_gate[_edge_key(a, b)] = None
    for edge, color in zip(random.sample(list(gated_edges), len(gated_edges)), gate_colors):
        edge_gate[edge] = color

    used_gate_colors = {color for color in edge_gate.values() if color is not None}
    valid_active_sets = [
        set(combo)
        for combo in combinations(GATE_COLORS, len(active_colors))
        if used_gate_colors & set(combo) and used_gate_colors - set(combo)
    ]
    if not valid_active_sets:
        return None
    active_colors = random.choice(valid_active_sets)

    grid.fill(BLACK)
    _draw_frame(grid, key_box, KEY_FRAME_COLOR)
    r0, c0, r1, c1 = key_box
    interior = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    for (r, c), color in zip(random.sample(interior, len(active_colors)), sorted(active_colors)):
        grid[r, c] = color

    start_leaf = _choose_start_leaf(leaves, adj, edge_gate, active_colors)
    if start_leaf is None:
        return None

    _draw_network(grid, anchor, nodes, edges, edge_gate)
    _place_terminal_markers(grid, anchor, leaves, start_leaf, adj)

    out = main(grid)
    terminals = [tuple(map(int, rc)) for rc in np.argwhere(grid == MAROON)]
    reached = sum(int(out[r, c]) == YELLOW for r, c in terminals)
    removed = sum(int(out[r, c]) == BLACK for r, c in terminals)
    if reached < 1 or removed < 1:
        return None
    if not np.any(out != grid):
        return None
    if not (used_gate_colors & active_colors):
        return None
    if not (used_gate_colors - active_colors):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gated-network ARC input")
