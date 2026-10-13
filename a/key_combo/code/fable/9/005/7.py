from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
WALL = 8
PALETTE = (1, 2, 3, 4, 5, 6, 7, 9)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(shape: tuple[int, int], r: int, c: int):
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if _in_bounds(shape, nr, nc):
            yield nr, nc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        q = deque([(r, c)])
        coords = []
        while q:
            cr, cc = q.popleft()
            coords.append((cr, cc))
            for nr, nc in _neighbors4(mask.shape, cr, cc):
                if mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        parts.append(np.array(coords, dtype=int))
    return parts


def _draw_line(mask: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> None:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        c_lo, c_hi = sorted((c0, c1))
        mask[r0, c_lo : c_hi + 1] = True
    else:
        r_lo, r_hi = sorted((r0, r1))
        mask[r_lo : r_hi + 1, c0] = True


def _wall_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _detect_legend(grid: np.ndarray, wall_color: int) -> tuple[tuple[int, int], tuple[int, int], int, int]:
    h, w = grid.shape
    specs = [
        ((0, 0), [(0, 1), (1, 0)]),
        ((0, w - 1), [(0, w - 2), (1, w - 1)]),
        ((h - 1, 0), [(h - 1, 1), (h - 2, 0)]),
        ((h - 1, w - 1), [(h - 1, w - 2), (h - 2, w - 1)]),
    ]

    candidates = []
    for corner, inward_options in specs:
        corner_color = int(grid[corner])
        if corner_color in (wall_color, BLACK):
            continue
        colored = [
            pos
            for pos in inward_options
            if _in_bounds(grid.shape, pos[0], pos[1]) and int(grid[pos]) not in (wall_color, BLACK)
        ]
        if len(colored) != 1:
            continue
        mate = colored[0]
        candidates.append((corner, mate, corner_color, int(grid[mate])))

    assert len(candidates) == 1, "expected exactly one corner legend"
    return candidates[0]


def _room_mask(traversable: np.ndarray) -> np.ndarray:
    room = np.zeros_like(traversable, dtype=bool)
    for r in range(traversable.shape[0] - 1):
        for c in range(traversable.shape[1] - 1):
            if bool(np.all(traversable[r : r + 2, c : c + 2])):
                room[r : r + 2, c : c + 2] = True
    return room


def _bfs_dist(mask: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    dist = np.full(mask.shape, -1, dtype=int)
    sr, sc = start
    dist[sr, sc] = 0
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(mask.shape, r, c):
            if mask[nr, nc] and dist[nr, nc] == -1:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    wall_color = _wall_color(g)
    legend_first_pos, legend_second_pos, first_color, second_color = _detect_legend(g, wall_color)

    first_coords = [tuple(map(int, rc)) for rc in np.argwhere(g == first_color)]
    second_coords = [tuple(map(int, rc)) for rc in np.argwhere(g == second_color)]
    assert len(second_coords) == 1 and second_coords[0] == legend_second_pos, "expected one second-color legend cell"
    seed_candidates = [rc for rc in first_coords if rc != legend_first_pos]
    assert len(seed_candidates) == 1, "expected one seed sharing the first legend color"
    seed = seed_candidates[0]

    traversable = g == BLACK
    traversable[seed] = True
    room = _room_mask(traversable)
    corridor = traversable & ~room
    assert corridor[seed], "seed must lie in a corridor"

    dist = _bfs_dist(corridor, seed)
    assert np.all(dist[corridor] >= 0), "corridor network must be connected from the seed"

    rooms = []
    for coords in _components(room):
        neighbors = set()
        for r, c in coords:
            for nr, nc in _neighbors4(g.shape, int(r), int(c)):
                if corridor[nr, nc]:
                    neighbors.add((nr, nc))
        assert len(neighbors) == 1, "each room must have exactly one corridor contact"
        hallway_cell = next(iter(neighbors))
        fill_color = first_color if (dist[hallway_cell] + 1) % 2 == 0 else second_color
        rooms.append((coords, fill_color, hallway_cell))

    return {
        "grid": g,
        "wall_color": wall_color,
        "seed": seed,
        "legend": (legend_first_pos, legend_second_pos),
        "colors": (first_color, second_color),
        "traversable": traversable,
        "corridor": corridor,
        "room": room,
        "rooms": rooms,
        "dist": dist,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each room by the parity color reached at its doorway while leaving the corridor wire invisible.

    Input structure:
    - A teal wall field contains black traversable space split into one-cell-wide corridors and larger rooms.
    - Each room contains at least one 2x2 traversable block and touches the corridor tree through exactly one doorway.
    - A corner legend is a two-cell ordered color pair, and one extra seed pixel of the first legend color sits on a corridor cell.

    Rule:
    - Treat the seed as step 0 and walk along the corridor tree, alternating between the two legend colors every move.
    - For each room, find the unique corridor cell beside its doorway; stepping one more move into the doorway determines that room's color.
    - Fill every cell of the room with that color, while keeping walls unchanged and preserving the seed, legend, and black corridor cells.

    Key insight:
    - The visible answer depends only on corridor-path parity from the seed to each room entrance, so rooms must be separated from one-cell corridors before counting steps.
    """
    info = _analyze(grid)
    out = info["grid"].copy()
    for coords, color, _hallway_cell in info["rooms"]:
        out[coords[:, 0], coords[:, 1]] = color
    return out


GEN_MARGIN = 4
GEN_SPACING = 5
GEN_COARSE_MIN = 3
GEN_COARSE_MAX = 5
GEN_NODE_MIN = 5
GEN_ROOM_MIN = 2
GEN_ROOM_MAX = 5
GEN_ROOM_SIDES = (3, 4)
GEN_ATTEMPTS = 500


def _coarse_neighbors(node: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = node
    out = []
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            out.append((nr, nc))
    return out


def _build_random_tree(rows: int, cols: int) -> tuple[set[tuple[int, int]], dict[tuple[int, int], tuple[int, int]]]:
    total = rows * cols
    target = random.randint(GEN_NODE_MIN, min(total, GEN_NODE_MIN + 4))
    start = (random.randrange(rows), random.randrange(cols))
    nodes = {start}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    frontier = [(start, nbr) for nbr in _coarse_neighbors(start, rows, cols)]

    while frontier and len(nodes) < target:
        src, dst = frontier.pop(random.randrange(len(frontier)))
        if dst in nodes:
            continue
        nodes.add(dst)
        parent[dst] = src
        for nbr in _coarse_neighbors(dst, rows, cols):
            if nbr not in nodes:
                frontier.append((dst, nbr))

    return nodes, parent


def _tree_adjacency(
    nodes: set[tuple[int, int]], parent: dict[tuple[int, int], tuple[int, int]]
) -> dict[tuple[int, int], set[tuple[int, int]]]:
    adj = {node: set() for node in nodes}
    for child, par in parent.items():
        adj[child].add(par)
        adj[par].add(child)
    return adj


def _node_depths(adj: dict[tuple[int, int], set[tuple[int, int]]]) -> dict[tuple[int, int], int]:
    root = next(iter(adj))
    depths = {root: 0}
    q = deque([root])
    while q:
        node = q.popleft()
        for nxt in adj[node]:
            if nxt not in depths:
                depths[nxt] = depths[node] + 1
                q.append(nxt)
    return depths


def _fine_shape(rows: int, cols: int) -> tuple[int, int]:
    return (
        2 * GEN_MARGIN + (rows - 1) * GEN_SPACING + 1,
        2 * GEN_MARGIN + (cols - 1) * GEN_SPACING + 1,
    )


def _fine_coord(node: tuple[int, int]) -> tuple[int, int]:
    return (
        GEN_MARGIN + node[0] * GEN_SPACING,
        GEN_MARGIN + node[1] * GEN_SPACING,
    )


def _room_candidates(
    grid: np.ndarray,
    leaf: tuple[int, int],
    direction: tuple[int, int],
) -> list[tuple[np.ndarray, tuple[int, int], tuple[int, int, int, int]]]:
    r, c = leaf
    dr, dc = direction
    candidates = []
    sizes = [(h, w) for h in GEN_ROOM_SIDES for w in GEN_ROOM_SIDES]
    random.shuffle(sizes)

    for h, w in sizes:
        if dr != 0:
            offsets = list(range(1, w - 1))
            random.shuffle(offsets)
            for door_offset in offsets:
                r0 = r + 1 if dr == 1 else r - h
                c0 = c - door_offset
                door = (r + 1, c) if dr == 1 else (r - 1, c)
                candidates.append((h, w, r0, c0, door))
        else:
            offsets = list(range(1, h - 1))
            random.shuffle(offsets)
            for door_offset in offsets:
                r0 = r - door_offset
                c0 = c + 1 if dc == 1 else c - w
                door = (r, c + 1) if dc == 1 else (r, c - 1)
                candidates.append((h, w, r0, c0, door))

    valid = []
    for h, w, r0, c0, door in candidates:
        r1 = r0 + h - 1
        c1 = c0 + w - 1
        if not (0 <= r0 and 0 <= c0 and r1 < grid.shape[0] and c1 < grid.shape[1]):
            continue
        if np.any(grid[r0 : r1 + 1, c0 : c1 + 1] != WALL):
            continue

        room_cells = {(rr, cc) for rr in range(r0, r1 + 1) for cc in range(c0, c1 + 1)}
        adjacency = set()
        for rr, cc in room_cells:
            for nr, nc in _neighbors4(grid.shape, rr, cc):
                if (nr, nc) not in room_cells and grid[nr, nc] == BLACK:
                    adjacency.add((nr, nc))
        if adjacency != {leaf}:
            continue

        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(grid.shape[0] - 1, r1 + 1)
        cc1 = min(grid.shape[1] - 1, c1 + 1)
        moat_ok = True
        for rr in range(rr0, rr1 + 1):
            for cc in range(cc0, cc1 + 1):
                if (rr, cc) in room_cells or (rr, cc) == leaf:
                    continue
                if grid[rr, cc] == BLACK:
                    moat_ok = False
                    break
            if not moat_ok:
                break
        if not moat_ok:
            continue

        coords = np.array(sorted(room_cells), dtype=int)
        valid.append((coords, door, (r0, c0, r1, c1)))
    return valid


def _render_structure() -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], list[np.ndarray]] | None:
    coarse_rows = random.randint(GEN_COARSE_MIN, GEN_COARSE_MAX)
    coarse_cols = random.randint(GEN_COARSE_MIN, GEN_COARSE_MAX)
    nodes, parent = _build_random_tree(coarse_rows, coarse_cols)
    adj = _tree_adjacency(nodes, parent)
    if len(nodes) < GEN_NODE_MIN:
        return None

    depths = _node_depths(adj)
    leaves = [node for node, nbrs in adj.items() if len(nbrs) == 1]
    if len(leaves) < GEN_ROOM_MIN:
        return None

    even_leaves = [node for node in leaves if depths[node] % 2 == 0]
    odd_leaves = [node for node in leaves if depths[node] % 2 == 1]
    if not even_leaves or not odd_leaves:
        return None

    max_rooms = min(GEN_ROOM_MAX, len(leaves))
    room_count = random.randint(GEN_ROOM_MIN, max_rooms)
    room_leaves = None
    for _ in range(40):
        chosen = random.sample(leaves, room_count)
        parities = {depths[node] % 2 for node in chosen}
        if parities == {0, 1}:
            room_leaves = chosen
            break
    if room_leaves is None:
        return None

    corridor = np.zeros(_fine_shape(coarse_rows, coarse_cols), dtype=bool)
    for node in nodes:
        corridor[_fine_coord(node)] = True
    for child, par in parent.items():
        _draw_line(corridor, _fine_coord(child), _fine_coord(par))

    grid = np.full(corridor.shape, WALL, dtype=int)
    grid[corridor] = BLACK

    room_coords = []
    room_entries = []
    for node in room_leaves:
        par = next(iter(adj[node]))
        direction = (node[0] - par[0], node[1] - par[1])
        leaf_rc = _fine_coord(node)
        candidates = _room_candidates(grid, leaf_rc, direction)
        if not candidates:
            return None
        coords, _door, _bbox = random.choice(candidates)
        grid[coords[:, 0], coords[:, 1]] = BLACK
        room_coords.append(coords)
        room_entries.append(leaf_rc)

    return grid, corridor, room_entries, room_coords


def _seed_candidates(corridor: np.ndarray, room_entries: list[tuple[int, int]]) -> list[tuple[int, int]]:
    candidates = []
    for seed in [tuple(map(int, rc)) for rc in np.argwhere(corridor)]:
        dist = _bfs_dist(corridor, seed)
        parities = {(dist[entry] + 1) % 2 for entry in room_entries}
        if len(parities) == 2:
            candidates.append(seed)
    return candidates


def _place_legend(grid: np.ndarray, first_color: int, second_color: int) -> tuple[tuple[int, int], tuple[int, int]] | None:
    h, w = grid.shape
    options = [
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
        ((0, w - 1), (0, w - 2)),
        ((0, w - 1), (1, w - 1)),
        ((h - 1, 0), (h - 1, 1)),
        ((h - 1, 0), (h - 2, 0)),
        ((h - 1, w - 1), (h - 1, w - 2)),
        ((h - 1, w - 1), (h - 2, w - 1)),
    ]
    random.shuffle(options)
    for first_pos, second_pos in options:
        if grid[first_pos] == WALL and grid[second_pos] == WALL:
            grid[first_pos] = first_color
            grid[second_pos] = second_color
            return first_pos, second_pos
    return None


def _generated_input_ok(grid: np.ndarray) -> bool:
    try:
        info = _analyze(grid)
    except Exception:
        return False
    if not (GEN_ROOM_MIN <= len(info["rooms"]) <= GEN_ROOM_MAX):
        return False
    if max(grid.shape) > 30:
        return False
    if len({color for _coords, color, _cell in info["rooms"]}) != 2:
        return False
    out = main(grid)
    return bool(np.any(out != grid))


def _attempt() -> np.ndarray | None:
    rendered = _render_structure()
    if rendered is None:
        return None
    grid, corridor, room_entries, _room_coords = rendered

    seeds = _seed_candidates(corridor, room_entries)
    if not seeds:
        return None
    seed = random.choice(seeds)

    first_color, second_color = random.sample(PALETTE, 2)
    legend = _place_legend(grid, first_color, second_color)
    if legend is None:
        return None
    grid[seed] = first_color

    if not _generated_input_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for fable/9/005/7")
