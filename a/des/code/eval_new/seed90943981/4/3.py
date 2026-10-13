from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


BLACK = 0
BLUE = 1
DOOR_COLORS = (2, 3, 4, 5, 6, 7, 8, 9)
ROOM_GRID_OPTIONS = (
    (2, 3),
    (3, 2),
    (2, 4),
    (4, 2),
    (3, 3),
    (3, 4),
    (4, 3),
)
ROOM_SIZE_MIN = 2
ROOM_SIZE_MAX = 5
GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if seen[nr, nc] or not mask[nr, nc]:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _wall_color(grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in grid.ravel() if int(v) != BLACK)
    assert counts, "expected walls and/or doors"
    return max(counts, key=counts.get)


def _extract_rooms(grid: np.ndarray, bg: int) -> tuple[list[dict], np.ndarray]:
    h, w = grid.shape
    rooms: list[dict] = []
    owner = np.full((h, w), -1, dtype=int)
    for coords in _components(grid == bg):
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in coords):
            continue
        r0, c0, r1, c1 = _bbox(coords)
        assert len(coords) == (r1 - r0 + 1) * (c1 - c0 + 1), "rooms must be solid rectangles"
        room_id = len(rooms)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                assert int(grid[r, c]) == bg
                owner[r, c] = room_id
        rooms.append({"id": room_id, "bbox": (r0, c0, r1, c1), "coords": coords})
    assert rooms, "expected at least one enclosed room"
    return rooms, owner


def _door_pair(grid: np.ndarray, owner: np.ndarray, r: int, c: int) -> tuple[int, int]:
    h, w = grid.shape
    left = int(owner[r, c - 1]) if c > 0 and int(grid[r, c - 1]) == BLACK else -1
    right = int(owner[r, c + 1]) if c + 1 < w and int(grid[r, c + 1]) == BLACK else -1
    up = int(owner[r - 1, c]) if r > 0 and int(grid[r - 1, c]) == BLACK else -1
    down = int(owner[r + 1, c]) if r + 1 < h and int(grid[r + 1, c]) == BLACK else -1

    candidates = []
    if left != -1 and right != -1 and left != right:
        candidates.append((left, right))
    if up != -1 and down != -1 and up != down:
        candidates.append((up, down))
    assert len(candidates) == 1, "door must separate exactly one room pair"
    a, b = candidates[0]
    return (a, b) if a < b else (b, a)


def _analyze(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = BLACK
    wall = _wall_color(g)
    rooms, owner = _extract_rooms(g, bg)

    door_edges: list[tuple[int, int, int, tuple[int, int]]] = []
    first_pos: dict[int, tuple[int, int]] = {}
    for r, c in np.argwhere((g != bg) & (g != wall)):
        rr = int(r)
        cc = int(c)
        color = int(g[rr, cc])
        a, b = _door_pair(g, owner, rr, cc)
        door_edges.append((a, b, color, (rr, cc)))
        if color not in first_pos or (rr, cc) < first_pos[color]:
            first_pos[color] = (rr, cc)
    return g, wall, rooms, owner, door_edges, first_pos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge wall-separated rectangular rooms by door-color connectivity and fill each room with its winning door color.

    Input structure:
    - Black background cells form enclosed rectangular rooms separated by one-pixel walls.
    - One wall color dominates the non-black pixels and acts as the permanent room boundary.
    - Some shared walls contain a single non-black, non-wall door pixel; different door colors define different connection families.

    Rule:
    - Identify every enclosed rectangular Black room and every colored door that links exactly two rooms across a wall.
    - For each door color, treat all rooms connected through doors of that same color as one connectivity graph, while keeping different colors independent.
    - Fill each room interior with the door color(s) of the graph(s) it belongs to; if a room belongs to multiple colors, choose the globally rarest of those door colors. Rooms with no doors stay Black, and both walls and door pixels remain unchanged.

    Key insight:
    The doors do not just mark adjacent rooms; each color induces its own transitive room graph, and rooms touched by multiple graphs resolve the conflict by the rarest visible door color.
    """
    g, _wall, rooms, _owner, door_edges, first_pos = _analyze(grid)
    out = g.copy()
    if not door_edges:
        return out

    graphs: dict[int, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    door_counts = Counter()
    for a, b, color, _pos in door_edges:
        graphs[color][a].add(b)
        graphs[color][b].add(a)
        door_counts[color] += 1

    room_colors = [set() for _ in rooms]
    for color, graph in graphs.items():
        seen: set[int] = set()
        for start in graph:
            if start in seen:
                continue
            stack = [start]
            component: set[int] = set()
            seen.add(start)
            while stack:
                room_id = stack.pop()
                component.add(room_id)
                for nxt in graph[room_id]:
                    if nxt in seen:
                        continue
                    seen.add(nxt)
                    stack.append(nxt)
            for room_id in component:
                room_colors[room_id].add(color)

    for room, colors in zip(rooms, room_colors, strict=False):
        if not colors:
            continue
        chosen = min(
            colors,
            key=lambda color: (
                int(door_counts[color]),
                int(first_pos[color][0]),
                int(first_pos[color][1]),
            ),
        )
        for r, c in room["coords"]:
            out[r, c] = int(chosen)
    return out


def _room_positions(
    n_rows: int, n_cols: int, heights: list[int], widths: list[int]
) -> tuple[np.ndarray, dict[tuple[int, int], dict], dict[tuple[int, int], list[tuple[int, int]]]]:
    h = sum(heights) + n_rows + 1
    w = sum(widths) + n_cols + 1
    grid = np.full((h, w), BLUE, dtype=int)
    rooms: dict[tuple[int, int], dict] = {}

    r0 = 1
    for i, room_h in enumerate(heights):
        c0 = 1
        for j, room_w in enumerate(widths):
            r1 = r0 + room_h - 1
            c1 = c0 + room_w - 1
            grid[r0 : r1 + 1, c0 : c1 + 1] = BLACK
            rooms[(i, j)] = {"pos": (i, j), "bbox": (r0, c0, r1, c1)}
            c0 = c1 + 2
        r0 = r1 + 2

    neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for i in range(n_rows):
        for j in range(n_cols):
            pos = (i, j)
            cur: list[tuple[int, int]] = []
            if i > 0:
                cur.append((i - 1, j))
            if i + 1 < n_rows:
                cur.append((i + 1, j))
            if j > 0:
                cur.append((i, j - 1))
            if j + 1 < n_cols:
                cur.append((i, j + 1))
            neighbors[pos] = cur
    return grid, rooms, neighbors


def _edge_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (a, b) if a < b else (b, a)


def _paint_door(grid: np.ndarray, room_a: dict, room_b: dict, color: int) -> None:
    (ar0, ac0, ar1, ac1) = room_a["bbox"]
    (br0, bc0, br1, bc1) = room_b["bbox"]
    if ar0 == br0:
        if ac0 < bc0:
            door_col = ac1 + 1
            door_row = random.randint(ar0, ar1)
        else:
            door_col = bc1 + 1
            door_row = random.randint(br0, br1)
        grid[door_row, door_col] = color
        return

    if ar0 < br0:
        door_row = ar1 + 1
        door_col = random.randint(ac0, ac1)
    else:
        door_row = br1 + 1
        door_col = random.randint(bc0, bc1)
    grid[door_row, door_col] = color


def _pattern_choices(
    positions: list[tuple[int, int]], neighbors: dict[tuple[int, int], list[tuple[int, int]]]
) -> list[tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int], list[tuple[int, int]]]]:
    choices = []
    pos_set = set(positions)
    for hub in positions:
        i, j = hub
        if (i, j - 1) in pos_set and (i, j + 1) in pos_set:
            orth = [p for p in neighbors[hub] if p[0] != i]
            for side in orth:
                used = {hub, (i, j - 1), (i, j + 1), side}
                leftovers = [p for p in positions if p not in used]
                if leftovers:
                    choices.append((hub, (i, j - 1), (i, j + 1), side, leftovers))
        if (i - 1, j) in pos_set and (i + 1, j) in pos_set:
            orth = [p for p in neighbors[hub] if p[1] != j]
            for side in orth:
                used = {hub, (i - 1, j), (i + 1, j), side}
                leftovers = [p for p in positions if p not in used]
                if leftovers:
                    choices.append((hub, (i - 1, j), (i + 1, j), side, leftovers))
    return choices


def _sample_room_grid() -> tuple[int, int, list[int], list[int]]:
    n_rows, n_cols = random.choice(ROOM_GRID_OPTIONS)
    heights = [random.randint(ROOM_SIZE_MIN, ROOM_SIZE_MAX) for _ in range(n_rows)]
    widths = [random.randint(ROOM_SIZE_MIN, ROOM_SIZE_MAX) for _ in range(n_cols)]
    return n_rows, n_cols, heights, widths


def _gen_attempt() -> np.ndarray | None:
    n_rows, n_cols, heights, widths = _sample_room_grid()
    grid, rooms, neighbors = _room_positions(n_rows, n_cols, heights, widths)
    positions = list(rooms)
    choices = _pattern_choices(positions, neighbors)
    if not choices:
        return None

    hub, left_room, right_room, side_room, leftovers = random.choice(choices)
    isolated = random.choice(leftovers)
    base_rooms = {hub, left_room, right_room, side_room, isolated}

    color_pool = list(DOOR_COLORS)
    random.shuffle(color_pool)
    chain_color = color_pool.pop()
    rare_color = color_pool.pop()

    assigned: dict[tuple[tuple[int, int], tuple[int, int]], int] = {
        _edge_key(hub, left_room): chain_color,
        _edge_key(hub, right_room): chain_color,
        _edge_key(hub, side_room): rare_color,
    }

    mode = random.choice(("none", "chain_extra", "extra_group"))
    if mode == "chain_extra":
        remaining_edges = [
            _edge_key(pos, nei)
            for pos in positions
            for nei in neighbors[pos]
            if pos < nei and isolated not in (pos, nei) and _edge_key(pos, nei) not in assigned
        ]
        if remaining_edges:
            n_extra = random.randint(1, min(2, len(remaining_edges)))
            for edge in random.sample(remaining_edges, n_extra):
                assigned[edge] = chain_color
    elif mode == "extra_group" and color_pool:
        untouched = [pos for pos in positions if pos not in base_rooms]
        candidates = []
        for center in untouched:
            free_neighbors = [
                nei
                for nei in neighbors[center]
                if nei in untouched and _edge_key(center, nei) not in assigned
            ]
            if len(free_neighbors) >= 2:
                candidates.append((center, free_neighbors))
        if candidates:
            center, free_neighbors = random.choice(candidates)
            extra_color = color_pool.pop()
            for nei in random.sample(free_neighbors, 2):
                assigned[_edge_key(center, nei)] = extra_color

    for (a, b), color in assigned.items():
        _paint_door(grid, rooms[a], rooms[b], color)

    out = main(grid)
    assert out.shape == grid.shape
    assert np.any(out != grid), "transformation should be non-trivial"

    def room_fill(pos: tuple[int, int]) -> int:
        r0, c0, _r1, _c1 = rooms[pos]["bbox"]
        return int(out[r0, c0])

    if room_fill(left_room) != chain_color:
        return None
    if room_fill(right_room) != chain_color:
        return None
    if room_fill(hub) != rare_color:
        return None
    if room_fill(side_room) != rare_color:
        return None
    if room_fill(isolated) != BLACK:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-door puzzle")
