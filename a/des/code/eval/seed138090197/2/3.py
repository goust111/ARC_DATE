import random
from collections import Counter, deque

import numpy as np


BLACK = 0
WALL = 5
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]

ROOM_ROWS = [2, 3, 4]
ROOM_COLS = [2, 3, 4]
ROOM_SIZE_MIN = 3
ROOM_SIZE_MAX = 6
GEN_RETRIES = 300


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=int)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != -1:
                continue
            comp_id = len(components)
            labels[r, c] = comp_id
            queue = deque([(r, c)])
            coords = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = comp_id
                        queue.append((nr, nc))
            components.append(coords)
    return labels, components


def _classify_markers(grid: np.ndarray) -> tuple[list[dict], list[tuple[int, int, int]]]:
    h, w = grid.shape
    doors = []
    keys = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            color = int(grid[r, c])
            if color in (BLACK, WALL):
                continue
            vertical = (
                grid[r - 1, c] == WALL
                and grid[r + 1, c] == WALL
                and grid[r, c - 1] != WALL
                and grid[r, c + 1] != WALL
            )
            horizontal = (
                grid[r, c - 1] == WALL
                and grid[r, c + 1] == WALL
                and grid[r - 1, c] != WALL
                and grid[r + 1, c] != WALL
            )
            if vertical ^ horizontal:
                doors.append({"pos": (r, c), "color": color, "axis": "vertical" if vertical else "horizontal"})
            else:
                keys.append((r, c, color))
    return doors, keys


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    doors, keys = _classify_markers(g)
    assert doors and keys, "expected at least one door and one key"

    door_counts = Counter(door["color"] for door in doors)
    key_counts = Counter(color for _r, _c, color in keys)
    assert all(count == 1 for count in door_counts.values())
    assert all(count == 1 for count in key_counts.values())
    assert set(door_counts) == set(key_counts)

    closed_mask = g != WALL
    for door in doors:
        r, c = door["pos"]
        closed_mask[r, c] = False
    room_labels, _ = _label_components(closed_mask)

    key_room_by_color: dict[int, int] = {}
    key_positions = set()
    for r, c, color in keys:
        room_id = int(room_labels[r, c])
        assert room_id >= 0, "key must lie inside a room"
        key_room_by_color[color] = room_id
        key_positions.add((r, c))

    open_doors = set()
    for door in doors:
        r, c = door["pos"]
        if door["axis"] == "vertical":
            room_a = int(room_labels[r, c - 1])
            room_b = int(room_labels[r, c + 1])
        else:
            room_a = int(room_labels[r - 1, c])
            room_b = int(room_labels[r + 1, c])
        assert room_a >= 0 and room_b >= 0 and room_a != room_b
        key_room = key_room_by_color[door["color"]]
        assert key_room in (room_a, room_b), "door must touch the matching key room"
        open_doors.add((r, c))

    passable = g != WALL
    for door in doors:
        r, c = door["pos"]
        passable[r, c] = (r, c) in open_doors

    comp_labels, components = _label_components(passable)
    out = g.copy()
    for comp_id, coords in enumerate(components):
        colors = {int(g[r, c]) for r, c in coords if (r, c) in key_positions}
        assert len(colors) <= 1, "opened region must contain at most one key color"
        if not colors:
            continue
        fill = next(iter(colors))
        for r, c in coords:
            out[r, c] = fill
        assert np.all(comp_labels[[r for r, _ in coords], [c for _, c in coords]] == comp_id)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Open each color-matched door between rectangular rooms and repaint the merged
    region using the corresponding room's key color.
    """
    return _solve(grid)


def _sample_room_sizes(count: int) -> list[int]:
    return [random.randint(ROOM_SIZE_MIN, ROOM_SIZE_MAX) for _ in range(count)]


def _build_room_rects(row_heights: list[int], col_widths: list[int]) -> tuple[dict[int, tuple[int, int, int, int]], int, int]:
    rects = {}
    row_starts = []
    cur = 1
    for height in row_heights:
        row_starts.append(cur)
        cur += height + 1
    col_starts = []
    cur = 1
    for width in col_widths:
        col_starts.append(cur)
        cur += width + 1
    h = sum(row_heights) + len(row_heights) + 1
    w = sum(col_widths) + len(col_widths) + 1
    room_id = 0
    for rr, r0 in enumerate(row_starts):
        for cc, c0 in enumerate(col_starts):
            rects[room_id] = (r0, c0, r0 + row_heights[rr] - 1, c0 + col_widths[cc] - 1)
            room_id += 1
    return rects, h, w


def _adjacency_edges(n_rows: int, n_cols: int) -> list[tuple[int, int]]:
    edges = []
    for r in range(n_rows):
        for c in range(n_cols):
            room_id = r * n_cols + c
            if c + 1 < n_cols:
                edges.append((room_id, room_id + 1))
            if r + 1 < n_rows:
                edges.append((room_id, room_id + n_cols))
    return edges


def _best_random_matching(edges: list[tuple[int, int]], n_rooms: int) -> list[tuple[int, int]]:
    best: list[tuple[int, int]] = []
    for _ in range(32):
        used = [False] * n_rooms
        order = edges[:]
        random.shuffle(order)
        current = []
        for a, b in order:
            if used[a] or used[b]:
                continue
            used[a] = True
            used[b] = True
            current.append((a, b))
        if len(current) > len(best):
            best = current
    return best


def _door_position(rect_a: tuple[int, int, int, int], rect_b: tuple[int, int, int, int]) -> tuple[int, int]:
    ra0, ca0, ra1, ca1 = rect_a
    rb0, cb0, rb1, cb1 = rect_b
    if ra0 == rb0 and ra1 == rb1:
        wall_col = min(ca1, cb1) + 1
        return random.randint(ra0, ra1), wall_col
    wall_row = min(ra1, rb1) + 1
    return wall_row, random.randint(ca0, ca1)


def _key_position(rect: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = rect
    return random.randint(r0 + 1, r1 - 1), random.randint(c0 + 1, c1 - 1)


def _gen_attempt() -> np.ndarray | None:
    n_rows = random.choice(ROOM_ROWS)
    n_cols = random.choice(ROOM_COLS)
    row_heights = _sample_room_sizes(n_rows)
    col_widths = _sample_room_sizes(n_cols)
    rects, h, w = _build_room_rects(row_heights, col_widths)
    if not (10 <= h <= 30 and 10 <= w <= 30):
        return None

    n_rooms = n_rows * n_cols
    edges = _adjacency_edges(n_rows, n_cols)
    matching = _best_random_matching(edges, n_rooms)
    if not matching:
        return None

    pair_count = random.randint(1, min(len(matching), len(COLOR_POOL)))
    random.shuffle(matching)
    matching = matching[:pair_count]
    colors = random.sample(COLOR_POOL, pair_count)

    grid = np.full((h, w), WALL, dtype=int)
    for rect in rects.values():
        r0, c0, r1, c1 = rect
        grid[r0 : r1 + 1, c0 : c1 + 1] = BLACK

    used_key_rooms = set()
    for color, edge in zip(colors, matching):
        a, b = edge
        source, target = (a, b) if random.random() < 0.5 else (b, a)
        used_key_rooms.add(source)
        kr, kc = _key_position(rects[source])
        grid[kr, kc] = color
        dr, dc = _door_position(rects[a], rects[b])
        grid[dr, dc] = color

    try:
        out = _solve(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate input after {GEN_RETRIES} attempts")
