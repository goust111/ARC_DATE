from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
WALL = 5
OBJECT_COLORS = (1, 2, 3, 4, 6)
SOURCE_SIZES = (2, 3)
GENERATION_RETRIES = 400
ROOM_ROWS_CHOICES = (2, 3, 4)
ROOM_COLS_CHOICES = (2, 3, 4, 5)
ROOM_SIDE_MIN = 4
ROOM_SIDE_MAX = 7


def _aligned_start(center: int, size: int, lo: int, hi: int) -> int:
    start = center - size // 2
    max_start = hi - size + 1
    if start < lo:
        return lo
    if start > max_start:
        return max_start
    return start


def _find_wall_rows(grid: np.ndarray) -> list[int]:
    counts = np.count_nonzero(grid == WALL, axis=1)
    rows = [int(r) for r, count in enumerate(counts) if int(count) > grid.shape[1] // 2]
    assert rows and rows[0] == 0 and rows[-1] == grid.shape[0] - 1
    assert all(b - a >= 2 for a, b in zip(rows, rows[1:]))
    return rows


def _find_wall_cols(grid: np.ndarray) -> list[int]:
    counts = np.count_nonzero(grid == WALL, axis=0)
    cols = [int(c) for c, count in enumerate(counts) if int(count) > grid.shape[0] // 2]
    assert cols and cols[0] == 0 and cols[-1] == grid.shape[1] - 1
    assert all(b - a >= 2 for a, b in zip(cols, cols[1:]))
    return cols


def _build_rooms(wall_rows: list[int], wall_cols: list[int]) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    rooms: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for rr in range(len(wall_rows) - 1):
        for cc in range(len(wall_cols) - 1):
            r0 = wall_rows[rr] + 1
            r1 = wall_rows[rr + 1] - 1
            c0 = wall_cols[cc] + 1
            c1 = wall_cols[cc + 1] - 1
            assert r0 <= r1 and c0 <= c1
            rooms[(rr, cc)] = (r0, r1, c0, c1)
    return rooms


def _add_edge(
    adjacency: dict[tuple[int, int], list[tuple[tuple[int, int], str, tuple[int, int]]]],
    room_a: tuple[int, int],
    room_b: tuple[int, int],
    door: tuple[int, int],
) -> None:
    ra, ca = room_a
    rb, cb = room_b
    if ra == rb and cb == ca + 1:
        adjacency[room_a].append((room_b, "right", door))
        adjacency[room_b].append((room_a, "left", door))
        return
    if ra == rb and ca == cb + 1:
        adjacency[room_a].append((room_b, "left", door))
        adjacency[room_b].append((room_a, "right", door))
        return
    if ca == cb and rb == ra + 1:
        adjacency[room_a].append((room_b, "down", door))
        adjacency[room_b].append((room_a, "up", door))
        return
    assert ca == cb and ra == rb + 1
    adjacency[room_a].append((room_b, "up", door))
    adjacency[room_b].append((room_a, "down", door))


def _parse_adjacency(
    grid: np.ndarray,
    wall_rows: list[int],
    wall_cols: list[int],
    rooms: dict[tuple[int, int], tuple[int, int, int, int]],
) -> dict[tuple[int, int], list[tuple[tuple[int, int], str, tuple[int, int]]]]:
    adjacency: dict[tuple[int, int], list[tuple[tuple[int, int], str, tuple[int, int]]]] = {
        room: [] for room in rooms
    }

    for rr in range(len(wall_rows) - 1):
        r0 = wall_rows[rr] + 1
        r1 = wall_rows[rr + 1] - 1
        for cc in range(len(wall_cols) - 2):
            wall_c = wall_cols[cc + 1]
            openings = [r for r in range(r0, r1 + 1) if int(grid[r, wall_c]) != WALL]
            assert len(openings) <= 1
            if openings:
                _add_edge(adjacency, (rr, cc), (rr, cc + 1), (openings[0], wall_c))

    for rr in range(len(wall_rows) - 2):
        wall_r = wall_rows[rr + 1]
        for cc in range(len(wall_cols) - 1):
            c0 = wall_cols[cc] + 1
            c1 = wall_cols[cc + 1] - 1
            openings = [c for c in range(c0, c1 + 1) if int(grid[wall_r, c]) != WALL]
            assert len(openings) <= 1
            if openings:
                _add_edge(adjacency, (rr, cc), (rr + 1, cc), (wall_r, openings[0]))

    return adjacency


def _parse_occupants(
    grid: np.ndarray, rooms: dict[tuple[int, int], tuple[int, int, int, int]]
) -> dict[tuple[int, int], dict[str, int]]:
    occupants: dict[tuple[int, int], dict[str, int]] = {}
    for room, (r0, r1, c0, c1) in rooms.items():
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        colors = [int(color) for color in np.unique(patch) if int(color) != BG]
        assert len(colors) <= 1
        if not colors:
            continue
        color = colors[0]
        mask = patch == color
        rows, cols = np.where(mask)
        top = int(rows.min())
        bottom = int(rows.max())
        left = int(cols.min())
        right = int(cols.max())
        height = bottom - top + 1
        width = right - left + 1
        assert int(mask.sum()) == height * width
        assert height == width
        occupants[room] = {"color": color, "size": height}
    assert occupants
    return occupants


def _draw_copy(
    out: np.ndarray,
    bounds: tuple[int, int, int, int],
    direction: str,
    door: tuple[int, int],
    color: int,
    size: int,
) -> None:
    r0, r1, c0, c1 = bounds
    if direction == "right":
        top = _aligned_start(door[0], size, r0, r1)
        left = c1 - size + 1
    elif direction == "left":
        top = _aligned_start(door[0], size, r0, r1)
        left = c0
    elif direction == "down":
        top = r1 - size + 1
        left = _aligned_start(door[1], size, c0, c1)
    else:
        assert direction == "up"
        top = r0
        left = _aligned_start(door[1], size, c0, c1)
    out[top : top + size, left : left + size] = color


def _propagate(
    grid: np.ndarray,
    rooms: dict[tuple[int, int], tuple[int, int, int, int]],
    adjacency: dict[tuple[int, int], list[tuple[tuple[int, int], str, tuple[int, int]]]],
    occupants: dict[tuple[int, int], dict[str, int]],
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    room_state = {room: {"color": info["color"], "size": info["size"]} for room, info in occupants.items()}
    queue = deque((room, occupants[room]["color"], occupants[room]["size"]) for room in sorted(occupants))

    while queue:
        room, color, size = queue.popleft()
        if size <= 1:
            continue
        next_size = size - 1
        for neighbor, direction, door in adjacency[room]:
            if neighbor in room_state:
                continue
            room_state[neighbor] = {"color": color, "size": next_size}
            _draw_copy(out, rooms[neighbor], direction, door, color, next_size)
            if next_size > 1:
                queue.append((neighbor, color, next_size))

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each room's colored square through doors into neighboring empty rooms,
    shrinking the square by one cell per step and placing each copy against the wall
    opposite the door it used.
    """
    g = np.asarray(grid, dtype=int)
    wall_rows = _find_wall_rows(g)
    wall_cols = _find_wall_cols(g)
    rooms = _build_rooms(wall_rows, wall_cols)
    adjacency = _parse_adjacency(g, wall_rows, wall_cols, rooms)
    occupants = _parse_occupants(g, rooms)
    return _propagate(g, rooms, adjacency, occupants)


def _wall_positions(room_sizes: list[int]) -> list[int]:
    positions = [0]
    cur = 0
    for size in room_sizes:
        cur += size + 1
        positions.append(cur)
    return positions


def _snake_path(n_rows: int, n_cols: int, rng: random.Random) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    if rng.random() < 0.5:
        for rr in range(n_rows):
            cols = range(n_cols) if rr % 2 == 0 else range(n_cols - 1, -1, -1)
            for cc in cols:
                path.append((rr, cc))
    else:
        for cc in range(n_cols):
            rows = range(n_rows) if cc % 2 == 0 else range(n_rows - 1, -1, -1)
            for rr in rows:
                path.append((rr, cc))

    if rng.random() < 0.5:
        path = path[::-1]
    if rng.random() < 0.5:
        path = [(n_rows - 1 - rr, cc) for rr, cc in path]
    if rng.random() < 0.5:
        path = [(rr, n_cols - 1 - cc) for rr, cc in path]
    return path


def _build_structure(
    room_heights: list[int], room_widths: list[int], path: list[tuple[int, int]], rng: random.Random
) -> tuple[
    np.ndarray,
    dict[tuple[int, int], tuple[int, int, int, int]],
    dict[tuple[int, int], list[tuple[tuple[int, int], str, tuple[int, int]]]],
]:
    wall_rows = _wall_positions(room_heights)
    wall_cols = _wall_positions(room_widths)
    rooms = _build_rooms(wall_rows, wall_cols)
    adjacency: dict[tuple[int, int], list[tuple[tuple[int, int], str, tuple[int, int]]]] = {
        room: [] for room in rooms
    }

    for room_a, room_b in zip(path, path[1:]):
        ra, ca = room_a
        rb, cb = room_b
        if ra == rb:
            row_lo = rooms[room_a][0]
            row_hi = rooms[room_a][1]
            door = (rng.randint(row_lo, row_hi), wall_cols[min(ca, cb) + 1])
        else:
            col_lo = rooms[room_a][2]
            col_hi = rooms[room_a][3]
            door = (wall_rows[min(ra, rb) + 1], rng.randint(col_lo, col_hi))
        _add_edge(adjacency, room_a, room_b, door)

    grid = np.full((wall_rows[-1] + 1, wall_cols[-1] + 1), BG, dtype=int)
    grid[wall_rows, :] = WALL
    grid[:, wall_cols] = WALL
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for room, edges in adjacency.items():
        for neighbor, _direction, door in edges:
            key = tuple(sorted((room, neighbor)))
            if key in seen:
                continue
            seen.add(key)
            grid[door] = BG

    return grid, rooms, adjacency


def _choose_sources(
    path: list[tuple[int, int]], rng: random.Random
) -> list[tuple[int, tuple[int, int], int]]:
    max_sources = 1
    if len(path) >= 7:
        max_sources = 2
    if len(path) >= 13:
        max_sources = 3
    target = rng.randint(1, max_sources)

    size_by_index: dict[int, int] = {}
    candidates = list(range(len(path)))
    rng.shuffle(candidates)

    if len(path) >= 3 and rng.random() < 0.5:
        forced = rng.randint(1, len(path) - 2)
        size_by_index[forced] = rng.choice(SOURCE_SIZES)
        candidates = [idx for idx in candidates if idx != forced]

    for idx in candidates:
        if len(size_by_index) >= target:
            break
        size = rng.choice(SOURCE_SIZES)
        radius = size - 1
        ok = True
        for other_idx, other_size in size_by_index.items():
            if abs(idx - other_idx) < radius + (other_size - 1) + 1:
                ok = False
                break
        if ok:
            size_by_index[idx] = size

    if not size_by_index:
        return []

    chosen = sorted(size_by_index.items())
    colors = rng.sample(list(OBJECT_COLORS), k=len(chosen))
    return [(idx, path[idx], colors[pos], size) for pos, (idx, size) in enumerate(chosen)]


def _place_sources(
    grid: np.ndarray,
    rooms: dict[tuple[int, int], tuple[int, int, int, int]],
    sources: list[tuple[int, tuple[int, int], int, int]],
    rng: random.Random,
) -> dict[tuple[int, int], dict[str, int]]:
    occupants: dict[tuple[int, int], dict[str, int]] = {}
    for _idx, room, color, size in sources:
        r0, r1, c0, c1 = rooms[room]
        top = rng.randint(r0, r1 - size + 1)
        left = rng.randint(c0, c1 - size + 1)
        grid[top : top + size, left : left + size] = color
        occupants[room] = {"color": color, "size": size}
    return occupants


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    n_rows = rng.choice(ROOM_ROWS_CHOICES)
    n_cols = rng.choice(ROOM_COLS_CHOICES)
    room_heights = [rng.randint(ROOM_SIDE_MIN, ROOM_SIDE_MAX) for _ in range(n_rows)]
    room_widths = [rng.randint(ROOM_SIDE_MIN, ROOM_SIDE_MAX) for _ in range(n_cols)]
    path = _snake_path(n_rows, n_cols, rng)

    grid, rooms, adjacency = _build_structure(room_heights, room_widths, path, rng)
    sources = _choose_sources(path, rng)
    if not sources:
        return None

    inp = np.array(grid, copy=True)
    occupants = _place_sources(inp, rooms, sources, rng)
    expected = _propagate(inp, rooms, adjacency, occupants)

    try:
        solved = main(inp)
    except Exception:
        return None
    if np.array_equal(solved, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GENERATION_RETRIES} attempts")
