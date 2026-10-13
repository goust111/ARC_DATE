from __future__ import annotations

from collections import deque

import numpy as np


BG = 8
WALL = 9
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_corner_pattern(grid: np.ndarray, corner: str) -> np.ndarray:
    if corner == "tl":
        zone = grid[:3, :3]
    elif corner == "tr":
        zone = grid[:3, -3:]
    elif corner == "bl":
        zone = grid[-3:, :3]
    elif corner == "br":
        zone = grid[-3:, -3:]
    else:
        raise ValueError(f"unknown corner {corner}")

    mask = (zone != BG) & (zone != WALL)
    b = _bbox(mask)
    assert b is not None, f"missing reference pattern in {corner}"
    r0, c0, r1, c1 = b
    return zone[r0 : r1 + 1, c0 : c1 + 1].copy()


def _tile_patch(patch: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    ph, pw = map(int, patch.shape)
    h, w = map(int, shape)
    reps_r = (h + ph - 1) // ph
    reps_c = (w + pw - 1) // pw
    tiled = np.tile(patch, (reps_r, reps_c))
    return tiled[:h, :w]


def _room_gap_direction(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> str | None:
    if grid[r0, c0] != WALL or grid[r0, c1] != WALL or grid[r1, c0] != WALL or grid[r1, c1] != WALL:
        return None
    if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == WALL):
        return None

    top = grid[r0, c0 : c1 + 1]
    bottom = grid[r1, c0 : c1 + 1]
    left = grid[r0 : r1 + 1, c0]
    right = grid[r0 : r1 + 1, c1]

    for side in (top, bottom, left, right):
        if np.any((side != WALL) & (side != BG)):
            return None

    missing_top = np.flatnonzero(top != WALL)
    missing_bottom = np.flatnonzero(bottom != WALL)
    missing_left = np.flatnonzero(left != WALL)
    missing_right = np.flatnonzero(right != WALL)
    total_missing = (
        len(missing_top) + len(missing_bottom) + len(missing_left) + len(missing_right)
    )
    if total_missing != 1:
        return None

    if len(missing_top) == 1 and 0 < int(missing_top[0]) < c1 - c0:
        return "up"
    if len(missing_bottom) == 1 and 0 < int(missing_bottom[0]) < c1 - c0:
        return "down"
    if len(missing_left) == 1 and 0 < int(missing_left[0]) < r1 - r0:
        return "left"
    if len(missing_right) == 1 and 0 < int(missing_right[0]) < r1 - r0:
        return "right"
    return None


def _find_rooms(grid: np.ndarray) -> list[tuple[int, int, int, int, str]]:
    h, w = map(int, grid.shape)
    rooms: list[tuple[int, int, int, int, str]] = []
    for r0 in range(h - 2):
        for r1 in range(r0 + 2, h):
            for c0 in range(w - 2):
                for c1 in range(c0 + 2, w):
                    direction = _room_gap_direction(grid, r0, c0, r1, c1)
                    if direction is None:
                        continue
                    rooms.append((r0, c0, r1, c1, direction))
    return rooms


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify rectangular rooms with one wall gap and fill each interior by tiling
    the corner reference pattern associated with the gap direction.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    patterns = {
        "up": _extract_corner_pattern(g, "tl"),
        "right": _extract_corner_pattern(g, "tr"),
        "left": _extract_corner_pattern(g, "bl"),
        "down": _extract_corner_pattern(g, "br"),
    }

    for r0, c0, r1, c1, direction in _find_rooms(g):
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[:, :] = _tile_patch(patterns[direction], interior.shape)

    return out


def _random_pattern(rng: np.random.Generator, seen: set[bytes]) -> np.ndarray:
    colors = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=int)
    for _ in range(200):
        h = int(rng.integers(2, 4))
        w = int(rng.integers(2, 4))
        n_colors = int(rng.integers(2, min(4, h * w) + 1))
        palette = rng.choice(colors, size=n_colors, replace=False)
        patch = rng.choice(palette, size=(h, w), replace=True).astype(int)
        if len(set(map(int, patch.ravel()))) < 2:
            continue
        key = patch.tobytes() + bytes([h, w])
        if key in seen:
            continue
        seen.add(key)
        return patch
    raise RuntimeError("failed to sample distinct reference pattern")


def _room_wall_cells(room: dict[str, object]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = room["bbox"]
    gap = room["gap"]
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    uniq = []
    seen = set()
    for cell in cells:
        if cell == gap or cell in seen:
            continue
        uniq.append(cell)
        seen.add(cell)
    return uniq


def _draw_room(grid: np.ndarray, room: dict[str, object]) -> None:
    r0, c0, r1, c1 = room["bbox"]
    grid[r0, c0 : c1 + 1] = WALL
    grid[r1, c0 : c1 + 1] = WALL
    grid[r0 : r1 + 1, c0] = WALL
    grid[r0 : r1 + 1, c1] = WALL
    gr, gc = room["gap"]
    grid[gr, gc] = BG


def _place_corner_patterns(grid: np.ndarray, patterns: dict[str, np.ndarray]) -> None:
    tl = patterns["up"]
    tr = patterns["right"]
    bl = patterns["left"]
    br = patterns["down"]
    grid[: tl.shape[0], : tl.shape[1]] = tl
    grid[: tr.shape[0], -tr.shape[1] :] = tr
    grid[-bl.shape[0] :, : bl.shape[1]] = bl
    grid[-br.shape[0] :, -br.shape[1] :] = br


def _inflate_bbox(bbox: tuple[int, int, int, int], h: int, w: int, pad: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return (
        max(0, r0 - pad),
        max(0, c0 - pad),
        min(h - 1, r1 + pad),
        min(w - 1, c1 + pad),
    )


def _sample_room(
    rng: np.random.Generator,
    blocked: np.ndarray,
    shape: tuple[int, int],
    direction: str,
) -> dict[str, object] | None:
    h, w = map(int, shape)
    for _ in range(120):
        room_h = int(rng.integers(4, min(10, h - 7) + 1))
        room_w = int(rng.integers(4, min(10, w - 7) + 1))

        r0_min, r0_max = 4, h - room_h - 4
        c0_min, c0_max = 4, w - room_w - 4
        if r0_min > r0_max or c0_min > c0_max:
            return None

        r0 = int(rng.integers(r0_min, r0_max + 1))
        c0 = int(rng.integers(c0_min, c0_max + 1))
        r1 = r0 + room_h - 1
        c1 = c0 + room_w - 1
        ir0, ic0, ir1, ic1 = _inflate_bbox((r0, c0, r1, c1), h, w, 1)
        if blocked[ir0 : ir1 + 1, ic0 : ic1 + 1].any():
            continue

        if direction == "up":
            gap = (r0, int(rng.integers(c0 + 1, c1)))
        elif direction == "down":
            gap = (r1, int(rng.integers(c0 + 1, c1)))
        elif direction == "left":
            gap = (int(rng.integers(r0 + 1, r1)), c0)
        else:
            gap = (int(rng.integers(r0 + 1, r1)), c1)

        return {"bbox": (r0, c0, r1, c1), "gap": gap, "direction": direction}
    return None


def _bfs_path(
    grid: np.ndarray,
    blocked: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    h, w = map(int, grid.shape)
    dirs = list(DIRS4)
    rng.shuffle(dirs)

    q: deque[tuple[int, int]] = deque([start])
    prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            break
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if nxt in prev:
                continue
            if nxt != goal and blocked[nr, nc]:
                continue
            if nxt != goal and int(grid[nr, nc]) not in (BG, WALL):
                continue
            prev[nxt] = (r, c)
            q.append(nxt)

    if goal not in prev:
        return None

    path = []
    cur: tuple[int, int] | None = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _connect_rooms(grid: np.ndarray, rooms: list[dict[str, object]], blocked: np.ndarray, rng: np.random.Generator) -> bool:
    if len(rooms) < 2:
        return True

    anchors = [_room_wall_cells(room) for room in rooms]
    for idx in range(1, len(rooms)):
        connected = False
        for _ in range(30):
            start = anchors[0][int(rng.integers(len(anchors[0])))]
            goal = anchors[idx][int(rng.integers(len(anchors[idx])))]
            path = _bfs_path(grid, blocked, start, goal, rng)
            if path is None:
                continue
            for r, c in path:
                grid[r, c] = WALL
            connected = True
            break
        if not connected:
            return False
    return True


def _expected_output(grid: np.ndarray, rooms: list[dict[str, object]], patterns: dict[str, np.ndarray]) -> np.ndarray:
    out = grid.copy()
    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]
        patch = patterns[room["direction"]]
        out[r0 + 1 : r1, c0 + 1 : c1] = _tile_patch(patch, (r1 - r0 - 1, c1 - c0 - 1))
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    h = int(rng.integers(18, 29))
    w = int(rng.integers(18, 29))

    patterns: dict[str, np.ndarray] = {}
    seen: set[bytes] = set()
    for direction in ("up", "right", "left", "down"):
        patterns[direction] = _random_pattern(rng, seen)

    grid = np.full((h, w), BG, dtype=int)
    _place_corner_patterns(grid, patterns)

    blocked = np.zeros((h, w), dtype=bool)
    blocked[:3, :3] = True
    blocked[:3, -3:] = True
    blocked[-3:, :3] = True
    blocked[-3:, -3:] = True

    room_count = int(rng.integers(2, 6))
    order = ["up", "right", "left", "down"]
    rng.shuffle(order)
    directions = order[: min(room_count, 4)]
    while len(directions) < room_count:
        directions.append(order[int(rng.integers(4))])

    rooms: list[dict[str, object]] = []
    for direction in directions:
        room = _sample_room(rng, blocked, (h, w), direction)
        if room is None:
            return None
        rooms.append(room)
        r0, c0, r1, c1 = room["bbox"]
        ir0, ic0, ir1, ic1 = _inflate_bbox((r0, c0, r1, c1), h, w, 1)
        blocked[ir0 : ir1 + 1, ic0 : ic1 + 1] = True

    for room in rooms:
        _draw_room(grid, room)

    path_blocked = np.zeros((h, w), dtype=bool)
    path_blocked[:3, :3] = True
    path_blocked[:3, -3:] = True
    path_blocked[-3:, :3] = True
    path_blocked[-3:, -3:] = True
    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]
        path_blocked[r0 + 1 : r1, c0 + 1 : c1] = True
        gr, gc = room["gap"]
        path_blocked[gr, gc] = True

    if not _connect_rooms(grid, rooms, path_blocked, rng):
        return None

    expected = _expected_output(grid, rooms, patterns)
    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
