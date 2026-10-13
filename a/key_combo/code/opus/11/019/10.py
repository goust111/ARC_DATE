from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

GRID_WIDTH = 17
SEPARATOR_COL = 8
GEN_ATTEMPTS = 600
ROOM_SIZES = ((5, 5), (5, 6), (6, 5), (6, 6))
CARDINAL_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    for dr, dc in CARDINAL_DIRS:
        yield r + dr, c + dc


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_separator_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(cols) == 1, "expected one grey separator column"
    return int(cols[0])


def _unique_shortest_path(
    left: np.ndarray, start: tuple[int, int], goal: tuple[int, int]
) -> list[tuple[int, int]]:
    h, w = left.shape
    inf = 10**9
    dist = np.full((h, w), inf, dtype=int)
    ways = np.zeros((h, w), dtype=int)

    q: deque[tuple[int, int]] = deque([start])
    dist[start] = 0
    ways[start] = 1

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if not _in_bounds((h, w), nr, nc):
                continue
            if int(left[nr, nc]) == BLUE:
                continue
            nd = int(dist[r, c]) + 1
            if nd < dist[nr, nc]:
                dist[nr, nc] = nd
                ways[nr, nc] = ways[r, c]
                q.append((nr, nc))
            elif nd == dist[nr, nc]:
                ways[nr, nc] = min(2, ways[nr, nc] + ways[r, c])

    assert dist[goal] < inf, "red endpoints must be connected"
    assert ways[goal] == 1, "expected a unique shortest path"

    path = [goal]
    cur = goal
    while cur != start:
        r, c = cur
        prevs = [
            (nr, nc)
            for nr, nc in _neighbors4(r, c)
            if _in_bounds((h, w), nr, nc)
            and dist[nr, nc] == dist[r, c] - 1
            and ways[nr, nc] > 0
            and int(left[nr, nc]) != BLUE
        ]
        assert len(prevs) == 1, "shortest path backtracking must stay unique"
        cur = prevs[0]
        path.append(cur)
    path.reverse()
    return path


def _extract_path_sprite(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    sep = _find_separator_col(grid)
    left = np.asarray(grid[:, :sep], dtype=int)
    reds = [tuple(map(int, rc)) for rc in np.argwhere(left == RED)]
    assert len(reds) == 2, "expected exactly two red endpoints"
    reds.sort(key=lambda rc: (rc[0], rc[1]))
    path = _unique_shortest_path(left, reds[0], reds[1])
    interior = path[1:-1]
    assert interior, "path interior must be non-empty"
    r0, c0, r1, c1 = _bbox_from_coords(interior)
    sprite = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in interior:
        sprite[r - r0, c - c0] = True
    return sprite, path


def _stamp(out: np.ndarray, sprite: np.ndarray, top: int, left: int) -> None:
    h, w = sprite.shape
    for dr in range(h):
        for dc in range(w):
            if not sprite[dr, dc]:
                continue
            rr = top + dr
            cc = left + dc
            if not _in_bounds(out.shape, rr, cc):
                continue
            if int(out[rr, cc]) == BLACK:
                out[rr, cc] = GREEN


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the unique maze path, crop its shape, and stamp that cropped path onto each target with a half-dependent rotation.

    Input structure:
    - The grid has a black background, one full grey vertical separator column, and a left maze region containing exactly two red endpoints plus blue obstacle pixels.
    - The unique shortest 4-neighbor route between the red endpoints travels only through black cells in the left region.
    - The right region contains two or three yellow single-pixel target markers on black.

    Rule:
    - Keep the left maze, the red endpoints, the blue obstacles, the grey separator, and the yellow markers unchanged.
    - Compute the unique shortest path between the two red endpoints, drop the endpoints themselves, and crop the remaining path cells to their tight bounding box to obtain a boolean sprite.
    - For each yellow marker, use the cropped sprite as a green stamp aligned by its top-left corner at the marker: markers in the upper half keep the sprite as-is, while markers in the lower half use the 180-degree rotation; stamp only onto black cells.

    Key insight:
    - The repeated stamped object is not fixed in advance: the obstacle layout determines a unique path, and that path's cropped geometry becomes the reusable sprite.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep = _find_separator_col(g)
    sprite, _path = _extract_path_sprite(g)
    out = g.copy()
    mid_row = (g.shape[0] - 1) / 2.0

    for r, c in np.argwhere(g[:, sep + 1 :] == YELLOW):
        rr = int(r)
        cc = int(c) + sep + 1
        oriented = sprite if rr < mid_row else np.rot90(sprite, 2)
        _stamp(out, oriented, rr, cc)
    return out


def _room_inner_dims(room_h: int, room_w: int) -> tuple[int, int]:
    return room_h - 2, room_w - 2


def _random_walk(inner_h: int, inner_w: int) -> list[tuple[int, int]] | None:
    min_len = 5
    max_len = min(inner_h * inner_w, 8)
    if max_len < min_len:
        min_len = max_len

    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    for _ in range(120):
        target_len = random.randint(min_len, max_len)
        start = random.choice(cells)
        path = [start]
        seen = {start}

        def dfs() -> bool:
            if len(path) == target_len:
                return True
            moves = []
            r, c = path[-1]
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < inner_h and 0 <= nc < inner_w and (nr, nc) not in seen:
                    moves.append((nr, nc))
            random.shuffle(moves)
            for nxt in moves:
                seen.add(nxt)
                path.append(nxt)
                if dfs():
                    return True
                path.pop()
                seen.remove(nxt)
            return False

        if dfs() and path[0][0] < path[-1][0]:
            return path
    return None


def _turn_count(path: list[tuple[int, int]]) -> int:
    turns = 0
    for a, b, c in zip(path, path[1:], path[2:]):
        if (b[0] - a[0], b[1] - a[1]) != (c[0] - b[0], c[1] - b[1]):
            turns += 1
    return turns


def _grow_dead_end_branches(room: np.ndarray, max_additions: int) -> None:
    room_h, room_w = room.shape
    additions = random.randint(1, max_additions)
    for _ in range(additions):
        candidates = []
        for r in range(1, room_h - 1):
            for c in range(1, room_w - 1):
                if int(room[r, c]) != BLUE:
                    continue
                degree = sum(int(room[nr, nc]) != BLUE for nr, nc in _neighbors4(r, c))
                if degree == 1:
                    candidates.append((r, c))
        if not candidates:
            return
        room[random.choice(candidates)] = BLACK


def _build_left_room(height: int) -> np.ndarray | None:
    room_h, room_w = random.choice(ROOM_SIZES)
    inner_h, inner_w = _room_inner_dims(room_h, room_w)
    walk = _random_walk(inner_h, inner_w)
    if walk is None:
        return None

    room = np.full((room_h, room_w), BLUE, dtype=int)
    for r, c in walk:
        room[r + 1, c + 1] = BLACK
    room[walk[0][0] + 1, walk[0][1] + 1] = RED
    room[walk[-1][0] + 1, walk[-1][1] + 1] = RED
    _grow_dead_end_branches(room, max_additions=3)

    grid = np.zeros((height, SEPARATOR_COL), dtype=int)
    top = random.randint(0, height - room_h)
    left = random.randint(0, SEPARATOR_COL - room_w)
    grid[top : top + room_h, left : left + room_w] = room
    return grid


def _target_candidates(
    height: int, sprite_h: int, sprite_w: int, upper: bool
) -> list[tuple[int, int]]:
    mid_row = (height - 1) / 2.0
    rows = []
    for r in range(0, height - sprite_h + 1):
        if upper and r < mid_row:
            rows.append(r)
        if not upper and r > mid_row:
            rows.append(r)
    cols = list(range(SEPARATOR_COL + 1, GRID_WIDTH - sprite_w + 1))
    return [(r, c) for r in rows for c in cols]


def _boxes_separated(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0


def _place_targets(grid: np.ndarray, sprite: np.ndarray) -> bool:
    sprite_h, sprite_w = sprite.shape
    desired = 2 if random.random() < 0.55 else 3
    halves = [True, False]
    if desired == 3:
        halves.append(random.random() < 0.5)
    random.shuffle(halves)

    placed_boxes: list[tuple[int, int, int, int]] = []
    used_targets: set[tuple[int, int]] = set()
    for upper in halves:
        candidates = _target_candidates(grid.shape[0], sprite_h, sprite_w, upper)
        random.shuffle(candidates)
        chosen = None
        for r, c in candidates:
            box = (r, c, r + sprite_h - 1, c + sprite_w - 1)
            if any(not _boxes_separated(box, other, pad=1) for other in placed_boxes):
                continue
            chosen = (r, c)
            placed_boxes.append(box)
            break
        if chosen is None:
            return False
        used_targets.add(chosen)

    for r, c in used_targets:
        grid[r, c] = YELLOW
    return True


def _valid_generated(grid: np.ndarray) -> bool:
    try:
        sprite, path = _extract_path_sprite(grid)
        out = main(grid)
    except AssertionError:
        return False

    if not (1 <= sprite.shape[0] <= 5 and 1 <= sprite.shape[1] <= 5):
        return False
    if np.array_equal(sprite, np.rot90(sprite, 2)):
        return False
    if len(path) < 5:
        return False
    if _turn_count(path) < 1:
        return False
    if np.count_nonzero(grid == BLUE) < 10 or np.count_nonzero(grid == BLUE) > 20:
        return False

    sep = _find_separator_col(grid)
    yellow_rows = [int(r) for r, _ in np.argwhere(grid[:, sep + 1 :] == YELLOW)]
    if len(yellow_rows) not in (2, 3):
        return False
    mid_row = (grid.shape[0] - 1) / 2.0
    if not any(r < mid_row for r in yellow_rows):
        return False
    if not any(r > mid_row for r in yellow_rows):
        return False
    if np.count_nonzero(out == GREEN) < int(sprite.sum()):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        height = random.randint(15, 21)
        grid = np.zeros((height, GRID_WIDTH), dtype=int)
        grid[:, SEPARATOR_COL] = GREY

        left = _build_left_room(height)
        if left is None:
            continue
        grid[:, :SEPARATOR_COL] = left

        try:
            sprite, _ = _extract_path_sprite(grid)
        except AssertionError:
            continue
        if sprite.size == 0 or sprite.sum() < 2:
            continue

        if not _place_targets(grid, sprite):
            continue
        if _valid_generated(grid):
            return grid

    raise RuntimeError("failed to generate a valid input for puzzle 10")
