from __future__ import annotations

import random

import numpy as np


AZURE = 8
RED = 2
MARKER_COLORS = [0, 1, 3, 4, 5, 6, 7, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _ray_endpoint(
    grid: np.ndarray, start: tuple[int, int], step: tuple[int, int], wall: int = RED
) -> tuple[int, int]:
    r, c = start
    dr, dc = step
    while True:
        nr, nc = r + dr, c + dc
        if not _in_bounds(grid.shape, nr, nc):
            return r, c
        if int(grid[nr, nc]) == wall:
            return r, c
        r, c = nr, nc


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if (
                        _in_bounds(mask.shape, nr, nc)
                        and bool(mask[nr, nc])
                        and not bool(seen[nr, nc])
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each interior marker to the nearest wall-adjacent cell in the four
    cardinal directions, then erase the original marker.
    """
    g = np.asarray(grid, dtype=int)
    out = np.where(g == RED, RED, AZURE).astype(int)

    markers = [
        (int(r), int(c), int(g[r, c]))
        for r, c in np.argwhere((g != AZURE) & (g != RED))
    ]

    for r, c, color in markers:
        for step in DIRS4:
            rr, cc = _ray_endpoint(g, (r, c), step, wall=RED)
            out[rr, cc] = color

    return out


# --------------- generator ---------------

GRID_MIN, GRID_MAX = 19, 30
OUTER_MARGIN_MIN, OUTER_MARGIN_MAX = 1, 3
MIN_ROOM_SPAN = 5
MAX_SPLITS = 4
MAX_ATTEMPTS = 400

Room = tuple[int, int, int, int]


def _draw_outer_frame(grid: np.ndarray, room: Room) -> None:
    r0, r1, c0, c1 = room
    grid[r0 - 1, c0 - 1 : c1 + 2] = RED
    grid[r1 + 1, c0 - 1 : c1 + 2] = RED
    grid[r0 - 1 : r1 + 2, c0 - 1] = RED
    grid[r0 - 1 : r1 + 2, c1 + 1] = RED


def _split_actions(room: Room) -> list[tuple[str, int]]:
    r0, r1, c0, c1 = room
    actions: list[tuple[str, int]] = []
    for split_r in range(r0 + MIN_ROOM_SPAN, r1 - MIN_ROOM_SPAN + 1):
        actions.append(("h", split_r))
    for split_c in range(c0 + MIN_ROOM_SPAN, c1 - MIN_ROOM_SPAN + 1):
        actions.append(("v", split_c))
    return actions


def _apply_split(grid: np.ndarray, room: Room, action: tuple[str, int]) -> list[Room]:
    r0, r1, c0, c1 = room
    orient, split = action
    if orient == "h":
        grid[split, c0 - 1 : c1 + 2] = RED
        return [(r0, split - 1, c0, c1), (split + 1, r1, c0, c1)]
    grid[r0 - 1 : r1 + 2, split] = RED
    return [(r0, r1, c0, split - 1), (r0, r1, split + 1, c1)]


def _max_room_count(root: Room) -> int:
    r0, r1, c0, c1 = root
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    max_rows = max(1, (h + 1) // (MIN_ROOM_SPAN + 1))
    max_cols = max(1, (w + 1) // (MIN_ROOM_SPAN + 1))
    return max_rows * max_cols


def _partition_rooms(grid: np.ndarray, root: Room) -> list[Room] | None:
    rooms = [root]
    split_cap = min(MAX_SPLITS, _max_room_count(root) - 1)
    if split_cap < 1:
        return None
    n_splits = random.randint(1, split_cap)

    for _ in range(n_splits):
        candidates: list[tuple[int, tuple[str, int]]] = []
        for idx, room in enumerate(rooms):
            for action in _split_actions(room):
                candidates.append((idx, action))
        if not candidates:
            return None

        idx, action = random.choice(candidates)
        room = rooms.pop(idx)
        children = _apply_split(grid, room, action)
        random.shuffle(children)
        rooms.extend(children)

    return rooms


def _place_markers(grid: np.ndarray, room: Room) -> None:
    r0, r1, c0, c1 = room
    count = random.randint(1, 3)
    rows = random.sample(list(range(r0 + 1, r1)), count)
    cols = random.sample(list(range(c0 + 1, c1)), count)
    random.shuffle(cols)
    colors = random.sample(MARKER_COLORS, count)

    for color, r, c in zip(colors, rows, cols):
        grid[r, c] = color


def _closed_regions(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    regions = []
    for comp in _components4(grid != RED):
        if any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in comp):
            continue
        regions.append(comp)
    return regions


def _valid_generated_grid(grid: np.ndarray, expected_rooms: int) -> bool:
    regions = _closed_regions(grid)
    if len(regions) != expected_rooms:
        return False

    marker_total = int(np.count_nonzero((grid != AZURE) & (grid != RED)))
    seen_markers = 0
    for region in regions:
        markers = [(r, c) for r, c in region if int(grid[r, c]) not in (AZURE, RED)]
        if not 1 <= len(markers) <= 3:
            return False
        colors = [int(grid[r, c]) for r, c in markers]
        if len(set(colors)) != len(colors):
            return False
        rows = [r for r, _c in markers]
        cols = [c for _r, c in markers]
        if len(set(rows)) != len(rows) or len(set(cols)) != len(cols):
            return False
        seen_markers += len(markers)

    if seen_markers != marker_total:
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if not np.array_equal(out == RED, grid == RED):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    grid = np.full((h, w), AZURE, dtype=int)

    mt = random.randint(OUTER_MARGIN_MIN, OUTER_MARGIN_MAX)
    mb = random.randint(OUTER_MARGIN_MIN, OUTER_MARGIN_MAX)
    ml = random.randint(OUTER_MARGIN_MIN, OUTER_MARGIN_MAX)
    mr = random.randint(OUTER_MARGIN_MIN, OUTER_MARGIN_MAX)

    root = (mt + 1, h - mb - 2, ml + 1, w - mr - 2)
    if root[0] > root[1] or root[2] > root[3]:
        return None
    if root[1] - root[0] + 1 < 2 * MIN_ROOM_SPAN + 1 and root[3] - root[2] + 1 < 2 * MIN_ROOM_SPAN + 1:
        return None

    _draw_outer_frame(grid, root)
    rooms = _partition_rooms(grid, root)
    if rooms is None:
        return None

    random.shuffle(rooms)
    for room in rooms:
        _place_markers(grid, room)

    return grid if _valid_generated_grid(grid, len(rooms)) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        generated = _gen_attempt()
        if generated is not None:
            return generated
    raise RuntimeError("failed to generate a valid input")
