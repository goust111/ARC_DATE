from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
WALL = 5
SEED_PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

ROOM_SIZE_MIN = 5
ROOM_SIZE_MAX = 6
CORE_GAP_MIN = 2
CORE_GAP_MAX = 4
BRANCH_GAP_MIN = 2
BRANCH_GAP_MAX = 3
GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_union(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _bfs_dist(passable: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    dist = np.full(passable.shape, -1, dtype=int)
    sr, sc = start
    dist[sr, sc] = 0
    queue: deque[tuple[int, int]] = deque([(sr, sc)])
    while queue:
        r, c = queue.popleft()
        nd = dist[r, c] + 1
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < passable.shape[0] and 0 <= nc < passable.shape[1]:
                if passable[nr, nc] and dist[nr, nc] < 0:
                    dist[nr, nc] = nd
                    queue.append((nr, nc))
    return dist


def _seed_data(grid: np.ndarray) -> list[tuple[int, int, int]]:
    coords = np.argwhere((grid != BG) & (grid != WALL))
    assert coords.shape[0] == 2, "expected exactly two seed pixels"
    seeds = [(int(r), int(c), int(grid[r, c])) for r, c in coords]
    assert seeds[0][2] != seeds[1][2], "seed colors must be distinct"
    return seeds


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color each reachable room-and-corridor cell by the nearer seed along walkable paths, turning equal-distance cells into grey separators.

    Input structure:
    - Black cells are walkable background; grey cells are walls.
    - Several hollow rectangular rooms are linked by one-cell-wide corridors through openings in their walls.
    - Exactly two room-interior cells are non-grey, non-black seed pixels with different colors.

    Rule:
    - Treat every non-grey cell as passable and expand the two seed colors through passable cells with 4-connectivity.
    - For each reachable black cell, paint it with the color of the seed whose path distance is smaller.
    - If both seeds reach a cell in the same number of steps, paint that cell grey; keep walls, seeds, and unreachable black cells unchanged.

    Key insight:
    - Distances are geodesic distances constrained by room walls and corridor topology, not straight-line or plain Manhattan distance on the canvas.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    seeds = _seed_data(g)
    passable = g != WALL
    dist0 = _bfs_dist(passable, (seeds[0][0], seeds[0][1]))
    dist1 = _bfs_dist(passable, (seeds[1][0], seeds[1][1]))

    out = g.copy()
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if g[r, c] == WALL:
                continue
            if (r, c) == (seeds[0][0], seeds[0][1]) or (r, c) == (seeds[1][0], seeds[1][1]):
                continue
            d0 = int(dist0[r, c])
            d1 = int(dist1[r, c])
            if d0 < 0 and d1 < 0:
                continue
            if d1 < 0 or (d0 >= 0 and d0 < d1):
                out[r, c] = seeds[0][2]
            elif d0 < 0 or d1 < d0:
                out[r, c] = seeds[1][2]
            else:
                out[r, c] = WALL
    return out


def _room_interior(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return r0 + 1, c0 + 1, r1 - 1, c1 - 1


def _carve_room(grid: np.ndarray, box: tuple[int, int, int, int]) -> None:
    ir0, ic0, ir1, ic1 = _room_interior(box)
    grid[ir0 : ir1 + 1, ic0 : ic1 + 1] = BG


def _carve_h(grid: np.ndarray, row: int, c0: int, c1: int) -> None:
    left, right = sorted((c0, c1))
    grid[row, left : right + 1] = BG


def _carve_v(grid: np.ndarray, col: int, r0: int, r1: int) -> None:
    top, bottom = sorted((r0, r1))
    grid[top : bottom + 1, col] = BG


def _random_room_cell(box: tuple[int, int, int, int]) -> tuple[int, int]:
    ir0, ic0, ir1, ic1 = _room_interior(box)
    return random.randint(ir0, ir1), random.randint(ic0, ic1)


def _choice_biased(options: list[tuple[str, str]]) -> tuple[str, str]:
    diagonal = [pair for pair in options if pair in [("tl", "br"), ("tr", "bl")]]
    if diagonal and random.random() < 0.65:
        return random.choice(diagonal)
    return random.choice(options)


def _core_layout():
    h_top = random.randint(ROOM_SIZE_MIN, ROOM_SIZE_MAX)
    h_bottom = random.randint(ROOM_SIZE_MIN, ROOM_SIZE_MAX)
    w_left = random.randint(ROOM_SIZE_MIN, ROOM_SIZE_MAX)
    w_right = random.randint(ROOM_SIZE_MIN, ROOM_SIZE_MAX)
    gap_x = random.randint(CORE_GAP_MIN, CORE_GAP_MAX)
    gap_y = random.randint(CORE_GAP_MIN, CORE_GAP_MAX)

    rooms = {
        "tl": (0, 0, h_top - 1, w_left - 1),
        "tr": (0, w_left + gap_x, h_top - 1, w_left + gap_x + w_right - 1),
        "bl": (h_top + gap_y, 0, h_top + gap_y + h_bottom - 1, w_left - 1),
        "br": (
            h_top + gap_y,
            w_left + gap_x,
            h_top + gap_y + h_bottom - 1,
            w_left + gap_x + w_right - 1,
        ),
    }

    row_top = random.randint(1, h_top - 2)
    row_bottom = h_top + gap_y + random.randint(1, h_bottom - 2)
    col_left = random.randint(1, w_left - 2)
    col_right = w_left + gap_x + random.randint(1, w_right - 2)

    corridors = [
        ("h", row_top, rooms["tl"][3], rooms["tr"][1]),
        ("h", row_bottom, rooms["bl"][3], rooms["br"][1]),
        ("v", col_left, rooms["tl"][2], rooms["bl"][0]),
        ("v", col_right, rooms["tr"][2], rooms["br"][0]),
    ]
    return rooms, corridors


def _branch_layout(
    rooms: dict[str, tuple[int, int, int, int]],
    corridors: list[tuple[str, int, int, int]],
) -> tuple[dict[str, tuple[int, int, int, int]], list[tuple[str, int, int, int]]]:
    if random.random() >= 0.55:
        return rooms, corridors

    side = random.choice(["left", "right", "top", "bottom"])
    branch_h = random.randint(ROOM_SIZE_MIN, ROOM_SIZE_MAX)
    branch_w = random.randint(ROOM_SIZE_MIN, ROOM_SIZE_MAX)
    gap = random.randint(BRANCH_GAP_MIN, BRANCH_GAP_MAX)

    if side == "left":
        host = random.choice(["tl", "bl"])
        hr0, hc0, hr1, _hc1 = rooms[host]
        room = (hr0, hc0 - gap - branch_w, hr0 + branch_h - 1, hc0 - gap - 1)
        overlap_top = max(room[0] + 1, hr0 + 1)
        overlap_bottom = min(room[2] - 1, hr1 - 1)
        row = random.randint(overlap_top, overlap_bottom)
        corridors = corridors + [("h", row, room[3], rooms[host][1])]
    elif side == "right":
        host = random.choice(["tr", "br"])
        hr0, _hc0, hr1, hc1 = rooms[host]
        room = (hr0, hc1 + gap + 1, hr0 + branch_h - 1, hc1 + gap + branch_w)
        overlap_top = max(room[0] + 1, hr0 + 1)
        overlap_bottom = min(room[2] - 1, hr1 - 1)
        row = random.randint(overlap_top, overlap_bottom)
        corridors = corridors + [("h", row, rooms[host][3], room[1])]
    elif side == "top":
        host = random.choice(["tl", "tr"])
        hr0, hc0, _hr1, hc1 = rooms[host]
        room = (hr0 - gap - branch_h, hc0, hr0 - gap - 1, hc0 + branch_w - 1)
        overlap_left = max(room[1] + 1, hc0 + 1)
        overlap_right = min(room[3] - 1, hc1 - 1)
        col = random.randint(overlap_left, overlap_right)
        corridors = corridors + [("v", col, room[2], rooms[host][0])]
    else:
        host = random.choice(["bl", "br"])
        _hr0, hc0, hr1, hc1 = rooms[host]
        room = (hr1 + gap + 1, hc0, hr1 + gap + branch_h, hc0 + branch_w - 1)
        overlap_left = max(room[1] + 1, hc0 + 1)
        overlap_right = min(room[3] - 1, hc1 - 1)
        col = random.randint(overlap_left, overlap_right)
        corridors = corridors + [("v", col, rooms[host][2], room[0])]

    rooms = dict(rooms)
    rooms["branch"] = room
    return rooms, corridors


def _shift_boxes(
    rooms: dict[str, tuple[int, int, int, int]],
    corridors: list[tuple[str, int, int, int]],
) -> tuple[dict[str, tuple[int, int, int, int]], list[tuple[str, int, int, int]], tuple[int, int, int, int], tuple[int, int]]:
    room_boxes = list(rooms.values())
    min_r, min_c, max_r, max_c = _bbox_union(room_boxes)
    outer = (min_r - 1, min_c - 1, max_r + 1, max_c + 1)
    block_h = outer[2] - outer[0] + 1
    block_w = outer[3] - outer[1] + 1

    min_h = block_h + 2
    min_w = block_w + 2
    max_h = min(30, min_h + 3)
    max_w = min(30, min_w + 3)
    if min_h > 30 or min_w > 30:
        raise ValueError("layout too large")

    rows = random.randint(min_h, max_h)
    cols = random.randint(min_w, max_w)
    dr = random.randint(1 - outer[0], rows - 2 - outer[2])
    dc = random.randint(1 - outer[1], cols - 2 - outer[3])

    shifted_rooms = {}
    for key, (r0, c0, r1, c1) in rooms.items():
        shifted_rooms[key] = (r0 + dr, c0 + dc, r1 + dr, c1 + dc)

    shifted_corridors = []
    for kind, fixed, start, end in corridors:
        shifted_corridors.append((kind, fixed + (dr if kind == "h" else dc), start + (dc if kind == "h" else dr), end + (dc if kind == "h" else dr)))

    shifted_outer = (outer[0] + dr, outer[1] + dc, outer[2] + dr, outer[3] + dc)
    return shifted_rooms, shifted_corridors, shifted_outer, (rows, cols)


def _render_input() -> np.ndarray:
    rooms, corridors = _core_layout()
    rooms, corridors = _branch_layout(rooms, corridors)
    rooms, corridors, outer, shape = _shift_boxes(rooms, corridors)

    grid = np.full(shape, BG, dtype=int)
    r0, c0, r1, c1 = outer
    grid[r0 : r1 + 1, c0 : c1 + 1] = WALL

    for box in rooms.values():
        _carve_room(grid, box)

    for kind, fixed, start, end in corridors:
        if kind == "h":
            _carve_h(grid, fixed, start, end)
        else:
            _carve_v(grid, fixed, start, end)

    seed_color_a, seed_color_b = random.sample(SEED_PALETTE, 2)
    core_pairs = [("tl", "tr"), ("tl", "bl"), ("tl", "br"), ("tr", "bl"), ("tr", "br"), ("bl", "br")]
    room_a, room_b = _choice_biased(core_pairs)
    ra, ca = _random_room_cell(rooms[room_a])
    rb, cb = _random_room_cell(rooms[room_b])
    grid[ra, ca] = seed_color_a
    grid[rb, cb] = seed_color_b
    return grid


def _distance_signature(grid: np.ndarray):
    seeds = _seed_data(grid)
    passable = grid != WALL
    dist0 = _bfs_dist(passable, (seeds[0][0], seeds[0][1]))
    dist1 = _bfs_dist(passable, (seeds[1][0], seeds[1][1]))
    return seeds, dist0, dist1


def _manhattan_relation(r: int, c: int, seeds: list[tuple[int, int, int]]) -> int:
    d0 = abs(r - seeds[0][0]) + abs(c - seeds[0][1])
    d1 = abs(r - seeds[1][0]) + abs(c - seeds[1][1])
    if d0 < d1:
        return 0
    if d1 < d0:
        return 1
    return 2


def _geodesic_relation(r: int, c: int, dist0: np.ndarray, dist1: np.ndarray) -> int:
    d0 = int(dist0[r, c])
    d1 = int(dist1[r, c])
    if d0 < 0 and d1 < 0:
        return -1
    if d1 < 0 or (d0 >= 0 and d0 < d1):
        return 0
    if d0 < 0 or d1 < d0:
        return 1
    return 2


def _valid_input(grid: np.ndarray, require_tie: bool) -> bool:
    try:
        seeds, dist0, dist1 = _distance_signature(grid)
        out = main(grid)
    except Exception:
        return False

    reachable = (dist0 >= 0) | (dist1 >= 0)
    carved_black = (grid == BG) & reachable
    if int(np.count_nonzero(carved_black)) < 22:
        return False

    unreachable_black = (grid == BG) & ~reachable
    if int(np.count_nonzero(unreachable_black)) == 0:
        return False

    tie_cells = (grid == BG) & (dist0 == dist1) & (dist0 >= 0)
    if require_tie and int(np.count_nonzero(tie_cells)) == 0:
        return False

    mismatch = False
    for r, c in np.argwhere(carved_black):
        if _geodesic_relation(int(r), int(c), dist0, dist1) != _manhattan_relation(int(r), int(c), seeds):
            mismatch = True
            break
    if not mismatch:
        return False

    seed_colors = [seed[2] for seed in seeds]
    filled_a = int(np.count_nonzero(out == seed_colors[0]))
    filled_b = int(np.count_nonzero(out == seed_colors[1]))
    if filled_a < 8 or filled_b < 8:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        require_tie = random.random() < 0.7
        try:
            grid = _render_input()
        except ValueError:
            continue
        if _valid_input(grid, require_tie):
            return grid
    raise RuntimeError("failed to generate a valid corridor-distance example")
