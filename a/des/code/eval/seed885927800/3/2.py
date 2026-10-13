from __future__ import annotations

from collections import deque
import heapq
import random

import numpy as np


BLACK = 0
WALL = 9
SEED_COLORS = (1, 2, 3, 4, 5, 6, 7, 8)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

MIN_ROOM_ROWS = 2
MAX_ROOM_ROWS = 4
MIN_ROOM_COLS = 2
MAX_ROOM_COLS = 5
MIN_ROOM_SIZE = 2
MAX_ROOM_SIZE = 5
MIN_GRID_SIZE = 10
MAX_GRID_SIZE = 30
GEN_RETRIES = 600


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _find_wall_lines(g: np.ndarray) -> tuple[list[int], list[int]]:
    h, w = g.shape
    wall_rows = [
        r
        for r in range(h)
        if bool(np.any((g[r, :-1] == WALL) & (g[r, 1:] == WALL)))
    ]
    wall_cols = [
        c
        for c in range(w)
        if bool(np.any((g[:-1, c] == WALL) & (g[1:, c] == WALL)))
    ]
    assert wall_rows and wall_cols, "expected visible wall rows and columns"
    assert wall_rows[0] == 0 and wall_rows[-1] == h - 1, "outer horizontal walls missing"
    assert wall_cols[0] == 0 and wall_cols[-1] == w - 1, "outer vertical walls missing"
    return wall_rows, wall_cols


def _parse_instance(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    wall_rows, wall_cols = _find_wall_lines(g)

    rooms: list[tuple[int, int, int, int]] = []
    room_id = -np.ones((h, w), dtype=int)
    for ri in range(len(wall_rows) - 1):
        r0 = wall_rows[ri] + 1
        r1 = wall_rows[ri + 1] - 1
        if r0 > r1:
            continue
        for ci in range(len(wall_cols) - 1):
            c0 = wall_cols[ci] + 1
            c1 = wall_cols[ci + 1] - 1
            if c0 > c1:
                continue
            rid = len(rooms)
            rooms.append((r0, c0, r1, c1))
            room_id[r0 : r1 + 1, c0 : c1 + 1] = rid

    seeds = []
    seen_rooms: set[int] = set()
    seen_colors: set[int] = set()
    for r, c in sorted(map(tuple, np.argwhere((g != BLACK) & (g != WALL)))):
        rr = int(r)
        cc = int(c)
        rid = int(room_id[rr, cc])
        assert rid >= 0, "seed must lie inside a room interior"
        color = int(g[rr, cc])
        assert rid not in seen_rooms, "each room may contain at most one seed"
        assert color not in seen_colors, "seed colors must be unique"
        seen_rooms.add(rid)
        seen_colors.add(color)
        r0, c0, r1, c1 = rooms[rid]
        budget = int(np.count_nonzero(g[r0 : r1 + 1, c0 : c1 + 1] == BLACK))
        seeds.append(
            {
                "r": rr,
                "c": cc,
                "color": color,
                "budget": budget,
                "room": rid,
            }
        )

    assert seeds, "expected at least one seed"
    return g, rooms, room_id, seeds


def _spread(g: np.ndarray, seeds: list[dict[str, int]]) -> np.ndarray:
    out = g.copy()
    remaining = [int(seed["budget"]) for seed in seeds]
    pq: list[tuple[int, int, int, int]] = []

    for order, seed in enumerate(seeds):
        heapq.heappush(pq, (0, order, int(seed["r"]), int(seed["c"])))

    h, w = g.shape
    while pq:
        dist, order, r, c = heapq.heappop(pq)
        if remaining[order] <= 0:
            continue
        color = int(seeds[order]["color"])
        for nr, nc in _neighbors4(r, c, h, w):
            if int(out[nr, nc]) != BLACK:
                continue
            out[nr, nc] = color
            remaining[order] -= 1
            heapq.heappush(pq, (dist + 1, order, nr, nc))
            if remaining[order] == 0:
                break

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Spread each seed through the wall maze with a per-seed fill budget equal to
    the black-cell area of the seed's original room.
    """
    g, _rooms, _room_id, seeds = _parse_instance(grid)
    return _spread(g, seeds)


def _wall_positions(sizes: list[int]) -> list[int]:
    pos = [0]
    cur = 0
    for size in sizes:
        cur += int(size) + 1
        pos.append(cur)
    return pos


def _room_boxes_from_sizes(
    heights: list[int], widths: list[int], wall_rows: list[int], wall_cols: list[int]
) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    boxes: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for i, _ in enumerate(heights):
        for j, _ in enumerate(widths):
            boxes[(i, j)] = (
                wall_rows[i] + 1,
                wall_cols[j] + 1,
                wall_rows[i + 1] - 1,
                wall_cols[j + 1] - 1,
            )
    return boxes


def _room_components(
    adjacency: dict[tuple[int, int], set[tuple[int, int]]]
) -> list[list[tuple[int, int]]]:
    comps: list[list[tuple[int, int]]] = []
    seen: set[tuple[int, int]] = set()
    for start in adjacency:
        if start in seen:
            continue
        comp = []
        stack = [start]
        seen.add(start)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in adjacency[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        comps.append(comp)
    return comps


def _distance_map(g: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    h, w = g.shape
    sr, sc = start
    dist = -np.ones((h, w), dtype=int)
    dist[sr, sc] = 0
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if dist[nr, nc] != -1:
                continue
            if int(g[nr, nc]) == WALL:
                continue
            if (nr, nc) != (sr, sc) and int(g[nr, nc]) != BLACK:
                continue
            dist[nr, nc] = int(dist[r, c] + 1)
            q.append((nr, nc))
    return dist


def _distance_maps(g: np.ndarray, seeds: list[dict[str, int]]) -> list[np.ndarray]:
    return [_distance_map(g, (int(seed["r"]), int(seed["c"]))) for seed in seeds]


def _unique_nearest_seed(g: np.ndarray, dmaps: list[np.ndarray]) -> bool:
    black_cells = np.argwhere(g == BLACK)
    for r, c in black_cells:
        best = None
        count = 0
        for dist in dmaps:
            d = int(dist[int(r), int(c)])
            if d < 0:
                continue
            if best is None or d < best:
                best = d
                count = 1
            elif d == best:
                count += 1
                if count > 1:
                    return False
    return True


def _reachable_black_counts(g: np.ndarray, dmaps: list[np.ndarray]) -> list[int]:
    counts = []
    for dist in dmaps:
        counts.append(int(np.count_nonzero((dist >= 1) & (g == BLACK))))
    return counts


def _has_overlap(g: np.ndarray, dmaps: list[np.ndarray]) -> bool:
    if len(dmaps) < 2:
        return False
    black = g == BLACK
    reach = np.zeros(g.shape, dtype=int)
    for dist in dmaps:
        reach += ((dist >= 1) & black).astype(int)
    return bool(np.any(reach >= 2))


def _sample_layout() -> tuple[list[int], list[int]] | None:
    room_rows = random.randint(MIN_ROOM_ROWS, MAX_ROOM_ROWS)
    room_cols = random.randint(MIN_ROOM_COLS, MAX_ROOM_COLS)
    heights = [random.randint(MIN_ROOM_SIZE, MAX_ROOM_SIZE) for _ in range(room_rows)]
    widths = [random.randint(MIN_ROOM_SIZE, MAX_ROOM_SIZE) for _ in range(room_cols)]
    h = sum(heights) + room_rows + 1
    w = sum(widths) + room_cols + 1
    if not (MIN_GRID_SIZE <= h <= MAX_GRID_SIZE and MIN_GRID_SIZE <= w <= MAX_GRID_SIZE):
        return None
    return heights, widths


def _open_doors(
    grid: np.ndarray,
    boxes: dict[tuple[int, int], tuple[int, int, int, int]],
    wall_rows: list[int],
    wall_cols: list[int],
    open_prob: float,
) -> dict[tuple[int, int], set[tuple[int, int]]]:
    heights = len(wall_rows) - 1
    widths = len(wall_cols) - 1
    adjacency = {room: set() for room in boxes}
    candidates: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []

    for i in range(heights - 1):
        wr = wall_rows[i + 1]
        for j in range(widths):
            left = wall_cols[j] + 1
            right = wall_cols[j + 1] - 1
            candidates.append(((i, j), (i + 1, j), (wr, random.randint(left, right))))

    for i in range(heights):
        for j in range(widths - 1):
            wc = wall_cols[j + 1]
            top = wall_rows[i] + 1
            bottom = wall_rows[i + 1] - 1
            candidates.append(((i, j), (i, j + 1), (random.randint(top, bottom), wc)))

    random.shuffle(candidates)
    opened = 0
    for a, b, (r, c) in candidates:
        if random.random() >= open_prob:
            continue
        grid[r, c] = BLACK
        adjacency[a].add(b)
        adjacency[b].add(a)
        opened += 1

    if opened == 0 and candidates:
        a, b, (r, c) = random.choice(candidates)
        grid[r, c] = BLACK
        adjacency[a].add(b)
        adjacency[b].add(a)

    return adjacency


def _seed_count_for_mode(mode: str, room_count: int) -> int:
    if room_count == 1:
        return 1
    if mode == "sealed":
        return 1 if random.random() < 0.75 else 2
    if mode == "race":
        return min(room_count, random.randint(2, min(3, room_count)))
    return min(room_count, random.choice([2, 2, 3, min(4, room_count)]))


def _place_seeds(
    grid: np.ndarray,
    boxes: dict[tuple[int, int], tuple[int, int, int, int]],
    adjacency: dict[tuple[int, int], set[tuple[int, int]]],
    mode: str,
) -> bool:
    all_rooms = list(boxes)
    seed_count = _seed_count_for_mode(mode, len(all_rooms))

    if mode == "race":
        comps = [comp for comp in _room_components(adjacency) if len(comp) >= seed_count]
        if not comps:
            return False
        chosen_rooms = random.sample(random.choice(comps), seed_count)
    else:
        chosen_rooms = random.sample(all_rooms, seed_count)

    colors = random.sample(SEED_COLORS, seed_count)
    for room, color in zip(chosen_rooms, colors, strict=False):
        r0, c0, r1, c1 = boxes[room]
        r = random.randint(r0, r1)
        c = random.randint(c0, c1)
        grid[r, c] = color
    return True


def _generate_attempt() -> np.ndarray | None:
    layout = _sample_layout()
    if layout is None:
        return None
    heights, widths = layout
    wall_rows = _wall_positions(heights)
    wall_cols = _wall_positions(widths)
    h = wall_rows[-1] + 1
    w = wall_cols[-1] + 1

    grid = np.full((h, w), BLACK, dtype=int)
    grid[wall_rows, :] = WALL
    grid[:, wall_cols] = WALL

    boxes = _room_boxes_from_sizes(heights, widths, wall_rows, wall_cols)
    mode = random.choice(("sealed", "race", "mixed"))
    if mode == "sealed":
        open_prob = random.choice((0.15, 0.25, 0.35))
    elif mode == "race":
        open_prob = random.choice((0.55, 0.7, 0.85))
    else:
        open_prob = random.choice((0.3, 0.45, 0.6))

    adjacency = _open_doors(grid, boxes, wall_rows, wall_cols, open_prob)
    if not _place_seeds(grid, boxes, adjacency, mode):
        return None

    try:
        parsed_grid, _rooms, _room_id, seeds = _parse_instance(grid)
    except AssertionError:
        return None

    dmaps = _distance_maps(parsed_grid, seeds)
    if not _unique_nearest_seed(parsed_grid, dmaps):
        return None

    reachable = _reachable_black_counts(parsed_grid, dmaps)
    budgets = [int(seed["budget"]) for seed in seeds]

    if mode == "sealed" and not any(r == b for r, b in zip(reachable, budgets, strict=False)):
        return None
    if mode == "race":
        if len(seeds) < 2 or not _has_overlap(parsed_grid, dmaps):
            return None
        if not any(r > b for r, b in zip(reachable, budgets, strict=False)):
            return None

    out = _spread(parsed_grid, seeds)
    if np.array_equal(out, parsed_grid):
        return None
    return parsed_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid maze instance")
