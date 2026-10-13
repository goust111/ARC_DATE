from __future__ import annotations

import heapq
import random
from collections import Counter, deque

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400
ROOM_COUNT_RANGE = (2, 4)
ROOM_SIZE_RANGE = (4, 6)
DOOR_COLOR_COUNT_RANGE = (2, 4)


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[counts.argmax()])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)

    return comps


def _wall_color(grid: np.ndarray, bg: int) -> int:
    values, counts = np.unique(grid[grid != bg], return_counts=True)
    assert values.size > 0, "expected at least one non-background color"
    return int(values[counts.argmax()])


def _door_kind(grid: np.ndarray, r: int, c: int, bg: int, wall: int) -> str | None:
    h, w = grid.shape
    if r <= 0 or r >= h - 1 or c <= 0 or c >= w - 1:
        return None

    left_bg = int(grid[r, c - 1]) == bg
    right_bg = int(grid[r, c + 1]) == bg
    up_bg = int(grid[r - 1, c]) == bg
    down_bg = int(grid[r + 1, c]) == bg

    left_wall = int(grid[r, c - 1]) == wall
    right_wall = int(grid[r, c + 1]) == wall
    up_wall = int(grid[r - 1, c]) == wall
    down_wall = int(grid[r + 1, c]) == wall

    vertical_wall_door = left_bg and right_bg and up_wall and down_wall
    horizontal_wall_door = up_bg and down_bg and left_wall and right_wall
    if vertical_wall_door == horizontal_wall_door:
        return None
    return "vertical" if vertical_wall_door else "horizontal"


def _room_graph(
    grid: np.ndarray,
) -> tuple[
    int,
    int,
    dict[tuple[int, int], tuple[str, int]],
    tuple[int, int],
    list[list[tuple[int, int]]],
    np.ndarray,
    dict[int, list[tuple[int, int, int, int]]],
]:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    wall = _wall_color(g, bg)

    doors: dict[tuple[int, int], tuple[str, int]] = {}
    seed_cells: list[tuple[int, int]] = []
    special = np.argwhere((g != bg) & (g != wall))
    for r, c in special:
        rr, cc = int(r), int(c)
        kind = _door_kind(g, rr, cc, bg, wall)
        if kind is None:
            seed_cells.append((rr, cc))
        else:
            doors[(rr, cc)] = (kind, int(g[rr, cc]))

    assert len(seed_cells) == 1, "expected exactly one non-door seed pixel"
    seed = seed_cells[0]

    rooms = _components(g == bg)
    assert rooms, "expected at least one background room"
    room_ids = np.full(g.shape, -1, dtype=int)
    for idx, comp in enumerate(rooms):
        for r, c in comp:
            room_ids[r, c] = idx

    seed_neighbors = {
        int(room_ids[nr, nc])
        for dr, dc in DIRS4
        if 0 <= (nr := seed[0] + dr) < g.shape[0]
        and 0 <= (nc := seed[1] + dc) < g.shape[1]
        and int(g[nr, nc]) == bg
    }
    assert len(seed_neighbors) == 1, "seed must lie inside exactly one room"
    seed_room = next(iter(seed_neighbors))

    adjacency: dict[int, list[tuple[int, int, int, int]]] = {idx: [] for idx in range(len(rooms))}
    for (r, c), (kind, color) in doors.items():
        if kind == "vertical":
            a = int(room_ids[r, c - 1])
            b = int(room_ids[r, c + 1])
        else:
            a = int(room_ids[r - 1, c])
            b = int(room_ids[r + 1, c])
        assert a >= 0 and b >= 0 and a != b, "door must connect two distinct rooms"
        adjacency[a].append((b, r, c, color))
        adjacency[b].append((a, r, c, color))

    return bg, wall, doors, seed, rooms, room_ids, adjacency


def _propagate_room_colors(
    room_count: int,
    adjacency: dict[int, list[tuple[int, int, int, int]]],
    seed_room: int,
    seed_color: int,
) -> dict[int, int]:
    best_sig: dict[int, tuple[int, int, int]] = {seed_room: (0, -1, -1)}
    fill_color: dict[int, int] = {seed_room: seed_color}
    heap: list[tuple[int, int, int, int, int]] = [(0, -1, -1, seed_room, seed_color)]

    while heap:
        dist, door_r, door_c, room, color = heapq.heappop(heap)
        if room >= room_count:
            continue
        if best_sig.get(room) != (dist, door_r, door_c) or fill_color.get(room) != color:
            continue

        for nxt, nr, nc, door_color in adjacency[room]:
            cand_sig = (dist + 1, nr, nc)
            prev_sig = best_sig.get(nxt)
            if prev_sig is None or cand_sig < prev_sig:
                best_sig[nxt] = cand_sig
                fill_color[nxt] = door_color
                heapq.heappush(heap, (cand_sig[0], cand_sig[1], cand_sig[2], nxt, door_color))

    return fill_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flood-fill the seed room and propagate through colored doors, changing fill color at each door crossing.

    Input structure:
    - The grid has a dominant background color and one orthogonal wall network in a single wall color.
    - The wall network partitions the background into rectangular rooms, and some wall cells are replaced by single-pixel colored doors.
    - Exactly one room contains one extra non-background, non-wall seed pixel inside the room interior.

    Rule:
    - Fill only background cells, never overwrite walls, doors, or the seed pixel itself.
    - The seed room's background cells become the seed color.
    - Crossing a door enters the adjacent room and changes the fill color to that door's color.
    - If a room can first be reached through multiple doors at the same topological distance from the seed room, use the reaching door with smaller row, then smaller column.

    Key insight:
    - Rooms are background components separated by treating the colored door pixels as wall barriers, and the final color of each reachable room is determined by the last door on the lexicographically earliest shortest path from the seed room.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, _wall, _doors, seed, rooms, room_ids, adjacency = _room_graph(g)
    seed_room = int(
        next(
            int(room_ids[nr, nc])
            for dr, dc in DIRS4
            if 0 <= (nr := seed[0] + dr) < g.shape[0]
            and 0 <= (nc := seed[1] + dc) < g.shape[1]
            and int(g[nr, nc]) == bg
        )
    )
    fill_color = _propagate_room_colors(len(rooms), adjacency, seed_room, int(g[seed]))

    out = g.copy()
    for room, color in fill_color.items():
        for r, c in rooms[room]:
            out[r, c] = color
    return out


def _room_id(i: int, j: int, cols: int) -> int:
    return i * cols + j


def _room_boxes(
    row_heights: list[int], col_widths: list[int]
) -> tuple[list[int], list[int], dict[int, tuple[int, int, int, int]]]:
    wall_rows = [0]
    for size in row_heights:
        wall_rows.append(wall_rows[-1] + size + 1)
    wall_cols = [0]
    for size in col_widths:
        wall_cols.append(wall_cols[-1] + size + 1)

    boxes: dict[int, tuple[int, int, int, int]] = {}
    cols = len(col_widths)
    for i, _ in enumerate(row_heights):
        for j, _ in enumerate(col_widths):
            boxes[_room_id(i, j, cols)] = (
                wall_rows[i] + 1,
                wall_rows[i + 1] - 1,
                wall_cols[j] + 1,
                wall_cols[j + 1] - 1,
            )
    return wall_rows, wall_cols, boxes


def _pair_cells(
    room_a: int,
    room_b: int,
    boxes: dict[int, tuple[int, int, int, int]],
) -> list[tuple[int, int]]:
    a_r0, a_r1, a_c0, a_c1 = boxes[room_a]
    b_r0, b_r1, b_c0, b_c1 = boxes[room_b]

    if a_r0 == b_r0 and a_r1 == b_r1:
        wall_c = min(a_c1, b_c1) + 1
        return [(r, wall_c) for r in range(a_r0, a_r1 + 1)]

    wall_r = min(a_r1, b_r1) + 1
    return [(wall_r, c) for c in range(a_c0, a_c1 + 1)]


def _non_adjacent_choices(cells: list[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    indexes = list(range(len(cells)))
    for _ in range(60):
        chosen = sorted(random.sample(indexes, count))
        if all(b - a >= 2 for a, b in zip(chosen, chosen[1:], strict=False)):
            return [cells[idx] for idx in chosen]
    return None


def _expected_fill_colors(
    room_count: int,
    doors: list[tuple[int, int, int, int, int]],
    seed_room: int,
    seed_color: int,
) -> dict[int, int]:
    adjacency: dict[int, list[tuple[int, int, int, int]]] = {idx: [] for idx in range(room_count)}
    for room_a, room_b, r, c, color in doors:
        adjacency[room_a].append((room_b, r, c, color))
        adjacency[room_b].append((room_a, r, c, color))
    return _propagate_room_colors(room_count, adjacency, seed_room, seed_color)


def _all_neighbor_pairs(rows: int, cols: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for i in range(rows):
        for j in range(cols):
            here = _room_id(i, j, cols)
            if j + 1 < cols:
                pairs.append((here, _room_id(i, j + 1, cols)))
            if i + 1 < rows:
                pairs.append((here, _room_id(i + 1, j, cols)))
    return pairs


def _grow_reachable_pairs(
    rows: int,
    cols: int,
    seed_room: int,
    target_size: int,
    pair_set: set[tuple[int, int]],
) -> set[int]:
    reachable = {seed_room}
    all_pairs = _all_neighbor_pairs(rows, cols)

    while len(reachable) < target_size:
        frontier = [
            pair
            for pair in all_pairs
            if (pair[0] in reachable) ^ (pair[1] in reachable)
        ]
        assert frontier, "reachable set should be expandable in a room lattice"
        pair = random.choice(frontier)
        pair_set.add(tuple(sorted(pair)))
        reachable.update(pair)

    return reachable


def _decorate_with_extra_pairs(
    selected_pairs: set[tuple[int, int]],
    reachable: set[int],
    all_pairs: list[tuple[int, int]],
) -> None:
    for pair in all_pairs:
        ordered = tuple(sorted(pair))
        if ordered in selected_pairs:
            continue
        if pair[0] in reachable and pair[1] in reachable and random.random() < 0.28:
            selected_pairs.add(ordered)
        elif pair[0] not in reachable and pair[1] not in reachable and random.random() < 0.20:
            selected_pairs.add(ordered)


def _generate_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    room_rows = random.randint(*ROOM_COUNT_RANGE)
    room_cols = random.randint(*ROOM_COUNT_RANGE)
    row_heights = [random.randint(*ROOM_SIZE_RANGE) for _ in range(room_rows)]
    col_widths = [random.randint(*ROOM_SIZE_RANGE) for _ in range(room_cols)]

    wall_rows, wall_cols, boxes = _room_boxes(row_heights, col_widths)
    height = wall_rows[-1] + 1
    width = wall_cols[-1] + 1
    if height > 30 or width > 30:
        return None

    colors = list(range(10))
    random.shuffle(colors)
    door_color_count = random.randint(*DOOR_COLOR_COUNT_RANGE)
    if 3 + door_color_count > len(colors):
        return None
    bg, wall, seed_color = colors[:3]
    door_palette = colors[3 : 3 + door_color_count]

    all_pairs = _all_neighbor_pairs(room_rows, room_cols)
    room_count = room_rows * room_cols

    force_tie = room_rows >= 2 and room_cols >= 2 and random.random() < 0.45
    force_double = random.random() < 0.50

    selected_pairs: set[tuple[int, int]] = set()
    if force_tie:
        block_r = random.randint(0, room_rows - 2)
        block_c = random.randint(0, room_cols - 2)
        seed_room = _room_id(block_r, block_c, room_cols)
        right = _room_id(block_r, block_c + 1, room_cols)
        down = _room_id(block_r + 1, block_c, room_cols)
        diag = _room_id(block_r + 1, block_c + 1, room_cols)
        selected_pairs.update(
            {
                tuple(sorted((seed_room, right))),
                tuple(sorted((seed_room, down))),
                tuple(sorted((right, diag))),
                tuple(sorted((down, diag))),
            }
        )
        min_reachable = 4
    else:
        seed_room = random.randrange(room_count)
        min_reachable = 1

    target_reachable = random.randint(min_reachable, room_count)
    reachable = _grow_reachable_pairs(room_rows, room_cols, seed_room, target_reachable, selected_pairs)
    _decorate_with_extra_pairs(selected_pairs, reachable, all_pairs)

    if not selected_pairs:
        selected_pairs.add(tuple(sorted(random.choice(all_pairs))))

    eligible_double = [pair for pair in selected_pairs if len(_pair_cells(pair[0], pair[1], boxes)) >= 3]
    forced_double_pair = random.choice(eligible_double) if force_double and eligible_double else None

    doors: list[tuple[int, int, int, int, int]] = []
    for pair in sorted(selected_pairs):
        cells = _pair_cells(pair[0], pair[1], boxes)
        door_count = 1
        if pair == forced_double_pair:
            door_count = 2
        elif len(cells) >= 3 and random.random() < 0.22:
            door_count = 2

        chosen = _non_adjacent_choices(cells, door_count)
        if chosen is None:
            return None

        if door_count == 2:
            color_choices = random.sample(door_palette, 2) if len(door_palette) >= 2 else [door_palette[0], door_palette[0]]
        else:
            color_choices = [random.choice(door_palette)]

        for (r, c), color in zip(chosen, color_choices, strict=False):
            doors.append((pair[0], pair[1], r, c, color))

    grid = np.full((height, width), bg, dtype=int)
    for r in wall_rows:
        grid[r, :] = wall
    for c in wall_cols:
        grid[:, c] = wall
    for room_a, room_b, r, c, color in doors:
        _ = room_a, room_b
        grid[r, c] = color

    r0, r1, c0, c1 = boxes[seed_room]
    seed_r = random.randint(r0 + 1, r1 - 1)
    seed_c = random.randint(c0 + 1, c1 - 1)
    grid[seed_r, seed_c] = seed_color

    fill_color = _expected_fill_colors(room_count, doors, seed_room, seed_color)
    expected = grid.copy()
    for room, color in fill_color.items():
        rr0, rr1, cc0, cc1 = boxes[room]
        region = expected[rr0 : rr1 + 1, cc0 : cc1 + 1]
        region[region == bg] = color

    counts = Counter(map(int, grid.ravel()))
    other_counts = [count for color, count in counts.items() if color not in {bg, wall}]
    if not other_counts:
        return None
    if counts[bg] <= counts[wall]:
        return None
    if counts[wall] <= max(other_counts):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        sample = _generate_attempt()
        if sample is not None:
            return sample[0]
    raise RuntimeError("failed to generate a valid room-door flood-fill input")
