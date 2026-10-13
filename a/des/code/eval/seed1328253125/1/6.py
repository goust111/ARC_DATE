from __future__ import annotations

from collections import defaultdict, deque
import random
import numpy as np


BG = 0
WALL = 5
MAROON = 9
DOOR_COLORS = (1, 2, 3, 4, 6, 7, 8)


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    labels = np.zeros(mask.shape, dtype=int)
    h, w = mask.shape
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            count += 1
            labels[r, c] = count
            q = deque([(r, c)])
            while q:
                rr, cc = q.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = count
                        q.append((nr, nc))
    return labels, count


def _component_bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    floor_mask = (g == BG) | (g == MAROON)
    room_labels, room_count = _label_components(floor_mask)
    assert room_count >= 2

    maroon_labels, maroon_count = _label_components(g == MAROON)
    object_bbox = None
    target_pos = None
    for label in range(1, maroon_count + 1):
        coords = np.argwhere(maroon_labels == label)
        area = int(coords.shape[0])
        r0, c0, r1, c1 = _component_bbox(coords)
        if area == 4 and r1 - r0 == 1 and c1 - c0 == 1:
            assert object_bbox is None
            object_bbox = (r0, c0, r1, c1)
        elif area == 1:
            assert target_pos is None
            target_pos = (int(coords[0, 0]), int(coords[0, 1]))
        else:
            raise AssertionError("unexpected maroon component")

    assert object_bbox is not None
    assert target_pos is not None

    object_room = int(room_labels[object_bbox[0], object_bbox[1]])
    target_room = int(room_labels[target_pos[0], target_pos[1]])
    assert object_room > 0 and target_room > 0 and object_room != target_room

    adj: dict[int, list[dict]] = defaultdict(list)
    doors = []
    door_mask = (g != BG) & (g != WALL) & (g != MAROON)
    for r, c in np.argwhere(door_mask):
        candidates = []
        if 0 < c < w - 1:
            left = int(room_labels[r, c - 1])
            right = int(room_labels[r, c + 1])
            if left > 0 and right > 0 and left != right:
                candidates.append((left, right))
        if 0 < r < h - 1:
            up = int(room_labels[r - 1, c])
            down = int(room_labels[r + 1, c])
            if up > 0 and down > 0 and up != down:
                candidates.append((up, down))
        assert len(candidates) == 1
        a, b = candidates[0]
        edge = {"rooms": (int(a), int(b)), "color": int(g[r, c]), "pos": (int(r), int(c))}
        doors.append(edge)
        adj[a].append({"other": int(b), "color": int(g[r, c]), "pos": (int(r), int(c))})
        adj[b].append({"other": int(a), "color": int(g[r, c]), "pos": (int(r), int(c))})

    assert adj[object_room] and adj[target_room]

    return {
        "grid": g,
        "room_labels": room_labels,
        "object_bbox": object_bbox,
        "target_pos": target_pos,
        "object_room": object_room,
        "target_room": target_room,
        "adj": adj,
        "doors": doors,
    }


def _trace_path(start_room: int, target_room: int, adj: dict[int, list[dict]]) -> list[dict]:
    assert len(adj[start_room]) == 1
    assert len(adj[target_room]) == 1

    path = []
    seen = {int(start_room)}
    prev = None
    current = int(start_room)
    while current != int(target_room):
        options = [edge for edge in adj[current] if edge["other"] != prev]
        assert len(options) == 1
        edge = options[0]
        nxt = int(edge["other"])
        assert nxt not in seen
        path.append(edge)
        prev, current = current, nxt
        seen.add(current)
        if current != int(target_room):
            assert len(adj[current]) == 2
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Teleport the 2x2 object through the unique room-to-room door chain.

    Input structure:
    - Grey walls partition the scene into rectangular rooms on a black background.
    - A colored door is a single non-grey wall pixel shared by two adjacent rooms.
    - One room contains a maroon 2x2 object, and a different room contains one maroon target pixel.

    Rule:
    - Follow the unique chain of doors from the object's room to the target room.
    - Remove the original 2x2 object.
    - Draw a 2x2 block at the target pixel using that pixel as the top-left corner.
    - The new block takes the color of the last door on the path.
    - Every used door pixel is recolored maroon.
    """
    info = _analyze(grid)
    g = info["grid"]
    out = g.copy()

    path = _trace_path(info["object_room"], info["target_room"], info["adj"])
    assert path
    final_color = int(path[-1]["color"])

    r0, c0, r1, c1 = info["object_bbox"]
    out[r0 : r1 + 1, c0 : c1 + 1] = BG

    tr, tc = info["target_pos"]
    room_labels = info["room_labels"]
    target_room = int(info["target_room"])
    assert tr + 1 < out.shape[0] and tc + 1 < out.shape[1]
    assert bool((room_labels[tr : tr + 2, tc : tc + 2] == target_room).all())
    out[tr : tr + 2, tc : tc + 2] = final_color

    for edge in path:
        rr, cc = edge["pos"]
        out[rr, cc] = MAROON

    return out


MIN_ROOM_SIZE = 4
ROOM_COUNT_CHOICES = (4, 5, 6)
HOUSE_H_RANGE = (14, 20)
HOUSE_W_RANGE = (14, 20)
HOUSE_MARGIN_RANGE = (0, 3)
MAX_GEN_ATTEMPTS = 500


def _room_height(room: tuple[int, int, int, int]) -> int:
    return int(room[1] - room[0] + 1)


def _room_width(room: tuple[int, int, int, int]) -> int:
    return int(room[3] - room[2] + 1)


def _split_options(room: tuple[int, int, int, int]) -> list[str]:
    opts = []
    if _room_width(room) >= 2 * MIN_ROOM_SIZE + 1:
        opts.append("v")
    if _room_height(room) >= 2 * MIN_ROOM_SIZE + 1:
        opts.append("h")
    return opts


def _split_room(
    room: tuple[int, int, int, int], orientation: str, wall_coord: int
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    r0, r1, c0, c1 = room
    if orientation == "v":
        left = (r0, r1, c0, wall_coord - 1)
        right = (r0, r1, wall_coord + 1, c1)
        return left, right
    top = (r0, wall_coord - 1, c0, c1)
    bottom = (wall_coord + 1, r1, c0, c1)
    return top, bottom


def _shared_wall(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> dict | None:
    ar0, ar1, ac0, ac1 = a
    br0, br1, bc0, bc1 = b

    if ac1 + 2 == bc0 or bc1 + 2 == ac0:
        overlap0 = max(ar0, br0)
        overlap1 = min(ar1, br1)
        if overlap0 + 1 <= overlap1 - 1:
            wall_c = ac1 + 1 if ac1 < bc0 else bc1 + 1
            cells = [(r, wall_c) for r in range(overlap0 + 1, overlap1)]
            return {"orientation": "v", "cells": cells}

    if ar1 + 2 == br0 or br1 + 2 == ar0:
        overlap0 = max(ac0, bc0)
        overlap1 = min(ac1, bc1)
        if overlap0 + 1 <= overlap1 - 1:
            wall_r = ar1 + 1 if ar1 < br0 else br1 + 1
            cells = [(wall_r, c) for c in range(overlap0 + 1, overlap1)]
            return {"orientation": "h", "cells": cells}

    return None


def _hamiltonian_paths(neighbors: dict[int, list[int]], n: int) -> list[list[int]]:
    paths: list[list[int]] = []

    def dfs(path: list[int], used: set[int]) -> None:
        if len(path) == n:
            paths.append(path.copy())
            return
        cur = path[-1]
        next_nodes = neighbors[cur][:]
        random.shuffle(next_nodes)
        for nxt in next_nodes:
            if nxt in used:
                continue
            path.append(nxt)
            used.add(nxt)
            dfs(path, used)
            used.remove(nxt)
            path.pop()

    starts = list(range(n))
    random.shuffle(starts)
    for start in starts:
        dfs([start], {start})
    return paths


def _sample_block_tl(room: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, r1, c0, c1 = room
    row_lo = r0 + 1 if r1 - r0 + 1 >= 4 else r0
    row_hi = r1 - 2 if r1 - r0 + 1 >= 4 else r1 - 1
    col_lo = c0 + 1 if c1 - c0 + 1 >= 4 else c0
    col_hi = c1 - 2 if c1 - c0 + 1 >= 4 else c1 - 1
    return random.randint(row_lo, row_hi), random.randint(col_lo, col_hi)


def _gen_attempt() -> np.ndarray | None:
    room_count = random.choice(ROOM_COUNT_CHOICES)
    house_h = random.randint(*HOUSE_H_RANGE)
    house_w = random.randint(*HOUSE_W_RANGE)
    top_margin = random.randint(*HOUSE_MARGIN_RANGE)
    bottom_margin = random.randint(*HOUSE_MARGIN_RANGE)
    left_margin = random.randint(*HOUSE_MARGIN_RANGE)
    right_margin = random.randint(*HOUSE_MARGIN_RANGE)

    h = house_h + top_margin + bottom_margin
    w = house_w + left_margin + right_margin
    grid = np.full((h, w), BG, dtype=int)

    outer_r0 = top_margin
    outer_c0 = left_margin
    outer_r1 = outer_r0 + house_h - 1
    outer_c1 = outer_c0 + house_w - 1
    grid[outer_r0 : outer_r1 + 1, outer_c0] = WALL
    grid[outer_r0 : outer_r1 + 1, outer_c1] = WALL
    grid[outer_r0, outer_c0 : outer_c1 + 1] = WALL
    grid[outer_r1, outer_c0 : outer_c1 + 1] = WALL

    rooms = [(outer_r0 + 1, outer_r1 - 1, outer_c0 + 1, outer_c1 - 1)]
    while len(rooms) < room_count:
        splittable = [(idx, _split_options(room)) for idx, room in enumerate(rooms) if _split_options(room)]
        if not splittable:
            return None
        room_idx, opts = random.choice(splittable)
        room = rooms.pop(room_idx)
        orientation = random.choice(opts)
        r0, r1, c0, c1 = room
        if orientation == "v":
            wall_c = random.randint(c0 + MIN_ROOM_SIZE, c1 - MIN_ROOM_SIZE)
            grid[r0 - 1 : r1 + 2, wall_c] = WALL
        else:
            wall_r = random.randint(r0 + MIN_ROOM_SIZE, r1 - MIN_ROOM_SIZE)
            grid[wall_r, c0 - 1 : c1 + 2] = WALL
        child_a, child_b = _split_room(room, orientation, wall_c if orientation == "v" else wall_r)
        rooms.extend([child_a, child_b])

    pair_to_wall: dict[frozenset[int], dict] = {}
    neighbors: dict[int, list[int]] = {idx: [] for idx in range(len(rooms))}
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            wall = _shared_wall(rooms[i], rooms[j])
            if wall is None:
                continue
            pair_to_wall[frozenset((i, j))] = wall
            neighbors[i].append(j)
            neighbors[j].append(i)

    paths = _hamiltonian_paths(neighbors, len(rooms))
    if not paths:
        return None
    path = random.choice(paths)
    if random.random() < 0.5:
        path = list(reversed(path))

    door_colors = random.sample(DOOR_COLORS, len(path) - 1)
    door_positions = []
    for color, a, b in zip(door_colors, path, path[1:]):
        wall = pair_to_wall.get(frozenset((a, b)))
        if wall is None or not wall["cells"]:
            return None
        pos = random.choice(wall["cells"])
        grid[pos] = int(color)
        door_positions.append((pos[0], pos[1], int(color)))

    start_room = rooms[path[0]]
    target_room = rooms[path[-1]]
    obj_r, obj_c = _sample_block_tl(start_room)
    tgt_r, tgt_c = _sample_block_tl(target_room)
    grid[obj_r : obj_r + 2, obj_c : obj_c + 2] = MAROON
    grid[tgt_r, tgt_c] = MAROON

    expected = grid.copy()
    expected[obj_r : obj_r + 2, obj_c : obj_c + 2] = BG
    expected[tgt_r : tgt_r + 2, tgt_c : tgt_c + 2] = door_colors[-1]
    for rr, cc, _color in door_positions:
        expected[rr, cc] = MAROON

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
